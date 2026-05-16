"""
1,600-row evaluation runner with Kaggle dataset publishing.

Self-contained module. The Streamlit app injects the classifier callables
so this file has no Streamlit / ADK dependency and can be unit-tested.

Resumability:
    Every CHECKPOINT_INTERVAL rows the partial results are flushed to disk
    so a refresh / crash / Ctrl+C doesn't lose work. A run is identified
    by `run_id`; resuming with the same id picks up where you left off.

Publishing:
    Uses the kaggle Python API authenticated via the already-bridged
    KAGGLE_USERNAME / KAGGLE_API_TOKEN env vars. Creates a NEW dataset under
    your account, or pushes a NEW VERSION to an existing one.
"""
import os
import json
import time
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional


CHECKPOINT_DIR = Path(os.environ.get("EMAKIA_EVAL_DIR", "/tmp/emakia_eval"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_INTERVAL = 10   # flush every N rows

# All combinations the harness knows how to run.
# Multi-agent pipeline is ADK-powered → cloud Gemini only (no local backend).
# Direct Gemma 4 classifier supports both cloud (AI Studio) and local (llama.cpp).
KNOWN_COMBINATIONS = ["multiagent_cloud", "direct_cloud", "direct_local"]


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────
def prepare_eval_sample(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    label_map: dict,
    n_rows: int = 1600,
    stratify: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n_rows from a labeled dataset, optionally stratified by class.
    Normalises labels via label_map. Returns columns:
        tweet_id, content, ground_truth
    """
    work = df[[text_col, label_col]].copy()
    work = work.dropna(subset=[text_col, label_col])
    work["ground_truth"] = work[label_col].map(label_map)
    work = work.dropna(subset=["ground_truth"])

    if len(work) == 0:
        raise ValueError(
            f"No rows survived label mapping. Check label_map={label_map} "
            f"against actual values in column '{label_col}'."
        )

    if stratify and work["ground_truth"].nunique() > 1:
        groups = []
        n_classes = work["ground_truth"].nunique()
        per_class = n_rows // n_classes
        for _, group in work.groupby("ground_truth"):
            take = min(per_class, len(group))
            groups.append(group.sample(n=take, random_state=seed))
        sample = pd.concat(groups, ignore_index=True)
        # Top up with random rows if minority classes were too small
        if len(sample) < n_rows:
            remaining = work.drop(sample.index, errors="ignore")
            top_up_n = min(n_rows - len(sample), len(remaining))
            if top_up_n > 0:
                top_up = remaining.sample(n=top_up_n, random_state=seed)
                sample = pd.concat([sample, top_up], ignore_index=True)
        sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        sample = work.sample(
            n=min(n_rows, len(work)), random_state=seed
        ).reset_index(drop=True)

    sample["tweet_id"] = ["row_" + str(i).zfill(5) for i in range(len(sample))]
    sample["content"] = sample[text_col].astype(str)
    return sample[["tweet_id", "content", "ground_truth"]]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────
def _checkpoint_path(run_id: str) -> Path:
    return CHECKPOINT_DIR / f"eval_{run_id}.csv"


def _meta_path(run_id: str) -> Path:
    return CHECKPOINT_DIR / f"eval_{run_id}.meta.json"


def _sample_path(run_id: str) -> Path:
    return CHECKPOINT_DIR / f"eval_{run_id}.sample.csv"


def save_checkpoint(
    run_id: str, completed: list, sample_df: pd.DataFrame, config: dict
) -> Path:
    """Persist completed rows + run config so we can resume."""
    cp = pd.DataFrame(completed)
    cp_path = _checkpoint_path(run_id)
    cp.to_csv(cp_path, index=False)

    meta = {
        "run_id":       run_id,
        "config":       config,
        "n_completed":  len(completed),
        "n_total":      len(sample_df),
        "last_updated": datetime.utcnow().isoformat() + "Z",
    }
    _meta_path(run_id).write_text(json.dumps(meta, indent=2))

    sp = _sample_path(run_id)
    if not sp.exists():
        sample_df.to_csv(sp, index=False)
    return cp_path


def load_checkpoint(run_id: str):
    """Return (completed_rows, original_sample, run_config) or ([], None, {})."""
    cp = _checkpoint_path(run_id)
    sp = _sample_path(run_id)
    mp = _meta_path(run_id)
    if not cp.exists() or not sp.exists():
        return [], None, {}
    completed = pd.read_csv(cp).to_dict(orient="records")
    sample = pd.read_csv(sp)
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    return completed, sample, meta.get("config", {})


def list_runs() -> list:
    """List all available runs sorted by last_updated, most recent first."""
    runs = []
    for mp in CHECKPOINT_DIR.glob("eval_*.meta.json"):
        try:
            meta = json.loads(mp.read_text())
            runs.append(meta)
        except Exception:
            pass
    runs.sort(key=lambda m: m.get("last_updated", ""), reverse=True)
    return runs


def delete_run(run_id: str):
    """Remove checkpoint files for a run."""
    for p in (_checkpoint_path(run_id), _meta_path(run_id), _sample_path(run_id)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Eval loop
# ─────────────────────────────────────────────────────────────────────────────
def _multiagent_to_label(toxicity_text: str) -> str:
    """Derive a normalised label from the multi-agent toxicity output."""
    tox = str(toxicity_text or "").lower()
    if "non-toxic" in tox or "not toxic" in tox:
        return "neutral"
    if "toxic" in tox:
        return "harassment"
    return "unknown"


def run_eval(
    sample_df: pd.DataFrame,
    run_id: str,
    combinations: list,
    classify_direct: Callable[[str], dict],   # direct Gemma 4 (uses backend toggle)
    run_multiagent: Callable[[list], list],   # multi-agent runner
    set_backend: Callable[[str], None],       # sets gemma backend ("cloud"|"local")
    config: dict,
    progress_cb: Optional[Callable[[int, int, dict], None]] = None,
    resume_from: Optional[list] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> list:
    """
    Run the eval over sample_df, computing the requested combinations.

    Each row in the returned list has:
        tweet_id, content, ground_truth,
        <combo>_label, <combo>_score, <combo>_severity, <combo>_reason   (for direct_*)
        <combo>_toxicity, <combo>_bias, <combo>_misinformation, <combo>_label  (for multiagent_*)

    `resume_from` is a list of already-completed result dicts; rows in the
    sample whose tweet_id is already there are skipped.

    `stop_flag` is a no-arg callable returning True if the user requested a pause.
    The function checkpoints and returns when stop_flag() goes True.
    """
    completed = list(resume_from) if resume_from else []
    completed_ids = {r["tweet_id"] for r in completed}

    pending = sample_df[~sample_df["tweet_id"].isin(completed_ids)].reset_index(drop=True)
    total = len(sample_df)

    for _, row in pending.iterrows():
        if stop_flag and stop_flag():
            save_checkpoint(run_id, completed, sample_df, config)
            return completed

        text = str(row["content"])
        out = {
            "tweet_id":     row["tweet_id"],
            "content":      text,
            "ground_truth": row["ground_truth"],
        }

        for combo in combinations:
            try:
                if combo == "direct_cloud":
                    set_backend("cloud")
                    res = classify_direct(text)
                    out[f"{combo}_label"]    = res.get("label", "error")
                    out[f"{combo}_score"]    = res.get("score", 0.0)
                    out[f"{combo}_severity"] = res.get("severity", "")
                    out[f"{combo}_reason"]   = res.get("reason", "")

                elif combo == "direct_local":
                    set_backend("local")
                    res = classify_direct(text)
                    out[f"{combo}_label"]    = res.get("label", "error")
                    out[f"{combo}_score"]    = res.get("score", 0.0)
                    out[f"{combo}_severity"] = res.get("severity", "")
                    out[f"{combo}_reason"]   = res.get("reason", "")

                elif combo == "multiagent_cloud":
                    batch = run_multiagent(
                        [{"content": text, "title": row["tweet_id"]}]
                    )
                    if batch:
                        r = batch[0]
                        out[f"{combo}_toxicity"]       = r.get("toxicity", "")
                        out[f"{combo}_bias"]           = r.get("bias", "")
                        out[f"{combo}_misinformation"] = r.get("misinformation", "")
                        out[f"{combo}_label"]          = _multiagent_to_label(
                            r.get("toxicity", "")
                        )
                    else:
                        out[f"{combo}_label"] = "error"
                        out[f"{combo}_error"] = "empty multi-agent response"

                else:
                    out[f"{combo}_label"] = "error"
                    out[f"{combo}_error"] = f"unknown combination '{combo}'"

            except Exception as e:
                out[f"{combo}_label"] = "error"
                out[f"{combo}_error"] = str(e)[:200]

        completed.append(out)

        if (len(completed) % CHECKPOINT_INTERVAL) == 0:
            save_checkpoint(run_id, completed, sample_df, config)

        if progress_cb:
            progress_cb(len(completed), total, out)

    save_checkpoint(run_id, completed, sample_df, config)
    return completed


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame, combinations: list) -> pd.DataFrame:
    """Per-combination accuracy + harassment-class precision/recall/F1."""
    rows = []
    for combo in combinations:
        col = f"{combo}_label"
        if col not in df.columns:
            continue
        valid = df[df[col].isin(["harassment", "neutral"])]
        n_err = len(df) - len(valid)
        if len(valid) == 0:
            rows.append({
                "combination": combo,
                "n_evaluated": 0,
                "n_errors":    n_err,
                "accuracy":    None,
                "precision":   None,
                "recall":      None,
                "f1":          None,
            })
            continue
        gt = valid["ground_truth"]
        pred = valid[col]
        accuracy = (gt == pred).mean()
        tp = ((gt == "harassment") & (pred == "harassment")).sum()
        fp = ((gt == "neutral") & (pred == "harassment")).sum()
        fn = ((gt == "harassment") & (pred == "neutral")).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        rows.append({
            "combination": combo,
            "n_evaluated": int(len(valid)),
            "n_errors":    int(n_err),
            "accuracy":    round(float(accuracy), 4),
            "precision":   round(float(precision), 4),
            "recall":      round(float(recall), 4),
            "f1":          round(float(f1), 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle publishing
# ─────────────────────────────────────────────────────────────────────────────
def publish_to_kaggle(
    files: list,           # list of (src_path, dest_filename) tuples
    slug: str,             # "owner/dataset-name"
    title: str,
    description: str,
    license_name: str = "CC0-1.0",
    new_version: bool = False,
    version_notes: str = "Updated results",
):
    """
    Create or update a Kaggle dataset under the authenticated user's account.

    Returns (ok: bool, message: str). On success, message is the dataset URL.
    """
    try:
        import kaggle
        kaggle.api.authenticate()
    except Exception as e:
        return False, f"Kaggle auth failed: {e}"

    # Validate slug owner matches the authenticated user (helpful early error)
    me = os.environ.get("KAGGLE_USERNAME", "").strip().lower()
    owner = slug.split("/")[0].lower() if "/" in slug else ""
    if me and owner and me != owner:
        return False, (
            f"Slug owner '{owner}' does not match authenticated user '{me}'. "
            f"Use '{me}/<dataset-name>' as the slug."
        )

    # Kaggle converts underscores to hyphens in slugs — sanitize so the
    # returned URL matches what Kaggle actually creates.
    if "/" in slug:
        slug_owner, slug_name = slug.split("/", 1)
        slug = f"{slug_owner}/{slug_name.replace('_', '-').lower()}"

    upload_dir = CHECKPOINT_DIR / f"_upload_{int(time.time())}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    try:
        for src, dest_name in files:
            shutil.copy(src, upload_dir / dest_name)

        metadata = {
            "title":       title[:50],   # Kaggle caps title length
            "id":          slug,
            "licenses":    [{"name": license_name}],
            "description": description,
        }
        (upload_dir / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )

        if new_version:
            kaggle.api.dataset_create_version(
                folder=str(upload_dir),
                version_notes=version_notes,
                public=True,
                quiet=False,
            )
        else:
            kaggle.api.dataset_create_new(
                folder=str(upload_dir),
                public=True,
                quiet=False,
            )
        return True, f"https://www.kaggle.com/datasets/{slug}"

    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "already exists" in lower or "409" in msg:
            return False, (
                f"Dataset `{slug}` already exists on Kaggle. "
                "Tick **'Update existing dataset (new version)'** and retry."
            )
        if "403" in msg or "forbidden" in lower:
            return False, (
                "Kaggle returned 403. The slug owner must match your "
                "authenticated KAGGLE_USERNAME, and you must accept the "
                "API terms at kaggle.com/settings."
            )
        return False, f"Upload failed: {msg}"
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)
