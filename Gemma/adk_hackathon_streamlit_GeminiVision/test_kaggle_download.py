#!/usr/bin/env python3
"""
Standalone Kaggle download test — no Streamlit, no app code, no cache.

Usage (from project root):
    python test_kaggle_download.py "datasets/istiyaque6ty3/facebook-post-reactions?utm_source=copilot.com"

Auto-loads credentials in this priority order:
    1. Existing env vars (KAGGLE_USERNAME / KAGGLE_API_TOKEN)
    2. ./.streamlit/secrets.toml (handles single AND double quotes)
    3. ~/.kaggle/kaggle.json
"""
import os
import sys
import json
import glob
import tempfile
import traceback
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Credentials
# ─────────────────────────────────────────────────────────────────────────────
def _load_secrets_toml(path: Path) -> dict:
    """Parse a TOML file. Uses tomllib on Python 3.11+, otherwise a tiny
    line-based parser sufficient for `KEY = "value"` and `KEY = 'value'`."""
    if not path.exists():
        return {}
    try:
        try:
            import tomllib  # 3.11+
            with open(path, "rb") as fh:
                return tomllib.load(fh)
        except ImportError:
            pass
        # Fallback parser — handles both quote styles
        out = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            # Strip matching outer quotes (either kind)
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            out[k.strip()] = v
        return out
    except Exception:
        return {}


def _load_kaggle_json() -> dict:
    p = Path.home() / ".kaggle" / "kaggle.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def ensure_credentials():
    """Returns (username, key, source). Empty strings if not found."""
    u = os.environ.get("KAGGLE_USERNAME", "").strip()
    k = os.environ.get("KAGGLE_KEY", "").strip()
    if u and k:
        return u, k, "environment variables"

    for candidate in (
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
    ):
        s = _load_secrets_toml(candidate)
        if s.get("KAGGLE_USERNAME") and s.get("KAGGLE_KEY"):
            os.environ["KAGGLE_USERNAME"] = str(s["KAGGLE_USERNAME"])
            os.environ["KAGGLE_KEY"] = str(s["KAGGLE_KEY"])
            return os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"], str(candidate)

    kj = _load_kaggle_json()
    if kj.get("username") and kj.get("key"):
        os.environ["KAGGLE_USERNAME"] = str(kj["username"])
        os.environ["KAGGLE_KEY"] = str(kj["key"])
        return os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"], "~/.kaggle/kaggle.json"

    return "", "", ""


def normalise_slug(raw: str) -> str:
    s = raw.strip()
    for prefix in (
        "https://www.kaggle.com/datasets/",
        "https://kaggle.com/datasets/",
        "http://www.kaggle.com/datasets/",
        "http://kaggle.com/datasets/",
        "www.kaggle.com/datasets/",
        "kaggle.com/datasets/",
    ):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.split("?")[0].split("#")[0].strip("/")
    if s.startswith("datasets/"):
        s = s[len("datasets/"):]
    parts = s.split("/")
    if len(parts) > 2:
        s = "/".join(parts[:2])
    return s


def section(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_kaggle_download.py <slug-or-url>")
        sys.exit(1)
    raw = sys.argv[1]

    section("Step 1: Credentials")
    user, key, source = ensure_credentials()
    print(f"KAGGLE_USERNAME: {user or '(NOT FOUND)'}")
    print(f"KAGGLE_KEY:      "
          f"{'(set, ' + str(len(key)) + ' chars)' if key else '(NOT FOUND)'}")
    print(f"Source:          {source or '(none)'}")
    if not user or not key:
        print()
        print("❌ No credentials found. Looked in:")
        print("   1. env vars KAGGLE_USERNAME / KAGGLE_KEY")
        print(f"   2. {Path.cwd() / '.streamlit' / 'secrets.toml'}")
        print(f"   3. {Path.home() / '.kaggle' / 'kaggle.json'}")
        print()
        print("Diagnostics:")
        sec = Path.cwd() / ".streamlit" / "secrets.toml"
        if sec.exists():
            kag_lines = [l for l in sec.read_text().splitlines() if "KAGGLE" in l]
            print(f"   secrets.toml exists. Lines containing 'KAGGLE':")
            for l in kag_lines:
                if "KAGGLE_KEY" in l and "=" in l:
                    k_part, v_part = l.split("=", 1)
                    print(f"     {k_part}= <redacted, raw value length {len(v_part.strip())} chars>")
                else:
                    print(f"     {l}")
        else:
            print(f"   {sec} does NOT exist.")
            print(f"   Current dir: {Path.cwd()}")
        sys.exit(2)

    section("Step 2: Slug normalisation")
    slug = normalise_slug(raw)
    print(f"Raw input:  {raw!r}")
    print(f"Normalised: {slug!r}")
    if "/" not in slug or slug.count("/") != 1:
        print("❌ Normalised slug doesn't look like 'owner/name'.")
        sys.exit(3)

    section("Step 3: Kaggle API auth")
    try:
        import kaggle
        kaggle.api.authenticate()
        print("✅ Authenticated.")
    except Exception as e:
        print(f"❌ Auth failed: {e}")
        traceback.print_exc()
        sys.exit(4)

    section("Step 4: List files via API (no download)")
    try:
        files = kaggle.api.dataset_list_files(slug).files
        print(f"✅ {len(files)} file(s) reported by API:")
        for f in files:
            # The Kaggle Python client has renamed this attribute over time:
            #   older versions: f.totalBytes
            #   newer versions: f.total_bytes  (and/or f.size)
            size = (
                getattr(f, "total_bytes", None)
                or getattr(f, "totalBytes", None)
                or getattr(f, "size", None)
            )
            name = getattr(f, "name", None) or getattr(f, "ref", "(unnamed)")
            if isinstance(size, (int, float)):
                print(f"   - {name}  ({int(size):,} bytes)")
            else:
                print(f"   - {name}  (size: {size!r})")
    except Exception as e:
        msg = str(e)
        print(f"❌ list_files failed: {msg}")
        if "403" in msg or "forbidden" in msg.lower():
            print(f"\n   Visit https://www.kaggle.com/datasets/{slug} and click "
                  f"'I Understand and Accept', then re-run.")
        elif "404" in msg or "not found" in msg.lower():
            print(f"\n   Dataset doesn't exist. Verify "
                  f"https://www.kaggle.com/datasets/{slug}")
        sys.exit(5)

    section("Step 5: Download + unzip")
    download_dir = tempfile.mkdtemp(prefix="kgl_test_")
    print(f"Target: {download_dir}")
    try:
        kaggle.api.dataset_download_files(slug, path=download_dir, unzip=True, quiet=False)
        print("✅ Download succeeded.")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        traceback.print_exc()
        sys.exit(6)

    section("Step 6: Files on disk")
    all_paths = sorted(
        p for p in glob.glob(os.path.join(download_dir, "**", "*"), recursive=True)
        if os.path.isfile(p)
    )
    if not all_paths:
        print("❌ Nothing on disk after unzip.")
        sys.exit(7)
    for p in all_paths:
        size = os.path.getsize(p)
        rel = os.path.relpath(p, download_dir)
        with open(p, "rb") as fh:
            head = fh.read(120)
        try:
            head_text = head.decode("utf-8", errors="replace")
            head_text = head_text.replace("\n", "\\n").replace("\t", "\\t")
        except Exception:
            head_text = repr(head)
        print(f"\n📄 {rel}")
        print(f"   {size:,} bytes ({size / 1024 / 1024:.1f} MB)")
        print(f"   First 120 bytes: {head_text[:120]}")

    section("Step 7: Read largest file as CSV (3-row peek)")
    biggest = max(all_paths, key=os.path.getsize)
    print(f"File: {os.path.basename(biggest)}")
    try:
        import pandas as pd
        df = pd.read_csv(biggest, nrows=3, on_bad_lines="skip", low_memory=False)
        print(f"✅ Parsed. {len(df.columns)} columns:")
        for c in df.columns:
            print(f"     - {c}")
        print("\nFirst 3 rows:")
        print(df.to_string(max_colwidth=80))

        cols_lower = {c.lower() for c in df.columns}
        text_cands = cols_lower & {
            "text", "tweet", "content", "post", "message", "body", "comment",
            "status", "title", "description",
        }
        label_cands = cols_lower & {
            "label", "class", "category", "harassment", "toxic", "is_toxic",
            "sentiment", "rating",
        }
        print()
        print(f"Text-like columns:  {sorted(text_cands) or 'NONE'}")
        print(f"Label-like columns: {sorted(label_cands) or 'NONE'}")
    except Exception as e:
        print(f"❌ pandas failed: {e}")

    section("Step 8: Verdict")
    print(f"Slug:           {slug}")
    print(f"Files:          {len(all_paths)}")
    print(f"Local path:     {download_dir}")
    print(f"Cleanup:        rm -rf {download_dir}")


if __name__ == "__main__":
    main()
