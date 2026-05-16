"""
test_kaggle.py
Run this to debug exactly why Kaggle is failing:
    python test_kaggle.py
"""

import os
import json
import glob
import tempfile
import datetime

KAGGLE_DATASET_SLUG = "corinnedavidemakia/emakia-dataset"   # correct slug
DOWNLOAD_PATH       = "/tmp/kaggle_emakia"

print("\n" + "="*50)
print("KAGGLE CREDENTIAL DIAGNOSTIC")
print("="*50)

# Step 1: Check env vars
username = os.environ.get("KAGGLE_USERNAME")
key      = os.environ.get("KAGGLE_API_TOKEN")

print(f"\n1. Env vars:")
print(f"   KAGGLE_USERNAME  : {'OK: ' + username if username else 'NOT SET (will fall back to kaggle.json)'}")
print(f"   KAGGLE_API_TOKEN : {'OK: set (hidden)' if key else 'NOT SET (will fall back to kaggle.json)'}")

# Step 2: Check ~/.kaggle/kaggle.json — auto-create if missing
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
print(f"\n2. kaggle.json file: {kaggle_json}")
if os.path.exists(kaggle_json):
    with open(kaggle_json) as f:
        data = json.load(f)
    print(f"   File exists")
    print(f"   username : {data.get('username', 'MISSING')}")
    print(f"   key      : {'OK (hidden)' if data.get('key') else 'MISSING'}")
    print(f"   token    : {'OK (hidden)' if data.get('token') else 'MISSING'}")
    mode = oct(os.stat(kaggle_json).st_mode)[-3:]
    print(f"   permissions: {mode} {'OK' if mode == '600' else 'WRONG - run: chmod 600 ~/.kaggle/kaggle.json'}")
else:
    print(f"   File not found - creating from env vars...")
    if username and key:
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"token": key}, f)   # new-style token format
        os.chmod(kaggle_json, 0o600)
        print(f"   Created ~/.kaggle/kaggle.json from env vars (new token format)")
    else:
        print(f"   Cannot create - KAGGLE_USERNAME or KAGGLE_API_TOKEN not set in env")

# Step 3: Authenticate
print(f"\n3. Kaggle API authentication:")
try:
    import kaggle
    kaggle.api.authenticate()
    print("   Authentication successful")
except Exception as e:
    print(f"   Auth failed: {e}")

# Step 4: Check dataset exists
print(f"\n4. Dataset access: {KAGGLE_DATASET_SLUG}")
try:
    import kaggle
    kaggle.api.authenticate()
    results = kaggle.api.dataset_list(user="corinnedavidemakia")
    if results:
        print(f"   Datasets found under corinnedavidemakia:")
        for d in results:
            is_private = getattr(d, 'isPrivate', getattr(d, 'is_private', 'unknown'))
            print(f"     - {d.ref}  private={is_private}")
    else:
        print("   No datasets found under corinnedavidemakia")
except Exception as e:
    print(f"   Dataset list failed: {e}")

# Step 5: Download
print(f"\n5. Download test:")
os.makedirs(DOWNLOAD_PATH, exist_ok=True)
try:
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET_SLUG,
        path=DOWNLOAD_PATH,
        unzip=True,
        quiet=False,
    )
    print(f"   Download succeeded")
except Exception as e:
    print(f"   Download failed: {e}")

# Step 6: Find politics-tweets.json via glob
print(f"\n6. Searching for politics-tweets.json:")
matches = glob.glob(os.path.join(DOWNLOAD_PATH, "**", "politics-tweets.json"), recursive=True)
if matches:
    print(f"   Found: {matches[0]}")
else:
    all_files = glob.glob(os.path.join(DOWNLOAD_PATH, "**", "*"), recursive=True)
    print(f"   Not found. All downloaded files:")
    for f in all_files:
        print(f"     {os.path.relpath(f, DOWNLOAD_PATH)}")

# Step 7: Load and preview JSON
print(f"\n7. JSON load and preview:")
if matches:
    try:
        with open(matches[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else data.get("root", [])
        print(f"   Loaded {len(records)} records")
        if records:
            print(f"   First record keys : {list(records[0].keys())}")
            print(f"   Sample text       : {str(records[0].get('text', ''))[:120]}")
    except Exception as e:
        print(f"   JSON load failed: {e}")
else:
    print("   Skipped - file not found in step 6")

# ── NEW ──────────────────────────────────────────────────────────────────────
# Step 8: WRITE TEST — create a tiny test dataset on Kaggle
print(f"\n8. Write test (publish a small dataset to Kaggle):")
try:
    import kaggle
    kaggle.api.authenticate()

    # Use username from env var or fall back to kaggle.json
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not username and os.path.exists(kaggle_json_path):
        with open(kaggle_json_path) as f:
            _creds = json.load(f)
        username = _creds.get("username", "corinnedavidemakia")
        print(f"   Username from kaggle.json: {username}")

    # Build a temp folder with one CSV and a dataset-metadata.json
    run_id    = datetime.datetime.utcnow().strftime("test_%Y%m%d_%H%M%S")
    slug      = f"emakia-write-test-{run_id.replace('_', '-').lower()}"
    tmp_dir   = tempfile.mkdtemp()

    # 1) Create a tiny CSV payload
    csv_path = os.path.join(tmp_dir, "write_test.csv")
    with open(csv_path, "w") as f:
        f.write("id,message,timestamp\n")
        f.write(f"1,write_test_ok,{datetime.datetime.utcnow().isoformat()}Z\n")

    # 2) Create dataset-metadata.json (required by Kaggle API)
    meta = {
        "title":    f"Emakia Write Test {run_id}",
        "id":       f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(tmp_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f)

    print(f"   Temp folder  : {tmp_dir}")
    print(f"   Dataset slug : {username}/{slug}")

    # 3) Create the dataset (new dataset, not an update)
    kaggle.api.dataset_create_new(
        folder=tmp_dir,
        public=True,
        quiet=False,
        convert_to_csv=False,
        dir_mode="zip",
    )
    print(f"   ✅ Write succeeded!")
    print(f"   Check it at  : https://www.kaggle.com/datasets/{username}/{slug}")

except Exception as e:
    print(f"   ❌ Write failed: {e}")
    print(f"   This is the error your app will hit when publishing benchmark results.")
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*50)
print("DONE")
print("="*50)
