import json
import os
import re

# ── 1. Load the downloaded JSON key ──────────────────────────────────────────
bq_key_path = os.path.expanduser("~/bq-key.json")

if not os.path.exists(bq_key_path):
    print(f"❌ Cannot find {bq_key_path}")
    print("   Run: gcloud iam service-accounts keys create ~/bq-key.json --iam-account=bq-streamlit-sa@emakia.iam.gserviceaccount.com")
    exit(1)

with open(bq_key_path, "r") as f:
    creds = json.load(f)

print("✅ Loaded bq-key.json")

# ── 2. Build valid TOML block ─────────────────────────────────────────────────
# CRITICAL: private_key newlines must be escaped as \n on a single line
private_key = creds["private_key"].replace("\n", "\\n")

toml_block = f"""[bq.creds]
type = "{creds['type']}"
project_id = "{creds['project_id']}"
private_key_id = "{creds['private_key_id']}"
private_key = "{private_key}"
client_email = "{creds['client_email']}"
client_id = "{creds['client_id']}"
auth_uri = "{creds['auth_uri']}"
token_uri = "{creds['token_uri']}"
auth_provider_x509_cert_url = "{creds['auth_provider_x509_cert_url']}"
client_x509_cert_url = "{creds['client_x509_cert_url']}"
universe_domain = "{creds.get('universe_domain', 'googleapis.com')}"
"""

# ── 3. Patch secrets.toml ─────────────────────────────────────────────────────
secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")

# Try current dir's .streamlit first, then look up
if not os.path.exists(secrets_path):
    secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")

if not os.path.exists(secrets_path):
    print(f"❌ Cannot find secrets.toml at {secrets_path}")
    print("\n📋 Paste this into your secrets.toml manually:\n")
    print(toml_block)
    exit(1)

with open(secrets_path, "r") as f:
    content = f.read()

# Remove old [bq.creds] block if it exists
content = re.sub(r'\[bq\.creds\].*?(?=\n\[|\Z)', '', content, flags=re.DOTALL).strip()

# Append new valid block
content = content + "\n\n" + toml_block

with open(secrets_path, "w") as f:
    f.write(content)

print(f"✅ secrets.toml updated at: {secrets_path}")
print("\n📋 New [bq.creds] block written:\n")
print(toml_block)

# ── 4. Quick validation ───────────────────────────────────────────────────────
try:
    import toml
    with open(secrets_path, "r") as f:
        parsed = toml.load(f)
    bq = parsed.get("bq", {}).get("creds", {})
    print("✅ TOML is valid — no syntax errors")
    print(f"   project_id  : {bq.get('project_id')}")
    print(f"   client_email: {bq.get('client_email')}")
    pk = bq.get("private_key", "")
    if pk.startswith("-----BEGIN PRIVATE KEY-----"):
        print("✅ private_key format: valid PEM")
    else:
        print("❌ private_key format: INVALID")
except Exception as e:
    print(f"❌ TOML still invalid: {e}")