import toml
import os
# Test BigQuery creds
from google.oauth2 import service_account
from google.cloud import bigquery

secrets_path = os.path.join(os.path.dirname(__file__), "secrets.toml")
with open(secrets_path, "r") as f:

    secrets = toml.load(f)

# ─── Top-level keys ───────────────────────────────────────────
top_level_keys = [
    "ANTHROPIC_API_KEY",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    "FACEBOOK_ACCESS_TOKEN",
    "FACEBOOK_C_USER",
    "FACEBOOK_XS",
]

print("=== Top-Level Keys ===")
for key in top_level_keys:
    value = secrets.get(key)
    if value:
        # Show first 6 and last 4 chars for security
        masked = str(value)[:6] + "..." + str(value)[-4:] if len(str(value)) > 10 else "***"
        print(f"✅ {key}: {masked}")
    else:
        print(f"❌ {key}: MISSING or EMPTY")

# ─── BigQuery credentials ─────────────────────────────────────
bq_keys = [
    "type",
    "project_id",
    "private_key_id",
    "client_email",
    "client_id",
    "auth_uri",
    "token_uri",
    "auth_provider_x509_cert_url",
    "client_x509_cert_url",
    "universe_domain",
]

print("\n=== BigQuery [bq.creds] ===")
bq_creds = secrets.get("bq", {}).get("creds", {})

for key in bq_keys:
    value = bq_creds.get(key)
    if value:
        masked = str(value)[:6] + "..." + str(value)[-4:] if len(str(value)) > 10 else str(value)
        print(f"✅ {key}: {masked}")
    else:
        print(f"❌ {key}: MISSING or EMPTY")

# ─── Special check: private_key format ────────────────────────
private_key = bq_creds.get("private_key", "")
print("\n=== Private Key Format Check ===")
if private_key.startswith("-----BEGIN RSA PRIVATE KEY-----") or \
   private_key.startswith("-----BEGIN PRIVATE KEY-----"):
    print("✅ private_key: Valid PEM format")
else:
    print("❌ private_key: Invalid format — must start with '-----BEGIN PRIVATE KEY-----'")


# Test Anthropic key actually works
import anthropic
client = anthropic.Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])
try:
    client.messages.create(model="claude-opus-4-6", max_tokens=10, messages=[{"role":"user","content":"hi"}])
    print("✅ ANTHROPIC_API_KEY: Live and working")
except Exception as e:
    print(f"❌ ANTHROPIC_API_KEY: {e}")


try:
    creds = service_account.Credentials.from_service_account_info(bq_creds)
    bq_client = bigquery.Client(credentials=creds, project=bq_creds["project_id"])
    list(bq_client.list_datasets())
    print("✅ BigQuery credentials: Live and working")
except Exception as e:
    print(f"❌ BigQuery credentials: {e}")