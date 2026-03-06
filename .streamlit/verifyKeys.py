import toml
import os
import requests

# ── Load secrets.toml ─────────────────────────────────────────────────────────
secrets_path = os.path.join(os.path.dirname(__file__), "secrets.toml")
with open(secrets_path, "r") as f:
    secrets = toml.load(f)

print("=" * 55)
print("       🔐 API KEY VERIFICATION TOOL")
print("=" * 55)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ANTHROPIC
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 1. ANTHROPIC ===")
try:
    import anthropic
    client = anthropic.Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])
    client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}]
    )
    print("✅ ANTHROPIC_API_KEY: Live and working")
except Exception as e:
    print(f"❌ ANTHROPIC_API_KEY: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GEMINI
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 2. GEMINI ===")
try:
    import google.generativeai as genai
    genai.configure(api_key=secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("hi")
    _ = response.text  # force evaluation
    print("✅ GEMINI_API_KEYY: Live and working")
except KeyError as e:
    print(f"❌ Missing key in secrets.toml: {e}")
except Exception as e:
    print(f"❌ GEMINI_API_KEY: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. REDDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 3. REDDIT ===")
try:
    client_id     = secrets["REDDIT_CLIENT_ID"]
    client_secret = secrets["REDDIT_CLIENT_SECRET"]
    user_agent    = secrets["REDDIT_USER_AGENT"]

    # Get OAuth token from Reddit
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {"grant_type": "client_credentials"}
    headers = {"User-Agent": user_agent}

    r = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=auth,
        data=data,
        headers=headers,
        timeout=10
    )

    if r.status_code == 200 and "access_token" in r.json():
        token = r.json()["access_token"]
        masked = token[:6] + "..." + token[-4:]
        print(f"✅ REDDIT_CLIENT_ID     : {client_id[:6]}...{client_id[-4:]}")
        print(f"✅ REDDIT_CLIENT_SECRET : {client_secret[:4]}****")
        print(f"✅ REDDIT_USER_AGENT    : {user_agent}")
        print(f"✅ Reddit OAuth Token   : {masked}")

        # Test actual API call
        api_headers = {**headers, "Authorization": f"bearer {token}"}
        test = requests.get(
            "https://oauth.reddit.com/r/python/hot?limit=1",
            headers=api_headers,
            timeout=10
        )
        if test.status_code == 200:
            print("✅ Reddit API call      : Connected successfully")
        else:
            print(f"⚠️  Reddit API call      : Status {test.status_code}")
    else:
        print(f"❌ Reddit auth failed   : {r.status_code} - {r.text}")

except KeyError as e:
    print(f"❌ Missing key in secrets.toml: {e}")
except Exception as e:
    print(f"❌ Reddit connection error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FACEBOOK
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 4. FACEBOOK ===")
try:
    access_token = secrets["FACEBOOK_ACCESS_TOKEN"]
    c_user       = secrets["FACEBOOK_C_USER"]
    xs           = secrets["FACEBOOK_XS"]

    # Verify access token via Graph API
    r = requests.get(
        "https://graph.facebook.com/me",
        params={"access_token": access_token, "fields": "id,name"},
        timeout=10
    )

    if r.status_code == 200:
        data = r.json()
        print(f"✅ FACEBOOK_ACCESS_TOKEN: Valid")
        print(f"   → Logged in as: {data.get('name', 'Unknown')} (ID: {data.get('id', 'Unknown')})")
    else:
        err = r.json().get("error", {})
        print(f"❌ FACEBOOK_ACCESS_TOKEN: {err.get('message', r.text)}")

    # Check token debug info (expiry, scopes)
    debug = requests.get(
        "https://graph.facebook.com/debug_token",
        params={
            "input_token": access_token,
            "access_token": access_token  # use same token as app token for basic check
        },
        timeout=10
    )
    if debug.status_code == 200:
        ddata = debug.json().get("data", {})
        is_valid = ddata.get("is_valid", False)
        expires  = ddata.get("expires_at", "never")
        scopes   = ddata.get("scopes", [])
        print(f"   → Token valid    : {is_valid}")
        print(f"   → Expires at     : {expires}")
        print(f"   → Scopes         : {', '.join(scopes) if scopes else 'none returned'}")

    # Just confirm the cookie values are present (can't verify server-side)
    print(f"✅ FACEBOOK_C_USER      : {str(c_user)[:6]}...{str(c_user)[-4:]}")
    print(f"✅ FACEBOOK_XS          : {str(xs)[:6]}...{str(xs)[-4:]}")
    print("ℹ️  Note: c_user & xs are session cookies — verified as present in secrets")

except KeyError as e:
    print(f"❌ Missing key in secrets.toml: {e}")
except Exception as e:
    print(f"❌ Facebook connection error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BIGQUERY
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 5. BIGQUERY ===")
try:
    from google.oauth2 import service_account
    from google.cloud import bigquery

    bq_creds = secrets.get("bq", {}).get("creds", {})

    if not bq_creds:
        print("❌ [bq.creds] section not found in secrets.toml")
    else:
        creds = service_account.Credentials.from_service_account_info(bq_creds)
        bq_client = bigquery.Client(credentials=creds, project=bq_creds["project_id"])
        datasets = list(bq_client.list_datasets())
        print(f"✅ BigQuery credentials : Connected to project '{bq_creds['project_id']}'")
        print(f"   → client_email      : {bq_creds.get('client_email', 'N/A')}")
        if datasets:
            print(f"   → Datasets found    : {[d.dataset_id for d in datasets]}")
        else:
            print("   → No datasets found (but connection works)")

except Exception as e:
    print(f"❌ BigQuery connection error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("       ✅ VERIFICATION COMPLETE")
print("=" * 55)