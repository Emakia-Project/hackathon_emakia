import requests

r = requests.get("https://graph.facebook.com/oauth/access_token", params={
    "grant_type": "fb_exchange_token",
    "client_id": "541684832346665",
    "client_secret": "1c118e151f44718c7a3bf4f40ce17de9",
    "fb_exchange_token": "EAAHsqNQDIikBQ5M4qHbuU3QLaPzPLUZBC7JRKki28SBWoVX14GZCx6IZBZBAKhaMKdytJ5ZCg9w2oPGS3ScPCL2cXooGCcsqLZAPmF2MRag8ZBatFa84OwlOExU0EDUWpPSBkS9LYh2A7XB3tD4xEGIbMJCdZCbE2aOqFY1Wzbv7OLq3URLVbdc9ASwm9bd98NZCNS3JJzV9zMPpj65a1lmZB4Jzq0qcPJm2byFK0kcR0SuPTKSIfJztO6wf5ZAxxKWt8ZBZCtZBMvmTQFM7ZA6kxoGvsZBOAMEZD"
})
print(r.json())
# → {"access_token": "EAABsb...", "token_type": "bearer", "expires_in": 5183944}