import requests
import json

# URL of your Flask API
API_URL = "http://127.0.0.1:5000/api/process_transaction"

# High-risk example transaction
payload = {
    "name": "Test User",
    "email": "testuser@example.com",
    "phone": "9999999999",
    "address": "123 Fraud St, Mumbai, India",
    "billing_country": "India",
    "payment_method": "card",
    "amount": 80000,
    "account_age_days": 2,
    "ip_address": "8.8.8.8"  # Simulate mismatch with billing country
}

# Optional: custom headers (User-Agent can affect features)
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Send POST request
response = requests.post(API_URL, data=json.dumps(payload), headers=headers)

# Print result
if response.status_code == 200:
    data = response.json()
    print("Fraud Score:", data.get("fraud_score"))
    print("Decision:", data.get("decision"))
    print("SHAP Explanation:", data.get("explanation"))
else:
    print("Error:", response.status_code, response.text)
