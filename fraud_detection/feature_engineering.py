import requests

class FeatureEngineer:
    def _init_(self):
        pass

    def extract_features(self, transaction_data: dict) -> dict:
        """
        Convert raw transaction input into model-ready features.
        Dynamically includes account age, past transaction count, and transaction speed.
        """
        features = {}

        # --- Amount features ---
        amount = float(transaction_data.get('amount', 0))
        features['TransactionAmt'] = amount
        features['is_high_amount'] = int(amount > 50000)
        features['is_very_high_amount'] = int(amount > 100000)

        # --- Device type ---
        user_agent = transaction_data.get('user_agent', '').lower()
        if 'mobile' in user_agent:
            features['DeviceType'] = 1
        elif 'tablet' in user_agent:
            features['DeviceType'] = 2
        else:
            features['DeviceType'] = 0

        # --- Account age ---
        account_age_days = int(transaction_data.get('account_age_days', 0))
        features['account_age_days'] = account_age_days
        features['is_new_account'] = int(account_age_days < 7)

        # --- Email features ---
        email = transaction_data.get('email', '').lower()
        disposable_domains = ['10minutemail.com', 'tempmail.org', 'guerrillamail.com']
        email_domain = email.split('@')[-1] if '@' in email else ''
        features['is_disposable_email'] = int(email_domain in disposable_domains)
        features['is_freemail'] = int(email_domain in ['gmail.com', 'yahoo.com', 'hotmail.com'])

        # --- IP vs Billing Country ---
        # --- IP vs Billing Country (Improved Logic) ---
        billing_country = str(transaction_data.get('billing_country', 'Unknown')).strip().lower()
        ip_address = transaction_data.get('ip_address', '')
        ip_country = self.get_ip_country(ip_address)

        print(f"🌍 IP Address: {ip_address}, Detected Country: {ip_country}, Billing Country: {billing_country}")

        # Normalize both sides for fair comparison
        ip_country_str = str(ip_country).strip().lower()

        # India (and IN) should be treated as same
        if ip_country_str in ['in', 'india'] and billing_country in ['in', 'india']:
            features['ip_country_mismatch'] = 0
        elif ip_country_str == "unknown":
            features['ip_country_mismatch'] = 0   # treat local dev IPs as safe
        else:
            features['ip_country_mismatch'] = 1

        # --- Name / Email mismatch ---
        name = transaction_data.get('name', '').lower().replace(' ', '')
        features['name_email_mismatch'] = int(name not in email)

        # --- Time & activity features ---
        

        # --- New dynamic behavioral features ---
        features['transactions_last_hour'] = int(transaction_data.get('transactions_last_hour', 0))
        features['past_transaction_count'] = int(transaction_data.get('past_transaction_count', 0))
        features['transaction_speed'] = float(transaction_data.get('transaction_speed', 0.0))

        # --- Payment method ---
        payment_method = transaction_data.get('payment_method', '').lower()
        features['payment_method_card'] = int(payment_method == 'card')
        features['payment_method_paypal'] = int(payment_method == 'paypal')
        features['payment_method_crypto'] = int(payment_method == 'crypto')
        features['payment_method_gpay'] = int(payment_method == 'gpay')   #added
        features['payment_method_paytm'] = int(payment_method == 'paytm')
        return features
        print("✅ Extracted features:", features, flush=True)
        return features  #added

    def get_ip_country(self, ip_address: str) -> str:
        """Return the country code of the given IP using ipinfo.io"""
        if not ip_address:
            return "Unknown"
        try:
            if ip_address in ["127.0.0.1", "localhost"]:
                response = requests.get("https://ipinfo.io/json")
            else:
                response = requests.get(f"https://ipinfo.io/{ip_address}/json")
            data = response.json()
            print(f"🌍 Detected IP Country for {ip_address}: {data.get('country')}") 
            return data.get("country", "Unknown")
        except Exception as e:
            print(f"Failed to get IP country for {ip_address}: {e}")
            return "Unknown"    
    











