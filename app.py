import os
import sqlite3
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# local project imports
from fraud_detection.model_trainer import FraudDetectionModel
from fraud_detection.feature_engineering import FeatureEngineer
from fraud_detection.database import DatabaseManager

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-key')
CORS(app)

# Initialize components (FraudDetectionModel will attempt to load model files)
db_manager = DatabaseManager()
feature_engineer = FeatureEngineer()
fraud_model = FraudDetectionModel()  # __init__ will call load_model internally

@app.route('/')
def index():
    return render_template('checkout.html')







@app.route('/api/process_transaction', methods=['POST'])
def process_transaction():
    try:
        print(">>> process_transaction called", flush=True)

        data = request.get_json() or {}
        print(">>> raw request JSON:", data, flush=True)
        user_agent = request.headers.get('User-Agent', '')
        ip_address = request.remote_addr or ''

        # Extract transaction data (include billing_country for IP mismatch)
        transaction_data = {
            'name': data.get('name'),
            'email': data.get('email'),
            'phone': data.get('phone'),
            'address': data.get('address'),
            'billing_country': data.get('billing_country'),  # <-- include billing_country if provided by form
            'payment_method': data.get('payment_method'),
            'amount': float(data.get('amount', 0)),
            'user_agent': user_agent,
            'ip_address': ip_address
        }
        print(">>> built transaction_data (pre-enrich):", transaction_data, flush=True)

        # Engineer features for fraud detection
        # ✅ Dynamically enrich user data before feature extraction
        user_id = db_manager.get_or_create_user(transaction_data['name'], transaction_data['email'])
        print(f">>> got user_id: {user_id}", flush=True)

        transaction_data['account_age_days'] = db_manager.get_account_age_days(user_id)
        transaction_data['past_transaction_count'] = db_manager.get_past_transaction_count(user_id,days=1)    #added
        transaction_data['transaction_speed'] = db_manager.get_user_transaction_speed(user_id,minutes=60)
        transaction_data['transactions_last_hour'] = db_manager.get_user_transaction_speed(user_id, minutes=60)

        print(">>> transaction_data after DB enrich:", transaction_data, flush=True)
        features = feature_engineer.extract_features(transaction_data)
        print("✅ Extracted features:", features,flush=True)   # <-- add this for debugging (remove later)


        # If model missing, return helpful error (so front-end shows message)
        if fraud_model.model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded or trained yet. Please train the model first.'
            }), 500

        # Get fraud prediction
        fraud_score, decision, shap_explanation = fraud_model.predict(features)

        # Store transaction in database
        user_id = db_manager.get_or_create_user(transaction_data['name'], transaction_data['email'])
        transaction_id = db_manager.store_transaction(
            user_id=user_id,
            amount=transaction_data['amount'],
            fraud_score=fraud_score,
            decision=decision,
            reason=shap_explanation.get('top_reason', 'Model analysis')
        )

        # Send alert if high risk (optionally)
        if fraud_score > 80:
            send_fraud_alert(transaction_id, transaction_data, fraud_score, shap_explanation)

        return jsonify({
            'success': True,
            'transaction_id': transaction_id,
            'fraud_score': fraud_score,
            'decision': decision,
            'explanation': shap_explanation,
            'message': f'Transaction {decision.lower()}. Fraud score: {fraud_score}/100'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    try:
        stats = db_manager.get_dashboard_stats()
        dashboard_data = db_manager.get_dashboard_data()
        return render_template('dashboard.html', stats=stats, dashboard_data=dashboard_data)
    except Exception as e:
        print(f"Dashboard error: {e}")
        return render_template('dashboard.html', stats={}, dashboard_data={})


@app.route('/api/dashboard_data')
def dashboard_data_api():
    try:
        data = db_manager.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        print(f"Dashboard API error: {e}")
        return jsonify({'daily_fraud_rate': [], 'fraud_reasons': [], 'recent_transactions': []})


def send_fraud_alert(transaction_id, transaction_data, fraud_score, shap_explanation):
    try:
        smtp_server = os.getenv('SMTP_SERVER')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        username = os.getenv('SMTP_USERNAME')
        password = os.getenv('SMTP_PASSWORD')
        admin_email = os.getenv('ADMIN_EMAIL')

        if not all([smtp_server, username, password, admin_email]):
            print("Email configuration incomplete. Skipping alert.")
            return

        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = admin_email
        msg['Subject'] = f'🚨 High Risk Transaction Alert - Score: {fraud_score}'

        body = f"""HIGH RISK TRANSACTION DETECTED

Transaction ID: {transaction_id}
Fraud Score: {fraud_score}/100

Customer Details:
- Name: {transaction_data.get('name')}
- Email: {transaction_data.get('email')}
- Amount: ${transaction_data.get('amount', 0):.2f}

Risk Factors:
{shap_explanation.get('detailed_explanation', '')}

Please review this transaction immediately.
"""
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        print(f"Fraud alert sent for transaction {transaction_id}")
    except Exception as e:
        print(f"Failed to send fraud alert: {e}")


if __name__ == '__main__':
    # Initialize database
    db_manager.init_database()

    # If model doesn't exist, train it (this will create the PKL files)
    if not fraud_model.model_exists():
        print("Model files not found — training model now (this may take a while)...")
        fraud_model.train_model()
        # reload model after training so Flask uses the freshly saved artifacts
        fraud_model.load_model()

    # Start server
    app.run(host='0.0.0.0', debug=True, port=int(os.getenv('PORT', 5000)))

