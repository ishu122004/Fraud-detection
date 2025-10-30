import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import shap

# --- Human-readable SHAP explanations ---
FEATURE_EXPLANATIONS = {
    "TransactionAmt": lambda v: (
        f"High transaction amount: ₹{float(v):,.0f}"
        if safe_numeric(v) and float(v) > 50000
        else (f"Very high transaction amount: ₹{float(v):,.0f}" if safe_numeric(v) and float(v) > 100000 else None)
    ),
    "is_new_account": lambda v: (
        "New account (created within last 7 days)" if int_or_zero(v) == 1 else None
    ),
    "ip_country_mismatch": lambda v: (
        "IP ≠ billing country" if int_or_zero(v) == 1 else None
    ),
    "is_disposable_email": lambda v: (
        "Disposable or temporary email used" if int_or_zero(v) == 1 else None
    ),
    "is_night": lambda v: (
        "Transaction made at unusual time (night hours)" if int_or_zero(v) == 1 else None
    ),
    "transactions_last_hour": lambda v: (
        f"Multiple rapid transactions ({int(v)} in last hour)" if safe_numeric(v) and int(v) > 2 else None
    ),
    "past_transaction_count": lambda v: (
        f"Frequent buyer: {int(v)} past transactions" if safe_numeric(v) and int(v) > 5 else None
    ),
    "transaction_speed": lambda v: (
        f"Unusually fast repeat orders (avg {float(v):.1f}/hr)" if safe_numeric(v) and float(v) > 1 else None
    ),
    "account_age_days": lambda v: (
        f"New account ({int(v)} days old)" if safe_numeric(v) and int(v) < 7 else None
    ),
    "name_email_mismatch": lambda v: (
        "Name and email do not match" if int_or_zero(v) == 1 else None
    ),
}


# small helpers used by FEATURE_EXPLANATIONS
def safe_numeric(v):
    try:
        float(v)
        return True
    except Exception:
        return False

def int_or_zero(v):
    try:
        return int(v)
    except Exception:
        return 0


class FraudDetectionModel:
    def __init__(self):
        # base dir = folder containing this file (fraud_detection)
        base_dir = os.path.dirname(__file__)
        self.models_dir = os.path.join(base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # paths inside fraud_detection/models/
        self.model_path = os.path.join(self.models_dir, "fraud_model.pkl")
        self.scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        self.feature_names_path = os.path.join(self.models_dir, "feature_names.pkl")

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.explainer = None

        # Try to load existing model if present
        self.load_model()

    def model_exists(self):
        return (
            os.path.exists(self.model_path)
            and os.path.exists(self.scaler_path)
            and os.path.exists(self.feature_names_path)
        )

    # --- Attempt to load model/scaler/feature_names ---
    def load_model(self):
        try:
            print("🔍 Checking model files:")
            print(" Model:", self.model_path)
            print(" Scaler:", self.scaler_path)
            print(" Feature names:", self.feature_names_path)

            if self.model_exists():
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                with open(self.feature_names_path, "rb") as f:
                    self.feature_names = pickle.load(f)

                if self.model is not None:
                    # build SHAP explainer
                    self.explainer = shap.TreeExplainer(self.model)
                print("✅ Model loaded successfully!")
            else:
                print("⚠ Model files not found. Train the model (model.train_model()).")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
            self.feature_names = None
            self.explainer = None

    # --- Load real Kaggle data if available (project root /data/) ---
    def _load_real_data(self):
        print("📂 Loading Kaggle dataset (if present)...")
        project_root = os.path.dirname(os.path.dirname(_file_))
        trans_path = os.path.join(project_root, "data", "train_transaction.csv")
        id_path = os.path.join(project_root, "data", "train_identity.csv")

        print(f" Looking for: {os.path.abspath(trans_path)}")
        print(f" Looking for: {os.path.abspath(id_path)}")

        if not os.path.exists(trans_path) or not os.path.exists(id_path):
            raise FileNotFoundError("train_transaction.csv or train_identity.csv not found in /data")
        trans = pd.read_csv(trans_path)
        identity = pd.read_csv(id_path)
        return trans.merge(identity, how="left", on="TransactionID")

    # --- Synthetic fallback (keeps pipeline working when Kaggle files absent) ---
    def _create_synthetic_data(self, n_samples=50000):
        print("⚙ Creating synthetic dataset (fallback)...")
        np.random.seed(42)
        data = {}
        data["TransactionAmt"] = np.random.lognormal(3, 1.5, n_samples)
        data["is_new_account"] = (np.random.exponential(50, n_samples) < 7).astype(int)
        data["ip_country_mismatch"] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        data["is_disposable_email"] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        data["is_night"] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        data["transactions_last_hour"] = np.random.poisson(0.5, n_samples)
        data["name_email_mismatch"] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        df = pd.DataFrame(data)
        # simple fraud probability function
        fraud_prob = (
            0.05
            + 0.3 * df["TransactionAmt"].gt(50000).astype(float)
            + 0.25 * df["is_new_account"]
            + 0.2 * df["ip_country_mismatch"]
            + 0.15 * df["is_disposable_email"]
            + 0.1 * (df["transactions_last_hour"] > 2).astype(int)
        )
        fraud_prob = np.minimum(fraud_prob, 0.95)
        y = np.random.binomial(1, fraud_prob)
        df["isFraud"] = y
        return df, "isFraud"

    # --- Feature engineering used for training (when using Kaggle) ---
    def _create_dynamic_features(self, df: pd.DataFrame):
        print("⚙ Creating dynamic features for training...")
        # basic dynamic features (if Kaggle provided)
        df = df.copy()
        # TransactionDT may be in seconds relative to some anchor — create hour-of-day
        if "TransactionDT" in df.columns:
            df["TransactionDT"] = df["TransactionDT"].astype(float)
            df["hour"] = ((df["TransactionDT"] / (60 * 60)) % 24).astype(int)
            df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
        else:
            df["is_night"] = 0

        # account_age_days simulated via frequency of card1
        if "card1" in df.columns and "TransactionDT" in df.columns:
            df["account_age_days"] = df.groupby("card1")["TransactionDT"].rank()
            df["is_new_account"] = (df["account_age_days"] <= 5).astype(int)
        else:
            df["is_new_account"] = 0

        # simulate ip_country_mismatch from addr1 parity if addr1 exists
        if "addr1" in df.columns:
            df["ip_country_mismatch"] = ((df["addr1"].fillna(0).astype(int) % 2) == 0).astype(int)
        else:
            df["ip_country_mismatch"] = 0

        # disposable email from P_emaildomain
        disposable_domains = ["mailinator.com", "guerrillamail.com", "tempmail.com"]
        if "P_emaildomain" in df.columns:
            df["is_disposable_email"] = df["P_emaildomain"].astype(str).apply(
                lambda x: int(any(dom in x for dom in disposable_domains))
            )
        else:
            df["is_disposable_email"] = 0

        if "card1" in df.columns and "TransactionDT" in df.columns:
            df["transactions_last_hour"] = df.groupby("card1")["TransactionDT"].diff().fillna(999999)
            df["transactions_last_hour"] = (df["transactions_last_hour"] < 3600).astype(int)
        else:
            df["transactions_last_hour"] = 0

        # Pick the features we'll train on
        features = [
            "TransactionAmt",
            "is_new_account",
            "ip_country_mismatch",
            "is_disposable_email",
            "transactions_last_hour",
        ]
        # add placeholder categorical columns if present
        if "card4" in df.columns:
            features.append("card4")
            df["card4"] = df["card4"].fillna("unknown").astype("category").cat.codes
        if "DeviceType" in df.columns:
            features.append("DeviceType")
            df["DeviceType"] = df["DeviceType"].fillna("desktop").astype("category").cat.codes

        # ensure isFraud exists
        if "isFraud" not in df.columns:
            df["isFraud"] = 0

        training_df = df[features + ["isFraud"]].fillna(0)
        return training_df

    # --- Train the model (uses Kaggle data if present, else synthetic fallback) ---
    def train_model(self):
        print("🚀 Training fraud detection model...")
        # Try Kaggle — if not present, fallback to synthetic
        try:
            full_df = self._load_real_data()
            df = self._create_dynamic_features(full_df)
            X = df.drop("isFraud", axis=1)
            y = df["isFraud"]
        except FileNotFoundError:
            print("⚠ Kaggle files not found; training on synthetic fallback.")
            synthetic_df, target_col = self._create_synthetic_data()
            y = synthetic_df["isFraud"]
            X = synthetic_df.drop("isFraud", axis=1)

        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="auc"
        )

        self.model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        print("\n📊 Model Performance:")
        try:
            print(classification_report(y_test, y_pred))
            print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        except Exception:
            print("Unable to compute full metrics (insufficient variability in y_test).")

        self.explainer = shap.TreeExplainer(self.model)
        self.save_model()
        print("✅ Model trained and saved!")

    # --- Predict used by Flask --- 
    def predict(self, features: dict):
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not loaded or trained yet. Please train the model first.")

        # build dataframe with expected feature order
        feature_df = pd.DataFrame([features])
        # ensure all expected columns exist and are scalar
        for f in self.feature_names:
            if f not in feature_df.columns:
                feature_df[f] = 0
        feature_df = feature_df[self.feature_names].astype(object).fillna(0)

        # convert all numpy types to Python native where necessary
        feature_df = feature_df.applymap(lambda x: (x.item() if hasattr(x, "item") else x))

        # scale and predict
        features_scaled = self.scaler.transform(feature_df)
        fraud_prob = float(self.model.predict_proba(features_scaled)[0][1])

        # small rule-based adjustments to make high-risk flags more influential
        # --- make risk contribution stronger ---
        adjust = 0.0

        amount = float(features.get("TransactionAmt", 0))
        if amount > 100000:
            adjust += 0.45
        elif amount > 50000:
            adjust += 0.30
        elif amount > 20000:
            adjust += 0.15

        if int_or_zero(features.get("is_new_account")) == 1:
            adjust += 0.20
        if int_or_zero(features.get("ip_country_mismatch")) == 1:
            adjust += 0.25
        if int_or_zero(features.get("is_disposable_email")) == 1:
            adjust += 0.20
        
        if int_or_zero(features.get("transactions_last_hour")) > 2:
            adjust += 0.15

# cap the combined adjustment so it doesn't exceed 1.0
        fraud_prob = min(1.0, fraud_prob + adjust)


        fraud_score = int(round(fraud_prob * 100))
        # ensure a small baseline so UI never shows 0 for edge-case transactions
        fraud_score = max(3, fraud_score)

        decision = "Approved" if fraud_score <= 70 else "Under Review"

        # Build SHAP-style human readable explanation
                # Build SHAP-style human readable explanation
        try:
            shap_values = self.explainer.shap_values(features_scaled)
            # shap_values may be list for multiclass or single array
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.array(shap_values).reshape(-1)

            feature_importance = list(zip(self.feature_names, shap_values))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            detailed_lines = []
            seen = set()

            for feature_name, impact in feature_importance:
                if feature_name.lower() in ["card4", "devicetype","is_night"]:
                    continue

                value = feature_df[feature_name].iloc[0]
                reason_fn = FEATURE_EXPLANATIONS.get(feature_name)
                reason_text = None

                # ✅ If we have a custom readable label (from FEATURE_EXPLANATIONS)
                if reason_fn:
                    try:
                        reason_text = reason_fn(value)
                    except Exception:
                        reason_text = None

                # 🧠 Additional custom mappings for technical columns
                if not reason_text:
                   
                    if feature_name.lower() == "card4":
                        reason_text = "Unusual card network usage"
                    elif feature_name.lower() == "devicetype":
                        reason_text = "Unrecognized device or browser type"
                    elif feature_name.lower() == "transactionamt":
                        reason_text = f"Transaction amount ₹{float(value):,.0f}"
                    elif feature_name.lower() == "ip_country_mismatch" and int(value) == 1:
                        reason_text = "IP ≠ billing country"
                    
                # Default fallback if nothing else applies
                if not reason_text:
                    direction = "↑" if float(impact) > 0 else "↓"
                    reason_text = f"{feature_name.replace('_', ' ').title()}: {value} {direction}"

                # Add only unique, meaningful lines
                reason_text = str(reason_text).strip()
                if reason_text and reason_text not in seen:
                    detailed_lines.append(f"- {reason_text}")
                    seen.add(reason_text)

                if len(seen) >= 5:
                    break

            # ✅ Ensure fallback if no explanations found
            if len(detailed_lines) == 0:
                detailed_lines = ["- Model analysis"]

            detailed_explanation = "Reasons:\n" + "\n".join(detailed_lines)
            top_reason = detailed_lines[0] if detailed_lines else "Model analysis"

            shap_explanation = {
                "top_reason": top_reason,
                "detailed_explanation": detailed_explanation,
                "feature_importance": [(f, float(i)) for f, i in feature_importance[:10]],
            }

        except Exception as e:
            shap_explanation = {
                "top_reason": "Model analysis",
                "detailed_explanation": f"Unable to generate SHAP explanation: {str(e)}",
                "feature_importance": [],
            }

        return fraud_score, decision, shap_explanation


             
   

    # --- Save model artifacts ---
    def save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(self.feature_names_path, "wb") as f:
            pickle.dump(self.feature_names, f)
        print("💾 Saved model, scaler and feature names.")
if __name__ == "__main__":   #added
    print("🚀 Starting model training...")
    model = FraudDetectionModel()
    model.train_model()
    print("🎉 Training completed!")

