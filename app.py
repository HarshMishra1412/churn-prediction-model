import streamlit as st
import pandas as pd
import joblib

# ── Load saved artifacts (must be in same folder as this file) ──
model = joblib.load('churn_model.pkl')
encoders = joblib.load('encoders.pkl')        # dict: {column_name: LabelEncoder}
feature_cols = joblib.load('features.pkl')    # list of columns in training order

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")
st.title("📊 Customer Churn Predictor")
st.caption("XGBoost model — 92% churn recall, ROC-AUC 0.76")

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    # Add more inputs here if your model uses more features
    # (e.g. OnlineBackup, DeviceProtection, PhoneService, etc.)

if st.button("Predict Churn", type="primary"):

    # 1. Raw inputs — add ALL original columns your model was trained on
    raw = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
    }

    # 2. Apply LabelEncoders to categorical columns (same as training)
    for col, le in encoders.items():
        if col in raw:
            raw[col] = le.transform([raw[col]])[0]

    # 3. Feature engineering — MUST match training exactly
    raw['charges_per_month'] = total_charges / (tenure + 1)
    raw['is_new_customer'] = 1 if tenure <= 6 else 0
    raw['is_loyal_customer'] = 1 if tenure >= 24 else 0

    # total_services: count of active service add-ons (adjust list as per your training code)
    service_cols = ['OnlineSecurity', 'TechSupport']  # add OnlineBackup, DeviceProtection etc if used
    raw['total_services'] = sum(1 for c in service_cols if raw.get(c) == 1)

    raw['has_full_protection'] = 1 if all(raw.get(c) == 1 for c in service_cols) else 0

    # is_high_value: example logic, tune based on your training code
    raw['is_high_value'] = 1 if (monthly_charges > 70 and tenure >= 24) else 0

    # 4. Build dataframe in EXACT training column order
    df = pd.DataFrame([raw])
    df = df.reindex(columns=feature_cols, fill_value=0)

    # 4b. Safety net: force everything numeric (catches any column that
    # didn't get encoded properly, which causes XGBoost dtype errors)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Debug info — expand this if predictions look wrong or error persists
    with st.expander("🔧 Debug info (click if something looks off)"):
        st.write("Encoder keys available:", list(encoders.keys()))
        st.write("Raw dict before reindex:", raw)
        st.write("Final dataframe sent to model:")
        st.dataframe(df)
        st.write("Column dtypes:")
        st.write(df.dtypes)

    # 5. Predict using your tuned threshold
    # .values bypasses XGBoost's feature_names check — important if SMOTE
    # converted X_train to a numpy array during training (common case)
    try:
        prob = model.predict_proba(df)[0][1]
    except ValueError:
        prob = model.predict_proba(df.values)[0][1]
    THRESHOLD = 0.35
    pred = int(prob >= THRESHOLD)

    st.divider()
    st.metric("Churn Probability", f"{prob:.1%}")

    if pred == 1:
        st.error("⚠️ Prediction: Will Churn")
    else:
        st.success("✅ Prediction: Will Stay")

    # 6. Risk tier (matches your README's risk buckets)
    if prob >= 0.70:
        st.warning("🔴 HIGH RISK — Immediate personal outreach + strong offer")
    elif prob >= 0.40:
        st.info("🟡 MEDIUM RISK — Automated retention email + small discount")
    else:
        st.info("🟢 LOW RISK — No action needed, monitor monthly")
