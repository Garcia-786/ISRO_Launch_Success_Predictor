import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.preprocess import load_and_clean
from scripts.model import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

# Cache dataset load/clean so it doesn't rerun every interaction
@st.cache_data
def load_data(path: str):
    return load_and_clean(path)

df = load_data("data/isro_launches.csv")

st.title("ISRO Launch Success Predictor")
st.sidebar.header("Launch Details Input")

# Build original options from one-hot columns
launch_vehicles = sorted(df.filter(like='launch_vehicle_').columns.str.replace('launch_vehicle_', ''))
orbit_types = sorted(df.filter(like='orbit_type_').columns.str.replace('orbit_type_', ''))
applications = sorted(df.filter(like='application_').columns.str.replace('application_', ''))

launch_vehicle_input = st.sidebar.selectbox("Launch Vehicle", launch_vehicles)
orbit_type_input = st.sidebar.selectbox("Orbit Type", orbit_types)
application_input = st.sidebar.selectbox("Application", applications)

# Prepare features and labels and split BEFORE training to avoid leakage
X = df.drop(['sl_no', 'name', 'launch_date', 'outcome'], axis=1, errors='ignore')
y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model once and cache it
@st.cache_resource
def get_trained_model(X_train, y_train):
    # train_model now accepts X_train and y_train
    return train_model(X_train, y_train)

with st.spinner("Training model..."):
    model, X_columns = get_trained_model(X_train, y_train)

# Predict function with graceful fallbacks
def predict_launch_success(model, X_columns, launch_vehicle, orbit_type, application):
    new_data = pd.DataFrame(0, index=[0], columns=X_columns)

    lv_col = "launch_vehicle_" + str(launch_vehicle).strip()
    orbit_col = "orbit_type_" + str(orbit_type).strip()
    app_col = "application_" + str(application).strip()

    missing = []
    for col in (lv_col, orbit_col, app_col):
        if col in new_data.columns:
            new_data.at[0, col] = 1
        else:
            missing.append(col)

    if missing:
        st.warning(f"These feature columns are missing from model features: {missing}")

    # Prefer predict_proba; otherwise fall back to predict
    try:
        prob = float(model.predict_proba(new_data)[0][1])
    except Exception:
        pred = model.predict(new_data)[0]
        prob = 1.0 if pred == 1 or str(pred).lower() in ("success", "true", "1") else 0.0

    return prob

# allow user to change threshold
threshold = st.sidebar.slider("Success threshold", 0.0, 1.0, 0.7, 0.01)

if st.sidebar.button("Predict Outcome"):
    prob = predict_launch_success(model, X_columns, launch_vehicle_input, orbit_type_input, application_input)
    prediction = "Success ✅" if prob > threshold else "Failure ❌"
    st.subheader("Prediction Result")
    st.write(f"Predicted Outcome: {prediction}")
    st.write(f"Probability of Success: {prob:.2f}")

# Evaluate on held-out test set
st.subheader("Model Evaluation")

y_pred = model.predict(X_test)

# compute metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
metrics_text = f"Accuracy: {acc:.3f} — F1: {f1:.3f}"

# compute ROC AUC if available
try:
    probs_test = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probs_test)
    metrics_text += f" — ROC AUC: {roc_auc:.3f}"
except Exception:
    st.info("ROC AUC not available: model doesn't support predict_proba.")

st.write(metrics_text)

st.write("Confusion Matrix on Test Set:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual Failure","Actual Success"], columns=["Pred Failure","Pred Success"])
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
st.pyplot(fig_cm)

# Outcome histogram
st.subheader("Launch Outcomes Distribution")
outcome_counts = df['outcome'].value_counts()

fig, ax = plt.subplots()
colors = ['tab:green', 'tab:red'] if len(outcome_counts) <= 2 else plt.cm.tab10.colors
ax.bar(outcome_counts.index.astype(str), outcome_counts.values, color=colors[:len(outcome_counts)])
ax.set_title("Launch Outcome Histogram")
ax.set_xlabel("Outcome")
ax.set_ylabel("Count")
st.pyplot(fig)
