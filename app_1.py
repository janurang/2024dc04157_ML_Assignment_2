import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("Machine Learning Classification Models App")

st.write("Upload CSV dataset (Last column should be target variable)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert categorical
    X = pd.get_dummies(X, drop_first=True)

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model Selection
    model_name = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()

    elif model_name == "KNN":
        model = KNeighborsClassifier()

    elif model_name == "Naive Bayes":
        model = GaussianNB()

    elif model_name == "Random Forest":
        model = RandomForestClassifier()

    elif model_name == "XGBoost":
        model = XGBClassifier(eval_metric='logloss')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("Evaluation Metrics")

    st.write({
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

# Confusion Matrix

st.subheader("Confusion Matrix")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a small figure
fig, ax = plt.subplots(figsize=(4, 3))

# Plot confusion matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=ax,
    annot_kws={"size": 12, "weight": "bold"}
)

# Axis labels
ax.set_xlabel("Predicted Label", fontsize=10)
ax.set_ylabel("Actual Label", fontsize=10)

# Title
ax.set_title(
    f"{model_name} - Confusion Matrix",
    fontsize=11
)

# Smaller tick labels
ax.tick_params(
    axis="both",
    labelsize=9
)

# Prevent excessive margins
plt.tight_layout()

# Display figure without stretching
st.pyplot(
    fig,
    use_container_width=False
)

# Close the figure
plt.close(fig)

# Classification Report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred)

st.text(report)
else:
st.info("Please upload dataset to start.")
