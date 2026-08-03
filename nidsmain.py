import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------

st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Based Network Intrusion Detection System")
st.markdown("""
This project uses the **Random Forest Machine Learning Algorithm**
to detect whether incoming network traffic is **BENIGN** or **DDoS Attack**.
""")

# ----------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

    return df


df = load_data()

# ----------------------------------------------------
# DATASET PREVIEW
# ----------------------------------------------------

st.header("Dataset Preview")

st.dataframe(df.head())

# ----------------------------------------------------
# DATASET INFORMATION
# ----------------------------------------------------

st.header("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

# ----------------------------------------------------
# REMOVE MISSING VALUES
# ----------------------------------------------------

df = df.dropna()

# ----------------------------------------------------
# LABEL ENCODING
# ----------------------------------------------------

st.header("Label Encoding")

if df["Label"].dtype == "object":

    encoder = LabelEncoder()

    df["Label"] = encoder.fit_transform(df["Label"])

    st.success("Label column encoded successfully.")

else:

    st.info("Label column is already numeric.")

# ----------------------------------------------------
# FEATURES & TARGET
# ----------------------------------------------------

X = df.drop("Label", axis=1)

y = df["Label"]

# ----------------------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

st.success("Dataset Split Completed")

st.write("Training Samples :", X_train.shape[0])
st.write("Testing Samples :", X_test.shape[0])

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.title("Control Panel")

n_estimators = st.sidebar.slider(
    "Number of Trees",
    10,
    300,
    100
)

max_depth = st.sidebar.slider(
    "Maximum Tree Depth",
    2,
    30,
    10
)

train_button = st.sidebar.button("Train Model")
# ----------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------

st.header("Model Training")

if train_button:

    with st.spinner("Training Random Forest Model..."):

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        st.success("Model Trained Successfully!")

        # Save model

        st.session_state["model"] = model

        st.session_state["predictions"] = predictions

        st.session_state["accuracy"] = accuracy

        st.session_state["precision"] = precision

        st.session_state["recall"] = recall

        st.session_state["f1"] = f1

# ----------------------------------------------------
# PERFORMANCE METRICS
# ----------------------------------------------------

if "model" in st.session_state:

    st.header("Performance Metrics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{st.session_state['accuracy']*100:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{st.session_state['precision']*100:.2f}%"
    )

    c3.metric(
        "Recall",
        f"{st.session_state['recall']*100:.2f}%"
    )

    c4.metric(
        "F1 Score",
        f"{st.session_state['f1']*100:.2f}%"
    )

# ----------------------------------------------------
# CLASSIFICATION REPORT
# ----------------------------------------------------

    st.header("Classification Report")

    report = classification_report(
        y_test,
        st.session_state["predictions"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

# ----------------------------------------------------
# CONFUSION MATRIX
# ----------------------------------------------------

    st.header("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        st.session_state["predictions"]
    )

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["BENIGN","ATTACK"],
        yticklabels=["BENIGN","ATTACK"],
        ax=ax
    )

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Actual")

    st.pyplot(fig)

else:

    st.info("Click 'Train Model' from the sidebar to begin.")
    # ----------------------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------------------

if "model" in st.session_state:

    st.header("Feature Importance")

    model = st.session_state["model"]

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(importance)

    fig, ax = plt.subplots(figsize=(10,5))

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature",
        palette="viridis",
        ax=ax
    )

    ax.set_title("Feature Importance")

    st.pyplot(fig)

# ----------------------------------------------------
# DATASET VISUALIZATION
# ----------------------------------------------------

st.header("Dataset Visualizations")

# Class Distribution

st.subheader("Class Distribution")

label_names = df["Label"].copy()

if 'encoder' in locals():

    label_names = encoder.inverse_transform(label_names)

class_df = pd.DataFrame(label_names, columns=["Traffic"])

fig, ax = plt.subplots(figsize=(6,4))

sns.countplot(
    data=class_df,
    x="Traffic",
    palette="Set2",
    ax=ax
)

ax.set_title("BENIGN vs ATTACK")

st.pyplot(fig)

# ----------------------------------------------------
# CORRELATION HEATMAP
# ----------------------------------------------------

st.subheader("Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10,7))

sns.heatmap(
    corr,
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax
)

st.pyplot(fig)

# ----------------------------------------------------
# FEATURE DISTRIBUTION
# ----------------------------------------------------

st.subheader("Feature Distribution")

selected_feature = st.selectbox(
    "Select Feature",
    X.columns
)

fig, ax = plt.subplots(figsize=(8,4))

sns.histplot(
    df[selected_feature],
    bins=30,
    kde=True,
    ax=ax
)

ax.set_title(selected_feature)

st.pyplot(fig)

# ----------------------------------------------------
# DESCRIPTIVE STATISTICS
# ----------------------------------------------------

st.header("Dataset Statistics")

st.dataframe(df.describe())

# ----------------------------------------------------
# SAMPLE RECORDS
# ----------------------------------------------------

st.header("Random Sample Records")

st.dataframe(df.sample(10))
# ----------------------------------------------------
# LIVE TRAFFIC PREDICTION
# ----------------------------------------------------

st.header("🔍 Live Traffic Prediction")

if "model" in st.session_state:

    model = st.session_state["model"]

    st.write("Enter network traffic values below.")

    destination_port = st.number_input(
        "Destination Port",
        min_value=0,
        max_value=65535,
        value=80
    )

    flow_duration = st.number_input(
        "Flow Duration",
        min_value=0,
        value=5000
    )

    total_packets = st.number_input(
        "Total Forward Packets",
        min_value=0,
        value=25
    )

    packet_length = st.number_input(
        "Packet Length Mean",
        min_value=0.0,
        value=450.0
    )

    active_mean = st.number_input(
        "Active Mean",
        min_value=0.0,
        value=120.0
    )

    if st.button("Predict Traffic"):

        sample = np.array([[
            destination_port,
            flow_duration,
            total_packets,
            packet_length,
            active_mean
        ]])

        prediction = model.predict(sample)[0]

        if 'encoder' in locals():
            prediction = encoder.inverse_transform([prediction])[0]

        if prediction in ["DDoS", 1]:

            st.error("⚠️ MALICIOUS TRAFFIC DETECTED")

        else:

            st.success("✅ BENIGN NETWORK TRAFFIC")

else:

    st.warning("Train the model first from the sidebar.")

# ----------------------------------------------------
# DOWNLOAD PREDICTIONS
# ----------------------------------------------------

if "predictions" in st.session_state:

    st.header("Download Predictions")

    results = X_test.copy()

    results["Actual"] = y_test.values

    results["Predicted"] = st.session_state["predictions"]

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction Results",
        csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

# ----------------------------------------------------
# PROJECT SUMMARY
# ----------------------------------------------------

st.header("Project Summary")

st.info("""
✔ Dataset Loaded Successfully

✔ Data Preprocessing Completed

✔ Random Forest Model Trained

✔ Model Evaluation Completed

✔ Live Prediction Available

✔ Project Ready for Demonstration
""")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.markdown("---")

st.markdown(
"""
### AI-Based Network Intrusion Detection System

**Developed Using**

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

Made for Academic Project Demonstration.
"""
)