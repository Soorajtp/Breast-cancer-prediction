import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rev03_2025aa05652_ml2 import (
    load_data,
    split_and_scale,
    get_model,
    train_and_evaluate,
    predict_on_new_data
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Breast Cancer ML App", layout="wide")
st.title("Breast Cancer Prediction")

#---------------------------------------------------
#Instructions
#---------------------------------------------------
st.info(
    """
    **Welcome! Thank you for using this Machine Learning prediction app.**

    Please follow the steps below:
    1. First, upload your data file to get predictions.
    2. You can also download a sample test specimen using the **Download Test Specimen** button and use that file for upload.
    3. Select a model to train and view its performance metrics and confusion matrix.
    4. After uploading your data click **Train & Evaluate** button to see the prediction result.
    5. You can scroll through the table to review all predictions (Benign or Malignant).
    """
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df, X, y = load_data()

# Split once so we can create test specimen download
X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y)

# Create TEST SPECIMEN (20% test data)
test_specimen = pd.DataFrame(X_test_scaled, columns=X.columns).head(50)

# --------------------------------------------------
# HEADER — UPLOAD YOUR DATA
# --------------------------------------------------
st.header("Upload Your Data")

# Download test specimen button
st.download_button(
    label="Download Test Specimen CSV",
    data=test_specimen.to_csv(index=False).encode("utf-8"),
    file_name="test_specimen.csv",
    mime="text/csv",
)

# Upload button
uploaded_file = st.file_uploader(
    "Upload your CSV file for prediction",
    type=["csv"]
)

# --------------------------------------------------
# TRAIN & EVALUATE MODELS
# --------------------------------------------------
st.header("Model Training & Evaluation")

model_name = st.selectbox(
    "Select Model to View Metrics",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

train_button = st.button("Train & Evaluate Models")

if train_button:

    # Train selected model
    model = get_model(model_name)
    metrics, y_pred, y_prob = train_and_evaluate(
        model, X_train_scaled, X_test_scaled, y_train, y_test
    )

    st.subheader("Model Metrics")
    metrics_df = pd.DataFrame([metrics])
    st.dataframe(metrics_df)

    # ---------------- BAR CHART ----------------
    st.subheader("Performance Bar Chart")

    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(metrics.keys(), metrics.values())
    ax.set_ylim(0,1)
    ax.set_title(f"{model_name} Performance")
    st.pyplot(fig)

    # ---------------- CONFUSION MATRIX ----------------
    from sklearn.metrics import confusion_matrix

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig2, ax2 = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    # Save trained model for prediction use
    trained_model = get_model("Random Forest")
    trained_model.fit(X_train_scaled, y_train)

# --------------------------------------------------
# PREDICTION ON UPLOADED DATA
# --------------------------------------------------
if uploaded_file is not None:

    st.header("Prediction Results")
    

    new_df = pd.read_csv(uploaded_file)
    #st.write("DEBUG: File loaded, shape:", new_df.shape)

    st.subheader("Uploaded File Preview")
    st.dataframe(new_df.head())
    

    # Ensure same column order as training data
    #new_df = new_df[X.columns]
    try:
        new_df = new_df[X.columns]
    except KeyError as e:
        st.error("Uploaded file does not have the required columns.")
        st.write("Expected columns:", list(X.columns))
        st.write("Your file has columns:", list(new_df.columns))
        st.stop()

#  Ensure all data is numeric
    new_df = new_df.apply(pd.to_numeric, errors="coerce")

#  Check for NaNs after conversion
    if new_df.isnull().any().any():
        st.error("Uploaded file contains non-numeric or missing values. Please fix and re-upload.")
        st.stop()

    # Train model (you can change model here if needed)
    model = get_model("Random Forest")
    model.fit(X_train_scaled, y_train)

    # Predict
    #preds = predict_on_new_data(model, scaler, new_df)
    #st.write("DEBUG: Prediction done, preds shape:", len(preds))
    try:
        preds = predict_on_new_data(model, scaler, new_df)
        #st.write("DEBUG: Prediction done, preds length:", len(preds))
    except Exception as e:
        st.error("Prediction failed!")
        st.exception(e)   # 👈 This will show the real error message
        st.stop()

    # Build results table
    result_df = new_df.copy()
    result_df["Prediction"] = pd.Series(preds).map({0: "Benign", 1: "Malignant"})

    st.subheader("Predictions (Scrollable Table)")
    st.dataframe(result_df, height=400)   

    # Download button
    st.download_button(
        label="Download Predictions CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
