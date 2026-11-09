import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Load pipeline and define helpers
# ---------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    return joblib.load("retail_sales_ml_pipeline.joblib")

try:
    bundle = load_model_bundle()
    preprocessor = bundle["preprocessor"]
    kmeans = bundle["kmeans"]
    classifier = bundle["classifier"]
    feature_columns = bundle["feature_columns"]
    numeric_features = bundle["numeric_features"]
    categorical_features = bundle["categorical_features"]
    schema = bundle["input_schema"]
except Exception as e:
    st.error(f"Failed to load model bundle: {e}")
    st.stop()

def predict(df_input, bundle):
    # Ensure exact feature columns and types
    cols = bundle["feature_columns"]
    df_cleaned = df_input[cols].copy()
    
    # Convert types dynamically
    for num_col in bundle["numeric_features"]:
        df_cleaned[num_col] = pd.to_numeric(df_cleaned[num_col], errors="coerce")
    for cat_col in bundle["categorical_features"]:
        df_cleaned[cat_col] = df_cleaned[cat_col].astype(str)
        
    # Transform and predict
    processed = bundle["preprocessor"].transform(df_cleaned)
    kmeans_preds = bundle["kmeans"].predict(processed)
    class_preds = bundle["classifier"].predict(processed)
    
    results = df_cleaned.copy()
    results["KMeans_Segment"] = kmeans_preds
    results["Classifier_Prediction"] = class_preds
    
    # Estimate probabilities if classifier supports it
    if hasattr(bundle["classifier"], "predict_proba"):
        probs = bundle["classifier"].predict_proba(processed)
        results["Classifier_Confidence"] = np.max(probs, axis=1)
        
    return results

# ---------------------------------------------------------
# Layout & Title
# ---------------------------------------------------------
st.set_page_config(page_title="Retail Sales Segment Predictor", layout="wide")
st.title("🏪 Retail Sales Segment Predictor")
st.write("This interactive app groups new sales records using our pretrained K-Means clusters and predicts their segments using a calibrated Logistic Regression classifier.")

single_tab, batch_tab, schema_tab = st.tabs([
    "🔍 Single Record Prediction", 
    "📁 Batch CSV Upload", 
    "📋 Input Schema Description"
])

# ---------------------------------------------------------
# Single record prediction
# ---------------------------------------------------------
with single_tab:
    st.subheader("Enter Sale Attributes manually")
    user_values = {}
    
    with st.form("single_prediction_form"):
        col_left, col_right = st.columns(2)
        
        for i, col in enumerate(feature_columns):
            # Alternate inputs between two columns
            target_col = col_left if i % 2 == 0 else col_right
            
            with target_col:
                if col in numeric_features:
                    col_info = schema.get(col, {})
                    min_val = float(col_info.get("minimum", 0.0)) if col_info.get("minimum") is not None else 0.0
                    max_val = float(col_info.get("maximum", 1000000.0)) if col_info.get("maximum") is not None else 1000000.0
                    user_values[col] = st.number_input(
                        f"{col} (Range: {min_val} to {max_val})",
                        min_value=min_val,
                        max_value=max_val,
                        value=min_val
                    )
                elif col in categorical_features:
                    allowed = schema.get(col, {}).get("allowed_values", [])
                    if allowed:
                        user_values[col] = st.selectbox(col, options=allowed)
                    else:
                        user_values[col] = st.text_input(col)
                else:
                    user_values[col] = st.text_input(col)

        submitted = st.form_submit_button("Predict Sales Segment", use_container_width=True)

    if submitted:
        try:
            input_df = pd.DataFrame([user_values], columns=feature_columns)
            prediction = predict(input_df, bundle)

            st.success("Prediction completed.")

            col1, col2 = st.columns(2)
            col1.metric("K-Means Segment", str(prediction.loc[0, "KMeans_Segment"]))
            col2.metric("Classifier Prediction", str(prediction.loc[0, "Classifier_Prediction"]))

            if "Classifier_Confidence" in prediction.columns:
                st.metric("Classifier Confidence", f"{prediction.loc[0, 'Classifier_Confidence']:.2%}")

            st.dataframe(prediction, use_container_width=True)

        except Exception as exc:
            st.error(f"Input validation or prediction failed: {exc}")

# ---------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------
with batch_tab:
    st.subheader("Upload a CSV file")
    st.write("The CSV must contain every model feature with the exact column names shown in the Input Schema tab.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Uploaded data preview")
            st.dataframe(batch_data.head(), use_container_width=True)

            if st.button("Generate Batch Predictions"):
                predictions = predict(batch_data, bundle)
                st.success(f"Predictions generated for {len(predictions)} rows.")
                st.dataframe(predictions, use_container_width=True)

                csv_data = predictions.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions CSV",
                    data=csv_data,
                    file_name="retail_sales_predictions.csv",
                    mime="text/csv"
                )
        except Exception as exc:
            st.error(f"Could not process the uploaded file: {exc}")

# ---------------------------------------------------------
# Input schema
# ---------------------------------------------------------
with schema_tab:
    st.subheader("Required input features")

    schema_rows = []
    for col in feature_columns:
        if col in numeric_features:
            info = schema.get(col, {})
            schema_rows.append({
                "Feature": col,
                "Data Type": "Numeric",
                "Required": True,
                "Allowed Values / Range": f"{info.get('minimum')} to {info.get('maximum')}" if info.get("minimum") is not None else "Any valid numeric value"
            })
        elif col in categorical_features:
            allowed = schema.get(col, {}).get("allowed_values", [])
            schema_rows.append({
                "Feature": col,
                "Data Type": "Categorical",
                "Required": True,
                "Allowed Values / Range": ", ".join(map(str, allowed))
            })
        else:
            schema_rows.append({
                "Feature": col,
                "Data Type": "Other",
                "Required": True,
                "Allowed Values / Range": "See training data"
            })

    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

st.divider()
st.caption("Educational demonstration. K-Means creates the segments. The classifier learns to reproduce those segments. A high classifier score does not prove that the segments are useful or correct for business decisions.")
