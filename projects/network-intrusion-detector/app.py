# web dashboard
#
# Upload network feature records, classify them, review alerts and export the predictions to CSV.

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.datasets import fetch_kddcup99


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "intrusion_model.joblib"


st.set_page_config(
    page_title="Network Intrusion Detector",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_model():
    # Load the trained preprocessing + classification pipeline
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_demo_data():
    # Get a small feature-only CSV sample for trying the dashboard
    dataset = fetch_kddcup99(
        subset="http",
        percent10=True,
        as_frame=True,
    )

    # Keep features only. Do not include the original target labels
    return dataset.data.head(20).copy()


st.title("Network Intrusion Detector")
st.caption(
    "Machine-learning baseline for classifying network records "
    "as normal or suspicious."
)

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run `python src/train.py` first."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


expected_columns = list(model.feature_names_in_)

with st.sidebar:
    st.header("About this project")
    st.write("**Model:** Random Forest")
    st.write("**Dataset:** KDD Cup 99 HTTP subset")
    st.write("**Task:** Binary classification")
    st.info(
        "This is an educational baseline. Predictions are not "
        "proof that traffic is malicious or safe."
    )

    st.divider()
    st.subheader("Need a sample CSV?")
    st.write(
        "Generate a small example containing the feature columns "
        "required by this model."
    )

    if st.button("Prepare demo CSV"):
        try:
            st.session_state["demo_csv"] = load_demo_data()
        except Exception as exc:
            st.error(f"Could not load demo data: {exc}")

    if "demo_csv" in st.session_state:
        demo_csv = st.session_state["demo_csv"].to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download demo CSV",
            data=demo_csv,
            file_name="network_traffic_demo.csv",
            mime="text/csv",
        )


st.header("Traffic analysis")

uploaded_file = st.file_uploader(
    "Upload a CSV of network feature records",
    type=["csv"],
    help=(
        "Use the same feature columns as the KDD Cup 99 HTTP "
        "subset. The model needs all expected feature columns."
    ),
)

if uploaded_file is None:
    st.info(
        "Upload a CSV to begin, or prepare the demo CSV from "
        "the sidebar and upload it here."
    )
    st.stop()

try:
    traffic_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if traffic_df.empty:
    st.warning("The uploaded CSV contains no records.")
    st.stop()

missing_columns = [
    column
    for column in expected_columns
    if column not in traffic_df.columns
]

if missing_columns:
    st.error(
        "This CSV is missing required feature columns: "
        + ", ".join(missing_columns)
    )
    st.write("Expected columns:", expected_columns)
    st.stop()

# Select and order the features exactly as the model expects.
X_input = traffic_df[expected_columns].copy()

st.subheader("Uploaded data")
st.write(f"Records uploaded: **{len(X_input):,}**")
st.dataframe(traffic_df.head(10), use_container_width=True)

if st.button("Analyze traffic", type="primary"):
    try:
        predictions = model.predict(X_input)
        probabilities = model.predict_proba(X_input)[:, 1]

        results = X_input.copy()

        results["prediction"] = [
            "Suspicious" if value == 1 else "Normal"
            for value in predictions
        ]

        results["suspicious_probability"] = probabilities

        suspicious_count = int((predictions == 1).sum())
        normal_count = int((predictions == 0).sum())

        st.session_state["analysis_results"] = results

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

if "analysis_results" in st.session_state:
    results = st.session_state["analysis_results"]

    suspicious_count = int(
        (results["prediction"] == "Suspicious").sum()
    )
    normal_count = int(
        (results["prediction"] == "Normal").sum()
    )

    st.divider()
    st.header("Analysis results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Records analyzed", f"{len(results):,}")
    col2.metric("Normal predictions", f"{normal_count:,}")
    col3.metric("Suspicious predictions", f"{suspicious_count:,}")

    st.subheader("Predicted traffic")
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    suspicious_results = results[
        results["prediction"] == "Suspicious"
    ]

    st.subheader("Suspicious records")
    if suspicious_results.empty:
        st.success("No records were classified as suspicious.")
    else:
        st.warning(
            f"{len(suspicious_results)} record(s) were classified "
            "as suspicious. Review them rather than treating "
            "these predictions as confirmed attacks."
        )
        st.dataframe(
            suspicious_results,
            use_container_width=True,
            hide_index=True,
        )

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Export predictions to CSV",
        data=csv_data,
        file_name="intrusion_predictions.csv",
        mime="text/csv",
    )

    st.caption(
        "Suspicious probability is the model's estimated probability "
        "for its suspicious class, not a guarantee of attack likelihood "
        "on real-world traffic."
    )