import io
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Banking Churn Management", layout="wide")

SAMPLE_DATA_PATH = "data/sample_churn_data.csv"


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH)


@st.cache_data
def load_uploaded_data(file: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(file)


def resolve_target(series: pd.Series) -> Tuple[pd.Series, str]:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int), "numeric"
    unique_values = list(series.dropna().unique())
    if len(unique_values) == 2:
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        return series.map(mapping).astype(int), "mapped"
    return series, "unsupported"


def build_preprocessor(
    df: pd.DataFrame, drop_columns: List[str]
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    feature_df = df.drop(columns=drop_columns)
    numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        col for col in feature_df.columns if col not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    feature_names = list(numeric_features)
    if categorical_features:
        encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_names = encoder.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(encoded_names)
    return feature_names


st.title("Banking Churn Management Dashboard")

st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload churn data (CSV)", type=["csv"])

sample_data = load_sample_data()

with open(SAMPLE_DATA_PATH, "rb") as sample_file:
    st.sidebar.download_button(
        label="Download sample data",
        data=sample_file,
        file_name="sample_churn_data.csv",
        mime="text/csv",
    )

if uploaded_file is not None:
    data = load_uploaded_data(uploaded_file)
    st.sidebar.success("Using uploaded data")
else:
    data = sample_data.copy()
    st.sidebar.info("Using bundled sample data")

st.subheader("Data Preview")
st.dataframe(data.head(20), use_container_width=True)

st.sidebar.header("Model Settings")
default_target = "churn" if "churn" in data.columns else data.columns[-1]
target_column = st.sidebar.selectbox("Target column", data.columns, index=data.columns.get_loc(default_target))

suggested_drop = [col for col in data.columns if "id" in col.lower() and col != target_column]
drop_columns = st.sidebar.multiselect(
    "Columns to drop", [col for col in data.columns if col != target_column], default=suggested_drop
)

n_estimators = st.sidebar.slider("Random forest trees", min_value=50, max_value=300, value=150, step=25)
max_depth = st.sidebar.slider("Max depth", min_value=3, max_value=15, value=6, step=1)

test_size = st.sidebar.slider("Test size", min_value=0.2, max_value=0.4, value=0.25, step=0.05)

if target_column in drop_columns:
    st.warning("Target column cannot be dropped. It has been removed from drop list.")
    drop_columns = [col for col in drop_columns if col != target_column]

if target_column not in data.columns:
    st.error("Target column missing from data. Please select a valid column.")
    st.stop()

target_series, target_mode = resolve_target(data[target_column])
if target_mode == "unsupported":
    st.error("Target column must be binary. Please upload data with a binary churn label.")
    st.stop()

feature_data = data.drop(columns=[target_column])
feature_data = feature_data.drop(columns=drop_columns)

if feature_data.empty:
    st.error("No features left after dropping columns. Please adjust selections.")
    st.stop()

preprocessor, numeric_features, categorical_features = build_preprocessor(
    feature_data, drop_columns=[]
)

X_train, X_test, y_train, y_test = train_test_split(
    feature_data, target_series, test_size=test_size, random_state=42, stratify=target_series
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)
proba = model.predict_proba(X_test_processed)[:, 1]

st.subheader("Model Performance Metrics")
metric_cols = st.columns(5)
metric_cols[0].metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
metric_cols[1].metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
metric_cols[2].metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
metric_cols[3].metric("F1", f"{f1_score(y_test, y_pred):.3f}")
metric_cols[4].metric("ROC AUC", f"{roc_auc_score(y_test, proba):.3f}")

st.markdown("**Classification Report**")
st.code(classification_report(y_test, y_pred), language="text")

plot_cols = st.columns(2)
with plot_cols[0]:
    st.markdown("**Confusion Matrix**")
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, cmap="Blues")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with plot_cols[1]:
    st.markdown("**ROC Curve**")
    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax_roc)
    st.pyplot(fig_roc)
    plt.close(fig_roc)

st.subheader("SHAP Explanations")
feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_processed)
shap_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values

fig_shap = plt.figure()
shap.summary_plot(shap_class1, X_test_processed, feature_names=feature_names, show=False)
st.pyplot(fig_shap)
plt.close(fig_shap)

mean_abs_shap = np.abs(shap_class1).mean(axis=0)
importance = pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap})
importance = importance.sort_values("importance", ascending=False).head(5)

st.markdown("**Key Drivers of Churn (SHAP)**")
for _, row in importance.iterrows():
    st.write(
        f"- **{row['feature']}** shows a strong influence on churn predictions "
        f"(mean |SHAP| = {row['importance']:.4f})."
    )

st.markdown(
    """
**Inference guidance:**
- Higher positive SHAP values push the model toward predicting churn.
- Negative SHAP values indicate factors that reduce churn risk.
- Focus retention actions on the features with the largest mean |SHAP| values.
"""
)
