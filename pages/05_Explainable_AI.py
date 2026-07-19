"""GEOInsightLab — Explainable AI Streamlit page."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import column_label, load_data
from utils.theme import apply_theme, page_header, scientific_note

try:
    from utils.maps import render_categorical_polygon_map
except ImportError:
    render_categorical_polygon_map = None

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, export_text
    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestClassifier = None
    DecisionTreeClassifier = None
    export_text = None
    train_test_split = None
    accuracy_score = balanced_accuracy_score = classification_report = None
    confusion_matrix = f1_score = precision_score = recall_score = None
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LimeTabularExplainer = None
    LIME_AVAILABLE = False

apply_theme()
page_header(
    "Explainable AI",
    "Understand how Decision Trees, Random Forest and LIME explain the urban profiles identified by K-means.",
)
scientific_note(
    "K-means cluster labels are used as the target. Supervised models reproduce and explain the clustering structure, following the GEOInsightLab thesis workflow."
)

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

MODEL_PATHS = {
    "Decision Tree": [MODELS_DIR / "decision_tree.joblib", MODELS_DIR / "decision_tree.pkl"],
    "Random Forest": [MODELS_DIR / "random_forest.joblib", MODELS_DIR / "random_forest.pkl"],
    "XGBoost": [
        MODELS_DIR / "xgboost_model.joblib",
        MODELS_DIR / "xgboost_model.pkl",
        MODELS_DIR / "xgboost_model.json",
    ],
}

SCALER_PATHS = [
    MODELS_DIR / "scaler.joblib",
    MODELS_DIR / "standard_scaler.joblib",
    MODELS_DIR / "preprocessor.joblib",
]

TARGET_ALIASES = [
    "cluster_kmeans",
    "cluster_gmm",
    "cluster_agglo",
    "urban_profile",
    "perfil_urbano",
    "cluster_label",
    "cluster",
    "Cluster",
    "profile",
]

# Variables used in the thesis to explain the K-means clustering.
DEFAULT_FEATURES = [
    "numero_servicos_proximos",
    "distancia_media_servicos",
    "pop_64_mais",
]

# Keep empty so that the published thesis model is not silently changed.
OPTIONAL_FEATURES: list[str] = []
DISPLAY_LABELS = {
    "Bancos": "Banks",
    "Centro Saude": "Health centres",
    "Farmacias": "Pharmacies",
    "Supermercados": "Supermarkets",
    "Parques ou jardins": "Parks and gardens",
    "Parques e jardins": "Parks and gardens",
    "Hospitais": "Hospitals",
    "CTT": "Post offices",
    "pop_64_mais": "Population aged 65+",
    "distancia_media_servicos": "Mean distance to services",
    "numero_servicos_proximos": "Nearby services",
    "numero_categorias_proximas": "Nearby service categories",
}
PROFILE_COLOURS = [
    [31, 119, 180, 190], [255, 127, 14, 190], [44, 160, 44, 190],
    [214, 39, 40, 190], [148, 103, 189, 190], [140, 86, 75, 190],
    [227, 119, 194, 190], [127, 127, 127, 190],
]


@dataclass
class ModelBundle:
    name: str
    model: Any
    feature_columns: list[str]
    class_labels: list[str]
    scaler: Any | None = None
    source: str = "pretrained model"
    x_train: pd.DataFrame | None = None
    x_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None


def display_label(field: str) -> str:
    return DISPLAY_LABELS.get(field, column_label(field))


def normalise_class(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    try:
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def class_mapping(metadata: dict[str, Any]) -> dict[str, str]:
    value = metadata.get("class_labels", {})
    if not isinstance(value, dict):
        return {}
    return {normalise_class(key): str(label) for key, label in value.items()}


def readable_profile(value: Any, mapping: dict[str, str]) -> str:
    key = normalise_class(value)
    return mapping.get(key, f"Profile {key}")


def resolve_target(dataframe: pd.DataFrame, metadata: dict[str, Any]) -> str | None:
    configured = metadata.get("target")
    if configured in dataframe.columns:
        return str(configured)
    return next((field for field in TARGET_ALIASES if field in dataframe.columns), None)


def resolve_features(dataframe: pd.DataFrame, metadata: dict[str, Any]) -> list[str]:
    configured = metadata.get("features")
    if isinstance(configured, list):
        valid = [str(field) for field in configured if str(field) in dataframe.columns]
        if valid:
            return valid

    fields: list[str] = []
    for field in DEFAULT_FEATURES:
        if field in dataframe.columns:
            fields.append(field)
        elif field == "Parques ou jardins" and "Parques e jardins" in dataframe.columns:
            fields.append("Parques e jardins")
    fields.extend(field for field in OPTIONAL_FEATURES if field in dataframe.columns)
    return list(dict.fromkeys(fields))


def clean_features(dataframe: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    result = dataframe[fields].copy()
    for field in fields:
        result[field] = pd.to_numeric(result[field], errors="coerce")
        median = result[field].median()
        result[field] = result[field].fillna(0.0 if pd.isna(median) else median)
    return result


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def load_scaler() -> Any | None:
    path = first_existing(SCALER_PATHS)
    if path is None or not JOBLIB_AVAILABLE:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_model(name: str) -> Any | None:
    path = first_existing(MODEL_PATHS[name])
    if path is None:
        return None
    if path.suffix.lower() == ".json" and name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            return None
        model = XGBClassifier()
        model.load_model(path)
        return model
    if not JOBLIB_AVAILABLE:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def estimator_classes(model: Any) -> list[str]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(list(model.named_steps.values())[-1], "classes_", None)
    return [] if classes is None else [normalise_class(value) for value in classes]


def transform(bundle: ModelBundle, frame: pd.DataFrame) -> Any:
    selected = frame[bundle.feature_columns]
    return selected if bundle.scaler is None else bundle.scaler.transform(selected)


def predict(bundle: ModelBundle, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    values = transform(bundle, frame)
    raw = bundle.model.predict(values)
    labels = np.asarray([normalise_class(value) for value in raw])
    probabilities = (
        np.asarray(bundle.model.predict_proba(values), dtype=float)
        if hasattr(bundle.model, "predict_proba")
        else None
    )
    return labels, probabilities


def colour_mapping(classes: list[str]) -> dict[str, list[int]]:
    return {name: PROFILE_COLOURS[index % len(PROFILE_COLOURS)] for index, name in enumerate(classes)}


def render_map(dataframe: pd.DataFrame, category: str, colours: dict[str, list[int]], maximum: int) -> None:
    if render_categorical_polygon_map is None:
        st.info("The categorical polygon-map utility is unavailable.")
        return
    parameters = set(inspect.signature(render_categorical_polygon_map).parameters)
    kwargs: dict[str, Any] = {}
    for candidate in ("dataframe", "df", "data"):
        if candidate in parameters:
            kwargs[candidate] = dataframe
            break
    for candidate in ("value_column", "category_column", "column", "colour_column", "color_column"):
        if candidate in parameters:
            kwargs[candidate] = category
            break
    for candidate in ("colour_mapping", "color_mapping", "colour_map", "color_map", "colours", "colors"):
        if candidate in parameters:
            kwargs[candidate] = colours
            break
    for candidate in ("maximum_polygons", "max_polygons", "polygon_limit", "maximum_features"):
        if candidate in parameters:
            kwargs[candidate] = maximum
            break
    try:
        render_categorical_polygon_map(**kwargs)
    except Exception as error:
        st.warning(f"The map could not be rendered: {error}")


def importance_table(bundle: ModelBundle) -> pd.DataFrame:
    model = bundle.model
    if hasattr(model, "named_steps"):
        model = list(model.named_steps.values())[-1]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    else:
        return pd.DataFrame()
    values = np.abs(values)
    if values.sum() > 0:
        values = values / values.sum()
    return pd.DataFrame({
        "Feature": [display_label(field) for field in bundle.feature_columns],
        "Importance": values,
    }).sort_values("Importance", ascending=False)


def local_effects(bundle: ModelBundle, row: pd.DataFrame, reference: pd.DataFrame, predicted: str) -> pd.DataFrame:
    if not hasattr(bundle.model, "predict_proba"):
        return pd.DataFrame()
    classes = bundle.class_labels or estimator_classes(bundle.model)
    try:
        position = classes.index(normalise_class(predicted))
    except ValueError:
        return pd.DataFrame()
    baseline = float(bundle.model.predict_proba(transform(bundle, row))[0, position])
    rows = []
    for field in bundle.feature_columns:
        modified = row[bundle.feature_columns].copy()
        median = float(reference[field].median())
        modified.loc[modified.index[0], field] = median
        probability = float(bundle.model.predict_proba(transform(bundle, modified))[0, position])
        rows.append({
            "Feature": display_label(field),
            "Building value": float(row.iloc[0][field]),
            "Reference value": median,
            "Local contribution": baseline - probability,
        })
    return pd.DataFrame(rows).sort_values(
        "Local contribution", key=lambda series: series.abs(), ascending=False
    )


def decision_path(bundle: ModelBundle, row: pd.DataFrame) -> list[str]:
    if not SKLEARN_AVAILABLE:
        return []
    model = bundle.model
    if hasattr(model, "named_steps"):
        model = list(model.named_steps.values())[-1]
    if not isinstance(model, DecisionTreeClassifier):
        return []
    array = np.asarray(transform(bundle, row))
    indicator = model.decision_path(array)
    leaf = int(model.apply(array)[0])
    nodes = indicator.indices[indicator.indptr[0]:indicator.indptr[1]]
    lines = []
    for node in nodes:
        if node == leaf:
            lines.append(f"Leaf node {node}")
            continue
        feature_index = int(model.tree_.feature[node])
        if feature_index < 0:
            continue
        field = bundle.feature_columns[feature_index]
        observed = float(array[0, feature_index])
        threshold = float(model.tree_.threshold[node])
        operator = "≤" if observed <= threshold else ">"
        lines.append(f"{display_label(field)}: {observed:.3f} {operator} {threshold:.3f}")
    return lines


def shap_values(bundle: ModelBundle, row: pd.DataFrame) -> pd.DataFrame:
    if not SHAP_AVAILABLE:
        return pd.DataFrame()
    try:
        explainer = shap.TreeExplainer(bundle.model)
        values = explainer.shap_values(transform(bundle, row))
    except Exception:
        return pd.DataFrame()
    array = np.asarray(values[0] if isinstance(values, list) else values, dtype=float)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2 or array.shape[1] != len(bundle.feature_columns):
        return pd.DataFrame()
    return pd.DataFrame({
        "Feature": [display_label(field) for field in bundle.feature_columns],
        "SHAP value": array[0],
    }).sort_values("SHAP value", key=lambda series: series.abs(), ascending=False)


def lime_values(bundle: ModelBundle, row: pd.DataFrame, reference: pd.DataFrame, predicted: str) -> pd.DataFrame:
    if not LIME_AVAILABLE or not hasattr(bundle.model, "predict_proba"):
        return pd.DataFrame()
    training = np.asarray(transform(bundle, reference), dtype=float)
    selected = np.asarray(transform(bundle, row), dtype=float)[0]
    classes = bundle.class_labels or estimator_classes(bundle.model)
    try:
        position = classes.index(normalise_class(predicted))
    except ValueError:
        return pd.DataFrame()
    explainer = LimeTabularExplainer(
        training_data=training,
        feature_names=[display_label(field) for field in bundle.feature_columns],
        class_names=classes,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    explanation = explainer.explain_instance(
        selected,
        bundle.model.predict_proba,
        labels=[position],
        num_features=min(10, len(bundle.feature_columns)),
    )
    return pd.DataFrame(explanation.as_list(label=position), columns=["Condition", "Contribution"])


@st.cache_resource(show_spinner=False)
def train_models(features: pd.DataFrame, target: pd.Series, test_size: float, depth: int, trees: int) -> dict[str, ModelBundle]:
    if not SKLEARN_AVAILABLE:
        return {}
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=RANDOM_STATE, stratify=target
    )
    estimators: dict[str, Any] = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    bundles: dict[str, ModelBundle] = {}
    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        bundles[name] = ModelBundle(
            name=name,
            model=estimator,
            feature_columns=list(features.columns),
            class_labels=[normalise_class(value) for value in estimator.classes_],
            source="temporary training",
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
    return bundles


def evaluate(bundle: ModelBundle) -> dict[str, float] | None:
    if not SKLEARN_AVAILABLE or bundle.x_test is None or bundle.y_test is None:
        return None
    predictions, _ = predict(bundle, bundle.x_test)
    return {
        "Accuracy": accuracy_score(bundle.y_test, predictions),
        "Balanced accuracy": balanced_accuracy_score(bundle.y_test, predictions),
        "Precision": precision_score(bundle.y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(bundle.y_test, predictions, average="weighted", zero_division=0),
        "F1-score": f1_score(bundle.y_test, predictions, average="weighted", zero_division=0),
    }


try:
    data = load_data()
except Exception as error:
    st.error(f"The research dataset could not be loaded: {error}")
    st.stop()

metadata = load_metadata()
mapping = class_mapping(metadata)
target_column = resolve_target(data, metadata)
feature_columns = resolve_features(data, metadata)

if len(feature_columns) < 3:
    st.error("The dataset does not contain enough predictor variables.")
    st.stop()

features = clean_features(data, feature_columns)
scaler = load_scaler()
pretrained: dict[str, ModelBundle] = {}

for model_name in MODEL_PATHS:
    model = load_model(model_name)
    if model is not None:
        pretrained[model_name] = ModelBundle(
            name=model_name,
            model=model,
            feature_columns=feature_columns,
            class_labels=estimator_classes(model),
            scaler=scaler,
        )

prepared_target = None
if target_column is not None:
    # Train with the original cluster identifiers (for example, 0 and 1).
    # Human-readable profile names are applied only when results are displayed.
    prepared_target = data[target_column].map(normalise_class)

with st.sidebar:
    st.subheader("Explainable AI controls")
    allow_training = st.toggle("Train temporary models when no artefacts are found", value=True)
    test_percentage = st.slider("Test sample (%)", 15, 40, 25, 5, disabled=not allow_training)
    tree_depth = st.slider("Decision-tree maximum depth", 2, 15, 6, 1, disabled=not allow_training)
    forest_trees = st.slider("Random-forest trees", 100, 700, 300, 100, disabled=not allow_training)
    maximum_polygons = st.slider("Maximum building polygons on the map", 1000, 15000, 6000, 1000)

fallback: dict[str, ModelBundle] = {}
if not pretrained and allow_training and prepared_target is not None:
    if not SKLEARN_AVAILABLE:
        st.error("Add `scikit-learn` and `joblib` to requirements.txt.")
        st.stop()
    valid = prepared_target.ne("Unknown")
    target = prepared_target.loc[valid]
    training_features = features.loc[valid]
    counts = target.value_counts()
    accepted = counts[counts >= 2].index
    mask = target.isin(accepted)
    target = target.loc[mask]
    training_features = training_features.loc[mask]
    if target.nunique() < 2:
        st.error("The target column must contain at least two usable classes.")
        st.stop()
    with st.spinner("Training temporary classification models..."):
        fallback = train_models(training_features, target, test_percentage / 100, tree_depth, forest_trees)

bundles = pretrained or fallback
if not bundles:
    st.error("No usable model was found. Add model artefacts to the `models` folder or enable temporary training.")
    st.code(
        "models/\n├── decision_tree.joblib\n├── random_forest.joblib\n├── xgboost_model.joblib\n├── scaler.joblib\n└── model_metadata.json",
        language="text",
    )
    st.stop()

with st.sidebar:
    selected_model_name = st.selectbox("Prediction model", list(bundles), index=min(1, len(bundles) - 1))

bundle = bundles[selected_model_name]
try:
    raw_predictions, probabilities = predict(bundle, features)
except Exception as error:
    st.error(f"The model could not process the dataset. Check feature order and preprocessing. Details: {error}")
    st.stop()

result = data.copy()
result["predicted_profile"] = [readable_profile(value, mapping) for value in raw_predictions]
if prepared_target is not None:
    result["observed_profile"] = prepared_target.map(lambda value: readable_profile(value, mapping))
result["prediction_confidence"] = probabilities.max(axis=1) if probabilities is not None else np.nan
classes = sorted(result["predicted_profile"].dropna().unique().tolist())
colours = colour_mapping(classes)

summary = st.columns(5)
summary[0].metric("Buildings", f"{len(result):,}")
summary[1].metric("Predicted profiles", len(classes))
summary[2].metric("Model", selected_model_name)
summary[3].metric("Source", bundle.source.title())
summary[4].metric("Mean confidence", f"{result['prediction_confidence'].mean():.3f}" if probabilities is not None else "Not available")

prediction_tab, global_tab, local_tab, comparison_tab, methodology_tab = st.tabs([
    "Building prediction", "Global explanations", "Local explanations", "Model comparison", "Methodology"
])

with prediction_tab:
    st.subheader(f"{selected_model_name} predictions")
    render_map(result, "predicted_profile", colours, maximum_polygons)
    st.caption("The map uses the original building polygons. Colours represent the predicted urban profile.")
    distribution = result["predicted_profile"].value_counts().rename_axis("Predicted profile").reset_index(name="Buildings")
    chart = px.bar(distribution, x="Predicted profile", y="Buildings", title="Distribution of predicted urban profiles")
    st.plotly_chart(chart, width="stretch")
    if probabilities is not None:
        confidence = px.histogram(
            result,
            x="prediction_confidence",
            color="predicted_profile",
            nbins=35,
            opacity=0.75,
            title="Prediction-confidence distribution",
        )
        confidence.update_layout(bargap=0.04, legend_title="Predicted profile")
        st.plotly_chart(confidence, width="stretch")

with global_tab:
    st.subheader(f"Global explanation — {selected_model_name}")
    importance = importance_table(bundle)
    if importance.empty:
        st.info("Global feature importance is not exposed by this model.")
    else:
        chart = px.bar(
            importance.sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Relative contribution of predictor variables",
        )
        st.plotly_chart(chart, width="stretch")
        table = importance.copy()
        table["Importance (%)"] = (table["Importance"] * 100).round(2)
        st.dataframe(table[["Feature", "Importance (%)"]], width="stretch", hide_index=True)
    if SKLEARN_AVAILABLE and selected_model_name == "Decision Tree":
        estimator = bundle.model
        if hasattr(estimator, "named_steps"):
            estimator = list(estimator.named_steps.values())[-1]
        if isinstance(estimator, DecisionTreeClassifier):
            st.subheader("Decision-tree rules")
            st.code(export_text(estimator, feature_names=[display_label(field) for field in bundle.feature_columns], decimals=3), language="text")

with local_tab:
    st.subheader("Building-level prediction inspection")
    identifier = next((field for field in ("osm_id", "building_id", "id") if field in result.columns), None)
    if identifier is None:
        result["_building_identifier"] = result.index.astype(str)
        identifier = "_building_identifier"
    identifiers = result[identifier].astype(str).tolist()
    selected_identifier = st.selectbox("Select a building", identifiers)
    position = identifiers.index(selected_identifier)
    selected_row = features.iloc[[position]]
    selected_record = result.iloc[position]

    cards = st.columns(4)
    cards[0].metric("Building", selected_identifier)
    cards[1].metric("Observed profile", str(selected_record.get("observed_profile", "Not available")))
    cards[2].metric("Predicted profile", str(selected_record["predicted_profile"]))
    cards[3].metric("Confidence", f"{float(selected_record['prediction_confidence']):.1%}" if probabilities is not None else "Not available")

    if probabilities is not None:
        raw_classes = bundle.class_labels or estimator_classes(bundle.model)
        probability_table = pd.DataFrame({
            "Urban profile": [readable_profile(value, mapping) for value in raw_classes],
            "Probability": probabilities[position],
        }).sort_values("Probability", ascending=False)
        probability_chart = px.bar(
            probability_table.sort_values("Probability", ascending=True),
            x="Probability",
            y="Urban profile",
            orientation="h",
            title="Predicted class probabilities",
        )
        probability_chart.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(probability_chart, width="stretch")

    reference = bundle.x_train if bundle.x_train is not None else features
    st.subheader("Local perturbation explanation")
    effects = local_effects(bundle, selected_row, reference, raw_predictions[position])
    if effects.empty:
        st.info("Local probability perturbations are unavailable for this model.")
    else:
        effects_chart = px.bar(
            effects.sort_values("Local contribution", ascending=True),
            x="Local contribution",
            y="Feature",
            orientation="h",
            hover_data=["Building value", "Reference value"],
            title="Probability change after replacing each variable with its median",
        )
        effects_chart.add_vline(x=0, line_width=1)
        st.plotly_chart(effects_chart, width="stretch")
        st.dataframe(effects.round(4), width="stretch", hide_index=True)

    if selected_model_name == "Decision Tree":
        st.subheader("Decision path")
        lines = decision_path(bundle, selected_row)
        if lines:
            for number, line in enumerate(lines, 1):
                st.markdown(f"**{number}.** {line}")
        else:
            st.info("A decision path could not be extracted.")

    st.subheader("SHAP explanation")
    if not SHAP_AVAILABLE:
        st.info("Add `shap` to requirements.txt to activate SHAP explanations.")
    else:
        table = shap_values(bundle, selected_row)
        if table.empty:
            st.info("SHAP values could not be calculated for this model.")
        else:
            chart = px.bar(table.sort_values("SHAP value", ascending=True), x="SHAP value", y="Feature", orientation="h")
            chart.add_vline(x=0, line_width=1)
            st.plotly_chart(chart, width="stretch")
            st.dataframe(table.round(4), width="stretch", hide_index=True)

    st.subheader("LIME explanation")
    if not LIME_AVAILABLE:
        st.info("Add `lime` to requirements.txt to activate LIME explanations.")
    else:
        try:
            table = lime_values(bundle, selected_row, reference, raw_predictions[position])
        except Exception as error:
            table = pd.DataFrame()
            st.warning(f"The LIME explanation could not be generated: {error}")
        if not table.empty:
            chart = px.bar(table.sort_values("Contribution", ascending=True), x="Contribution", y="Condition", orientation="h")
            chart.add_vline(x=0, line_width=1)
            st.plotly_chart(chart, width="stretch")
            st.dataframe(table.round(4), width="stretch", hide_index=True)

with comparison_tab:
    st.subheader("Model comparison")
    rows = []
    for name, candidate in bundles.items():
        metrics = evaluate(candidate)
        if metrics is not None:
            rows.append({"Model": name, **metrics})
    if not rows:
        st.info("Comparable test metrics are available for models trained during the current session.")
    else:
        table = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
        st.dataframe(table.round(3), width="stretch", hide_index=True)
        long = table.melt(id_vars="Model", var_name="Metric", value_name="Score")
        chart = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", title="Predictive performance by model")
        chart.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(chart, width="stretch")
        if bundle.x_test is not None and bundle.y_test is not None and SKLEARN_AVAILABLE:
            predicted, _ = predict(bundle, bundle.x_test)
            labels = sorted(set(bundle.y_test.astype(str)) | set(predicted.tolist()))
            matrix = confusion_matrix(bundle.y_test, predicted, labels=labels)
            figure = px.imshow(matrix, x=labels, y=labels, text_auto=True, aspect="auto", title=f"Confusion matrix — {selected_model_name}")
            st.plotly_chart(figure, width="stretch")
            report = pd.DataFrame(classification_report(bundle.y_test, predicted, output_dict=True, zero_division=0)).transpose().reset_index().rename(columns={"index": "Class"})
            st.dataframe(report.round(3), width="stretch", hide_index=True)

with methodology_tab:
    st.subheader("Analytical workflow")
    st.markdown(
        """
### Prediction objective

The models reproduce the K-means building clusters from the three variables used in the thesis: nearby services, mean distance to services and population aged 65 or over. The page combines prediction with global and local explanation.

### Model artefacts

The application first searches the `models` directory for pretrained artefacts. When none are available, it trains temporary Decision Tree and Random Forest models directly from `cluster_kmeans` and the three thesis variables.

### Fallback training

When no pretrained artefact is present, the page can train temporary Decision Tree and Random Forest models from the current dataset. This mode is intended for testing. The published application should use the exact fitted artefacts produced by the thesis workflow.

### Explanations

- Global importance summarises the variables used most by the model.
- Local perturbation replaces one variable with the reference median.
- Decision paths trace the thresholds followed by one building.
- SHAP estimates local model-specific contributions.
- LIME creates a local surrogate explanation.

### Interpretation limits

These explanations describe model behaviour and do not establish causal effects.
        """
    )

    st.subheader("Variables used by the selected model")
    st.dataframe(pd.DataFrame({
        "Dataset field": bundle.feature_columns,
        "Indicator": [display_label(field) for field in bundle.feature_columns],
    }), width="stretch", hide_index=True)

    st.subheader("Environment status")
    st.dataframe(pd.DataFrame({
        "Component": ["scikit-learn", "joblib", "XGBoost", "SHAP", "LIME", "Pretrained model", "Scaler", "Model metadata"],
        "Available": [SKLEARN_AVAILABLE, JOBLIB_AVAILABLE, XGBOOST_AVAILABLE, SHAP_AVAILABLE, LIME_AVAILABLE, bool(pretrained), scaler is not None, bool(metadata)],
    }), width="stretch", hide_index=True)

    download_columns = [field for field in [
        "osm_id", "designacao_simplificada", target_column, "observed_profile",
        "predicted_profile", "prediction_confidence", *bundle.feature_columns,
        "latitude", "longitude",
    ] if field and field in result.columns]

    st.download_button(
        "Download Explainable AI results",
        data=result[download_columns].to_csv(index=False).encode("utf-8"),
        file_name="geoinsightlab_explainable_ai.csv",
        mime="text/csv",
    )
