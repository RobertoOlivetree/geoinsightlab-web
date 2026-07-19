import inspect
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from utils.data_loader import column_label, load_data
from utils.maps import render_categorical_polygon_map
from utils.theme import apply_theme, page_header, scientific_note


# ============================================================
# Optional dependencies
# ============================================================

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


try:
    from lime.lime_tabular import LimeTabularExplainer

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


# ============================================================
# Page configuration
# ============================================================

apply_theme()

page_header(
    "Explainable AI",
    (
        "Understand why a building is assigned to a specific urban profile "
        "and identify the variables that drive the prediction."
    ),
)

scientific_note(
    (
        "This module compares interpretable and ensemble classification "
        "models and provides global and local explanations of building-level "
        "urban-profile predictions."
    )
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

TARGET_ALIASES = [
    "cluster",
    "Cluster",
    "cluster_label",
    "perfil_urbano",
    "urban_profile",
    "profile",
]

PARK_ALIASES = [
    "Parques ou jardins",
    "Parques e jardins",
]

BASE_FEATURES = [
    "Bancos",
    "Centro Saude",
    "Farmacias",
    "Supermercados",
    "Hospitais",
    "CTT",
    "pop_64_mais",
    "distancia_media_servicos",
]

OPTIONAL_FEATURES = [
    "numero_servicos_proximos",
    "numero_categorias_proximas",
]

DISPLAY_LABELS = {
    "Bancos": "Banks",
    "Centro Saude": "Health centres",
    "Farmacias": "Pharmacies",
    "Supermercados": "Supermarkets",
    "Hospitais": "Hospitals",
    "CTT": "Post offices",
    "pop_64_mais": "Population aged 65+",
    "distancia_media_servicos": "Mean distance to services",
    "numero_servicos_proximos": "Nearby services",
    "numero_categorias_proximas": "Nearby service categories",
    "Parques ou jardins": "Parks and gardens",
    "Parques e jardins": "Parks and gardens",
}

MODEL_LABELS = {
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}

PROFILE_COLOURS = [
    [31, 119, 180, 190],
    [255, 127, 14, 190],
    [44, 160, 44, 190],
    [214, 39, 40, 190],
    [148, 103, 189, 190],
    [140, 86, 75, 190],
    [227, 119, 194, 190],
    [127, 127, 127, 190],
]


# ============================================================
# Utility functions
# ============================================================

def display_label(field: str) -> str:
    return DISPLAY_LABELS.get(
        field,
        column_label(field),
    )


def resolve_target_column(
    dataframe: pd.DataFrame,
) -> str:
    for candidate in TARGET_ALIASES:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        "No urban-profile target column was found. "
        "Expected one of: "
        + ", ".join(TARGET_ALIASES)
    )


def resolve_park_column(
    dataframe: pd.DataFrame,
) -> str | None:
    for candidate in PARK_ALIASES:
        if candidate in dataframe.columns:
            return candidate

    return None


def resolve_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    features = [
        feature
        for feature in BASE_FEATURES
        if feature in dataframe.columns
    ]

    park_column = resolve_park_column(dataframe)

    if park_column is not None:
        features.append(park_column)

    for feature in OPTIONAL_FEATURES:
        if feature in dataframe.columns:
            features.append(feature)

    if len(features) < 3:
        raise ValueError(
            "The dataset does not contain enough predictor variables "
            "to train the Explainable AI models."
        )

    return features


def clean_target(
    series: pd.Series,
) -> pd.Series:
    target = series.copy()

    if pd.api.types.is_numeric_dtype(target):
        target = target.fillna(-1)
        target = target.astype(int)
        target = target.astype(str)
    else:
        target = target.fillna("Unknown")
        target = target.astype(str)

    target = target.str.strip()

    return target


def clean_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    features = dataframe[
        feature_columns
    ].copy()

    for column in feature_columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

        median = features[column].median()

        if pd.isna(median):
            median = 0.0

        features[column] = features[column].fillna(
            median
        )

    return features


def create_profile_labels(
    target: pd.Series,
) -> tuple[pd.Series, dict[str, str]]:
    unique_values = sorted(
        target.unique().tolist()
    )

    label_mapping = {}

    for value in unique_values:
        value_string = str(value)

        if value_string.lower().startswith("cluster"):
            readable_label = value_string
        elif value_string.lower().startswith("profile"):
            readable_label = value_string
        else:
            readable_label = f"Profile {value_string}"

        label_mapping[value_string] = readable_label

    readable_target = target.map(
        label_mapping
    )

    return readable_target, label_mapping


def build_colour_mapping(
    classes: list[str],
) -> dict[str, list[int]]:
    colour_mapping = {}

    for position, class_name in enumerate(classes):
        colour_mapping[class_name] = PROFILE_COLOURS[
            position % len(PROFILE_COLOURS)
        ]

    return colour_mapping


def render_categorical_map_compat(
    dataframe: pd.DataFrame,
    category_column: str,
    colour_mapping: dict[str, list[int]],
    maximum_polygons: int,
) -> None:
    """
    Call the existing map utility while accommodating different
    parameter names used in previous GEOInsightLab versions.
    """

    function_signature = inspect.signature(
        render_categorical_polygon_map
    )

    available_parameters = set(
        function_signature.parameters
    )

    arguments: dict[str, Any] = {}

    dataframe_names = [
        "dataframe",
        "df",
        "data",
    ]

    category_names = [
        "value_column",
        "category_column",
        "column",
        "colour_column",
        "color_column",
    ]

    colour_names = [
        "colour_mapping",
        "color_mapping",
        "colour_map",
        "color_map",
        "colours",
        "colors",
    ]

    maximum_names = [
        "maximum_polygons",
        "max_polygons",
        "polygon_limit",
        "maximum_features",
    ]

    for name in dataframe_names:
        if name in available_parameters:
            arguments[name] = dataframe
            break

    for name in category_names:
        if name in available_parameters:
            arguments[name] = category_column
            break

    for name in colour_names:
        if name in available_parameters:
            arguments[name] = colour_mapping
            break

    for name in maximum_names:
        if name in available_parameters:
            arguments[name] = maximum_polygons
            break

    if not arguments:
        raise TypeError(
            "The categorical map function has an unsupported signature."
        )

    render_categorical_polygon_map(
        **arguments
    )


def normalise_importances(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.abs(values)

    total = values.sum()

    if total <= 0:
        return np.zeros_like(values)

    return values / total


def get_model_estimator(
    model: Any,
) -> Any:
    if isinstance(model, Pipeline):
        return model.named_steps[
            "classifier"
        ]

    return model


def predict_probabilities(
    model: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The selected model does not provide class probabilities."
        )

    return model.predict_proba(
        features
    )


def model_classes(
    model: Any,
) -> np.ndarray:
    estimator = get_model_estimator(
        model
    )

    return np.asarray(
        estimator.classes_
    )


def calculate_permutation_local_effects(
    model: Any,
    selected_row: pd.DataFrame,
    reference_data: pd.DataFrame,
    predicted_class: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    classes = model_classes(
        model
    )

    class_positions = {
        str(class_name): position
        for position, class_name in enumerate(classes)
    }

    class_position = class_positions[
        str(predicted_class)
    ]

    baseline_probability = predict_probabilities(
        model,
        selected_row,
    )[0, class_position]

    effects = []

    for feature in feature_columns:
        modified_row = selected_row.copy()

        reference_value = reference_data[
            feature
        ].median()

        modified_row.loc[
            modified_row.index[0],
            feature,
        ] = reference_value

        modified_probability = predict_probabilities(
            model,
            modified_row,
        )[0, class_position]

        effect = (
            baseline_probability
            - modified_probability
        )

        effects.append(
            {
                "Feature": display_label(feature),
                "Building value": float(
                    selected_row.iloc[0][feature]
                ),
                "Reference value": float(
                    reference_value
                ),
                "Local contribution": float(
                    effect
                ),
            }
        )

    result = pd.DataFrame(
        effects
    )

    return result.sort_values(
        "Local contribution",
        key=lambda series: series.abs(),
        ascending=False,
    )


def decision_path_text(
    model: Any,
    selected_row: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    estimator = get_model_estimator(
        model
    )

    if not isinstance(
        estimator,
        DecisionTreeClassifier,
    ):
        return []

    row_array = selected_row[
        feature_columns
    ].to_numpy()

    node_indicator = estimator.decision_path(
        row_array
    )

    leaf_id = estimator.apply(
        row_array
    )[0]

    feature_indices = estimator.tree_.feature
    thresholds = estimator.tree_.threshold

    path_nodes = node_indicator.indices[
        node_indicator.indptr[0]:
        node_indicator.indptr[1]
    ]

    statements = []

    for node_id in path_nodes:
        if node_id == leaf_id:
            statements.append(
                f"Leaf node {node_id}"
            )
            continue

        feature_position = feature_indices[
            node_id
        ]

        if feature_position < 0:
            continue

        feature_name = feature_columns[
            feature_position
        ]

        threshold = thresholds[
            node_id
        ]

        observed_value = float(
            selected_row.iloc[0][
                feature_name
            ]
        )

        if observed_value <= threshold:
            operator = "≤"
        else:
            operator = ">"

        statements.append(
            (
                f"{display_label(feature_name)}: "
                f"{observed_value:.3f} {operator} "
                f"{threshold:.3f}"
            )
        )

    return statements


def generate_lime_explanation(
    model: Any,
    training_features: pd.DataFrame,
    selected_row: pd.DataFrame,
    feature_columns: list[str],
    predicted_class: str,
) -> pd.DataFrame:
    if not LIME_AVAILABLE:
        raise ImportError(
            "The lime package is not installed."
        )

    classes = [
        str(class_name)
        for class_name in model_classes(model)
    ]

    predicted_position = classes.index(
        str(predicted_class)
    )

    explainer = LimeTabularExplainer(
        training_data=training_features.to_numpy(
            dtype=float
        ),
        feature_names=[
            display_label(feature)
            for feature in feature_columns
        ],
        class_names=classes,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    explanation = explainer.explain_instance(
        data_row=selected_row[
            feature_columns
        ].iloc[0].to_numpy(
            dtype=float
        ),
        predict_fn=lambda array: predict_probabilities(
            model,
            pd.DataFrame(
                array,
                columns=feature_columns,
            ),
        ),
        labels=[
            predicted_position
        ],
        num_features=min(
            10,
            len(feature_columns),
        ),
    )

    explanation_rows = explanation.as_list(
        label=predicted_position
    )

    return pd.DataFrame(
        explanation_rows,
        columns=[
            "Condition",
            "Contribution",
        ],
    )


# ============================================================
# Model training
# ============================================================

@st.cache_resource(show_spinner=False)
def train_models(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    maximum_depth: int,
    forest_trees: int,
) -> dict[str, Any]:
    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    models: dict[str, Any] = {}

    decision_tree = DecisionTreeClassifier(
        max_depth=maximum_depth,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    random_forest = RandomForestClassifier(
        n_estimators=forest_trees,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    models["Decision Tree"] = decision_tree
    models["Random Forest"] = random_forest

    if XGBOOST_AVAILABLE:
        unique_classes = sorted(
            target.unique().tolist()
        )

        class_to_integer = {
            class_name: position
            for position, class_name
            in enumerate(unique_classes)
        }

        integer_to_class = {
            position: class_name
            for class_name, position
            in class_to_integer.items()
        }

        y_train_integer = y_train.map(
            class_to_integer
        )

        y_test_integer = y_test.map(
            class_to_integer
        )

        if len(unique_classes) == 2:
            xgboost_objective = (
                "binary:logistic"
            )

            evaluation_metric = "logloss"
        else:
            xgboost_objective = (
                "multi:softprob"
            )

            evaluation_metric = "mlogloss"

        xgboost_model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective=xgboost_objective,
            eval_metric=evaluation_metric,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        xgboost_model.fit(
            x_train,
            y_train_integer,
        )

        models["XGBoost"] = {
            "model": xgboost_model,
            "class_to_integer": class_to_integer,
            "integer_to_class": integer_to_class,
            "x_test": x_test,
            "y_test": y_test,
            "y_test_integer": y_test_integer,
        }

    trained_models = {}

    for model_name, model in models.items():
        if model_name == "XGBoost":
            trained_models[model_name] = model
            continue

        model.fit(
            x_train,
            y_train,
        )

        trained_models[model_name] = {
            "model": model,
            "x_test": x_test,
            "y_test": y_test,
            "x_train": x_train,
            "y_train": y_train,
        }

    trained_models["_split"] = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    return trained_models


def xgboost_predictions(
    model_information: dict[str, Any],
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    model = model_information[
        "model"
    ]

    integer_to_class = model_information[
        "integer_to_class"
    ]

    integer_predictions = model.predict(
        features
    )

    predictions = np.asarray(
        [
            integer_to_class[int(value)]
            for value in integer_predictions
        ]
    )

    probabilities = model.predict_proba(
        features
    )

    return predictions, probabilities


def get_predictions(
    model_name: str,
    model_information: dict[str, Any],
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "XGBoost":
        return xgboost_predictions(
            model_information,
            features,
        )

    model = model_information[
        "model"
    ]

    predictions = model.predict(
        features
    )

    probabilities = model.predict_proba(
        features
    )

    return predictions, probabilities


def get_classes(
    model_name: str,
    model_information: dict[str, Any],
) -> list[str]:
    if model_name == "XGBoost":
        integer_to_class = model_information[
            "integer_to_class"
        ]

        return [
            integer_to_class[position]
            for position in sorted(
                integer_to_class
            )
        ]

    model = model_information[
        "model"
    ]

    return [
        str(value)
        for value in model.classes_
    ]


def evaluate_models(
    trained_models: dict[str, Any],
) -> pd.DataFrame:
    results = []

    for model_name, model_information in trained_models.items():
        if model_name.startswith("_"):
            continue

        x_test = model_information[
            "x_test"
        ]

        y_test = model_information[
            "y_test"
        ]

        predictions, _ = get_predictions(
            model_name,
            model_information,
            x_test,
        )

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
                "Balanced accuracy": (
                    balanced_accuracy_score(
                        y_test,
                        predictions,
                    )
                ),
                "Precision": precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "F1-score": f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(
        results
    ).sort_values(
        "F1-score",
        ascending=False,
    )


def get_feature_importances(
    model_name: str,
    model_information: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    model = model_information[
        "model"
    ]

    if not hasattr(
        model,
        "feature_importances_",
    ):
        return pd.DataFrame(
            columns=[
                "Feature",
                "Importance",
            ]
        )

    importances = normalise_importances(
        model.feature_importances_
    )

    result = pd.DataFrame(
        {
            "Feature": [
                display_label(feature)
                for feature in feature_columns
            ],
            "Importance": importances,
        }
    )

    return result.sort_values(
        "Importance",
        ascending=False,
    )


# ============================================================
# Load and prepare data
# ============================================================

try:
    data = load_data()

    target_column = resolve_target_column(
        data
    )

    feature_columns = resolve_feature_columns(
        data
    )

except (
    FileNotFoundError,
    ValueError,
) as error:
    st.error(
        str(error)
    )
    st.stop()


model_data = data.copy()

model_data = model_data.dropna(
    subset=[
        target_column,
    ]
).copy()

raw_target = clean_target(
    model_data[target_column]
)

readable_target, target_label_mapping = (
    create_profile_labels(
        raw_target
    )
)

model_data[
    "urban_profile"
] = readable_target

features = clean_features(
    model_data,
    feature_columns,
)

target = model_data[
    "urban_profile"
].copy()


class_counts = target.value_counts()

valid_classes = class_counts[
    class_counts >= 2
].index

valid_mask = target.isin(
    valid_classes
)

model_data = model_data.loc[
    valid_mask
].copy()

features = features.loc[
    valid_mask
].copy()

target = target.loc[
    valid_mask
].copy()


if target.nunique() < 2:
    st.error(
        "At least two urban-profile classes are required "
        "to train the classification models."
    )
    st.stop()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.subheader(
        "Explainable AI controls"
    )

    available_models = [
        "Decision Tree",
        "Random Forest",
    ]

    if XGBOOST_AVAILABLE:
        available_models.append(
            "XGBoost"
        )

    selected_model_name = st.selectbox(
        "Prediction model",
        options=available_models,
        index=min(
            1,
            len(available_models) - 1,
        ),
    )

    test_percentage = st.slider(
        "Test sample",
        min_value=15,
        max_value=40,
        value=25,
        step=5,
    )

    tree_depth = st.slider(
        "Decision-tree maximum depth",
        min_value=2,
        max_value=15,
        value=6,
        step=1,
    )

    forest_trees = st.slider(
        "Random-forest trees",
        min_value=100,
        max_value=700,
        value=300,
        step=100,
    )

    maximum_polygons = st.slider(
        "Maximum building polygons on the map",
        min_value=1_000,
        max_value=15_000,
        value=6_000,
        step=1_000,
    )


with st.spinner(
    "Training classification models..."
):
    trained_models = train_models(
        features=features,
        target=target,
        test_size=test_percentage / 100,
        maximum_depth=tree_depth,
        forest_trees=forest_trees,
    )


selected_model_information = trained_models[
    selected_model_name
]

selected_model = selected_model_information[
    "model"
]


# ============================================================
# Predictions for the complete dataset
# ============================================================

complete_predictions, complete_probabilities = (
    get_predictions(
        selected_model_name,
        selected_model_information,
        features,
    )
)

prediction_classes = get_classes(
    selected_model_name,
    selected_model_information,
)

model_data[
    "predicted_profile"
] = complete_predictions

model_data[
    "prediction_confidence"
] = complete_probabilities.max(
    axis=1
)

model_data[
    "prediction_correct"
] = (
    model_data[
        "predicted_profile"
    ].astype(str)
    == model_data[
        "urban_profile"
    ].astype(str)
)

profile_colours = build_colour_mapping(
    sorted(
        model_data[
            "predicted_profile"
        ].astype(str).unique()
    )
)


# ============================================================
# Summary metrics
# ============================================================

evaluation_table = evaluate_models(
    trained_models
)

selected_evaluation = (
    evaluation_table[
        evaluation_table[
            "Model"
        ]
        == selected_model_name
    ]
    .iloc[0]
)

metric_columns = st.columns(
    5
)

metric_columns[0].metric(
    "Buildings",
    f"{len(model_data):,}",
)

metric_columns[1].metric(
    "Urban profiles",
    f"{target.nunique()}",
)

metric_columns[2].metric(
    "Accuracy",
    f"{selected_evaluation['Accuracy']:.3f}",
)

metric_columns[3].metric(
    "F1-score",
    f"{selected_evaluation['F1-score']:.3f}",
)

metric_columns[4].metric(
    "Mean confidence",
    (
        f"{model_data['prediction_confidence'].mean():.3f}"
    ),
)


# ============================================================
# Tabs
# ============================================================

(
    prediction_tab,
    global_tab,
    local_tab,
    comparison_tab,
    methodology_tab,
) = st.tabs(
    [
        "Building prediction",
        "Global explanations",
        "Local explanations",
        "Model comparison",
        "Methodology",
    ]
)


# ============================================================
# Building prediction tab
# ============================================================

with prediction_tab:
    st.subheader(
        f"{selected_model_name} predictions"
    )

    render_categorical_map_compat(
        dataframe=model_data,
        category_column="predicted_profile",
        colour_mapping=profile_colours,
        maximum_polygons=maximum_polygons,
    )

    st.caption(
        (
            "Each polygon represents the original building geometry. "
            "Colours indicate the profile predicted by the selected model."
        )
    )

    prediction_distribution = (
        model_data[
            "predicted_profile"
        ]
        .value_counts()
        .rename_axis(
            "Predicted profile"
        )
        .reset_index(
            name="Buildings"
        )
    )

    prediction_chart = px.bar(
        prediction_distribution,
        x="Predicted profile",
        y="Buildings",
        title="Distribution of predicted urban profiles",
    )

    prediction_chart.update_layout(
        xaxis_title="Predicted urban profile",
        yaxis_title="Number of buildings",
    )

    st.plotly_chart(
        prediction_chart,
        width="stretch",
    )

    confidence_chart = px.histogram(
        model_data,
        x="prediction_confidence",
        color="predicted_profile",
        nbins=30,
        opacity=0.75,
        title="Prediction-confidence distribution",
        labels={
            "prediction_confidence": (
                "Prediction confidence"
            ),
            "predicted_profile": (
                "Predicted profile"
            ),
        },
    )

    confidence_chart.update_layout(
        xaxis_title="Maximum predicted probability",
        yaxis_title="Number of buildings",
        legend_title="Predicted profile",
        bargap=0.04,
    )

    st.plotly_chart(
        confidence_chart,
        width="stretch",
    )


# ============================================================
# Global explanations tab
# ============================================================

with global_tab:
    st.subheader(
        f"Global feature importance — {selected_model_name}"
    )

    importance_table = get_feature_importances(
        selected_model_name,
        selected_model_information,
        feature_columns,
    )

    if importance_table.empty:
        st.info(
            "Global feature importance is not available "
            "for the selected model."
        )
    else:
        importance_chart = px.bar(
            importance_table.sort_values(
                "Importance",
                ascending=True,
            ),
            x="Importance",
            y="Feature",
            orientation="h",
            title=(
                "Relative contribution of predictor variables"
            ),
        )

        importance_chart.update_layout(
            xaxis_title="Normalised importance",
            yaxis_title="",
        )

        st.plotly_chart(
            importance_chart,
            width="stretch",
        )

        display_importance = (
            importance_table.copy()
        )

        display_importance[
            "Importance"
        ] = (
            display_importance[
                "Importance"
            ]
            .mul(100)
            .round(2)
        )

        display_importance = (
            display_importance.rename(
                columns={
                    "Importance": (
                        "Importance (%)"
                    ),
                }
            )
        )

        st.dataframe(
            display_importance,
            width="stretch",
            hide_index=True,
        )

    if selected_model_name == "Decision Tree":
        st.subheader(
            "Decision-tree rules"
        )

        tree_rules = export_text(
            selected_model,
            feature_names=[
                display_label(feature)
                for feature in feature_columns
            ],
            max_depth=tree_depth,
            decimals=3,
        )

        st.code(
            tree_rules,
            language="text",
        )


# ============================================================
# Local explanations tab
# ============================================================

with local_tab:
    st.subheader(
        "Building-level prediction inspection"
    )

    identifier_column = (
        "osm_id"
        if "osm_id" in model_data.columns
        else None
    )

    if identifier_column is None:
        model_data[
            "_building_identifier"
        ] = model_data.index.astype(
            str
        )

        identifier_column = (
            "_building_identifier"
        )

    identifier_values = (
        model_data[
            identifier_column
        ]
        .astype(str)
        .tolist()
    )

    selected_identifier = st.selectbox(
        "Select a building",
        options=identifier_values,
    )

    selected_position = identifier_values.index(
        selected_identifier
    )

    selected_row = features.iloc[
        [
            selected_position
        ]
    ]

    selected_record = model_data.iloc[
        selected_position
    ]

    selected_prediction = str(
        selected_record[
            "predicted_profile"
        ]
    )

    selected_observed = str(
        selected_record[
            "urban_profile"
        ]
    )

    selected_confidence = float(
        selected_record[
            "prediction_confidence"
        ]
    )

    selected_probabilities = (
        complete_probabilities[
            selected_position
        ]
    )

    local_metric_columns = st.columns(
        4
    )

    local_metric_columns[0].metric(
        "Building",
        selected_identifier,
    )

    local_metric_columns[1].metric(
        "Observed profile",
        selected_observed,
    )

    local_metric_columns[2].metric(
        "Predicted profile",
        selected_prediction,
    )

    local_metric_columns[3].metric(
        "Confidence",
        f"{selected_confidence:.1%}",
    )

    probability_table = pd.DataFrame(
        {
            "Urban profile": prediction_classes,
            "Probability": selected_probabilities,
        }
    ).sort_values(
        "Probability",
        ascending=False,
    )

    probability_chart = px.bar(
        probability_table.sort_values(
            "Probability",
            ascending=True,
        ),
        x="Probability",
        y="Urban profile",
        orientation="h",
        title="Predicted class probabilities",
    )

    probability_chart.update_layout(
        xaxis_title="Probability",
        yaxis_title="",
        xaxis_tickformat=".0%",
    )

    st.plotly_chart(
        probability_chart,
        width="stretch",
    )

    st.subheader(
        "Local variable contributions"
    )

    split_information = trained_models[
        "_split"
    ]

    local_effects = (
        calculate_permutation_local_effects(
            model=selected_model,
            selected_row=selected_row,
            reference_data=split_information[
                "x_train"
            ],
            predicted_class=selected_prediction,
            feature_columns=feature_columns,
        )
        if selected_model_name
        != "XGBoost"
        else pd.DataFrame()
    )

    if (
        selected_model_name == "XGBoost"
        or local_effects.empty
    ):
        st.info(
            (
                "The model-specific local permutation explanation "
                "is currently available for Decision Tree and "
                "Random Forest."
            )
        )
    else:
        contribution_chart = px.bar(
            local_effects.sort_values(
                "Local contribution",
                ascending=True,
            ),
            x="Local contribution",
            y="Feature",
            orientation="h",
            title=(
                "Change in predicted-class probability "
                "when each feature is replaced by its median"
            ),
            hover_data=[
                "Building value",
                "Reference value",
            ],
        )

        contribution_chart.add_vline(
            x=0,
            line_width=1,
        )

        contribution_chart.update_layout(
            xaxis_title=(
                "Contribution to predicted-class probability"
            ),
            yaxis_title="",
        )

        st.plotly_chart(
            contribution_chart,
            width="stretch",
        )

        display_local_effects = (
            local_effects.copy()
        )

        numeric_local_columns = [
            "Building value",
            "Reference value",
            "Local contribution",
        ]

        display_local_effects[
            numeric_local_columns
        ] = (
            display_local_effects[
                numeric_local_columns
            ].round(4)
        )

        st.dataframe(
            display_local_effects,
            width="stretch",
            hide_index=True,
        )

    if selected_model_name == "Decision Tree":
        st.subheader(
            "Decision path"
        )

        path_statements = decision_path_text(
            model=selected_model,
            selected_row=selected_row,
            feature_columns=feature_columns,
        )

        if path_statements:
            for step_number, statement in enumerate(
                path_statements,
                start=1,
            ):
                st.markdown(
                    f"**{step_number}.** {statement}"
                )
        else:
            st.info(
                "No decision path could be extracted."
            )

    st.subheader(
        "LIME explanation"
    )

    if not LIME_AVAILABLE:
        st.info(
            (
                "LIME is not installed. Add `lime` to "
                "`requirements.txt` to activate this explanation."
            )
        )
    elif selected_model_name == "XGBoost":
        st.info(
            (
                "LIME support in this page is currently configured "
                "for models with the original profile labels."
            )
        )
    else:
        try:
            lime_table = generate_lime_explanation(
                model=selected_model,
                training_features=split_information[
                    "x_train"
                ],
                selected_row=selected_row,
                feature_columns=feature_columns,
                predicted_class=selected_prediction,
            )

            lime_chart = px.bar(
                lime_table.sort_values(
                    "Contribution",
                    ascending=True,
                ),
                x="Contribution",
                y="Condition",
                orientation="h",
                title=(
                    "LIME explanation for the selected building"
                ),
            )

            lime_chart.add_vline(
                x=0,
                line_width=1,
            )

            lime_chart.update_layout(
                xaxis_title=(
                    "Contribution to the selected prediction"
                ),
                yaxis_title="",
            )

            st.plotly_chart(
                lime_chart,
                width="stretch",
            )

            st.dataframe(
                lime_table.round(
                    {
                        "Contribution": 4,
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        except Exception as lime_error:
            st.warning(
                (
                    "The LIME explanation could not be generated: "
                    f"{lime_error}"
                )
            )


# ============================================================
# Model comparison tab
# ============================================================

with comparison_tab:
    st.subheader(
        "Classification-model comparison"
    )

    comparison_display = (
        evaluation_table.copy()
    )

    metric_fields = [
        "Accuracy",
        "Balanced accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]

    comparison_display[
        metric_fields
    ] = (
        comparison_display[
            metric_fields
        ].round(3)
    )

    st.dataframe(
        comparison_display,
        width="stretch",
        hide_index=True,
    )

    comparison_long = evaluation_table.melt(
        id_vars="Model",
        value_vars=metric_fields,
        var_name="Metric",
        value_name="Score",
    )

    model_comparison_chart = px.bar(
        comparison_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Predictive performance by model",
    )

    model_comparison_chart.update_layout(
        xaxis_title="",
        yaxis_title="Score",
        yaxis_range=[
            0,
            1,
        ],
        legend_title="Metric",
    )

    st.plotly_chart(
        model_comparison_chart,
        width="stretch",
    )

    selected_x_test = (
        selected_model_information[
            "x_test"
        ]
    )

    selected_y_test = (
        selected_model_information[
            "y_test"
        ]
    )

    test_predictions, _ = get_predictions(
        selected_model_name,
        selected_model_information,
        selected_x_test,
    )

    confusion = confusion_matrix(
        selected_y_test,
        test_predictions,
        labels=prediction_classes,
    )

    confusion_figure = px.imshow(
        confusion,
        x=prediction_classes,
        y=prediction_classes,
        text_auto=True,
        aspect="auto",
        title=(
            f"Confusion matrix — {selected_model_name}"
        ),
        labels={
            "x": "Predicted profile",
            "y": "Observed profile",
            "color": "Buildings",
        },
    )

    st.plotly_chart(
        confusion_figure,
        width="stretch",
    )

    st.subheader(
        "Detailed classification report"
    )

    report = classification_report(
        selected_y_test,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )

    report_table = (
        pd.DataFrame(
            report
        )
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "Class",
            }
        )
    )

    numeric_report_columns = (
        report_table.select_dtypes(
            include="number"
        ).columns
    )

    report_table[
        numeric_report_columns
    ] = (
        report_table[
            numeric_report_columns
        ].round(3)
    )

    st.dataframe(
        report_table,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# Methodology tab
# ============================================================

with methodology_tab:
    st.subheader(
        "Analytical workflow"
    )

    st.markdown(
        """
### Prediction objective

The models reproduce the existing building-level urban-profile
classification using socio-spatial accessibility and demographic variables.
The objective is not only to maximise predictive performance, but also to
explain why each building is assigned to a given profile.

### Predictor variables

The available predictors are selected from:

- banks;
- health centres;
- pharmacies;
- supermarkets;
- parks and gardens;
- hospitals;
- post offices;
- population aged 65 or over;
- mean distance to services;
- number of nearby services;
- number of nearby service categories.

Only fields present in the research dataset are used.

### Models

**Decision Tree**

A directly interpretable classifier. Each prediction can be traced through
the sequence of thresholds from the root node to the final leaf.

**Random Forest**

An ensemble of decision trees trained on resampled observations and
predictor subsets. Global importance measures summarise the contribution of
each variable.

**XGBoost**

A gradient-boosted tree model included when the `xgboost` package is
available in the application environment.

### Validation

The dataset is divided into stratified training and test samples. Model
performance is assessed through:

- accuracy;
- balanced accuracy;
- weighted precision;
- weighted recall;
- weighted F1-score;
- confusion matrix.

### Explainability

Global explanations identify the variables that contribute most to the
model across all buildings.

Local explanations inspect one building at a time through:

- predicted class probabilities;
- local feature perturbation;
- decision paths for the Decision Tree;
- LIME explanations when the package is installed.

### Interpretation limits

Feature importance does not establish causality. A variable can be important
because it is associated with the profile structure learned from the data,
but this does not demonstrate that modifying that variable would necessarily
change the urban conditions of a building.

Local perturbation explanations replace one variable with its training-sample
median while holding the remaining variables constant. They should therefore
be interpreted as model-behaviour diagnostics rather than causal
counterfactuals.
        """
    )

    st.subheader(
        "Variables used by the models"
    )

    variable_table = pd.DataFrame(
        {
            "Dataset field": feature_columns,
            "Indicator": [
                display_label(feature)
                for feature in feature_columns
            ],
        }
    )

    st.dataframe(
        variable_table,
        width="stretch",
        hide_index=True,
    )

    st.subheader(
        "Software availability"
    )

    availability_table = pd.DataFrame(
        {
            "Component": [
                "Decision Tree",
                "Random Forest",
                "XGBoost",
                "LIME",
            ],
            "Available": [
                True,
                True,
                XGBOOST_AVAILABLE,
                LIME_AVAILABLE,
            ],
        }
    )

    st.dataframe(
        availability_table,
        width="stretch",
        hide_index=True,
    )

    download_columns = [
        column
        for column in [
            "osm_id",
            "designacao_simplificada",
            target_column,
            "urban_profile",
            "predicted_profile",
            "prediction_confidence",
            "prediction_correct",
            *feature_columns,
            "latitude",
            "longitude",
        ]
        if column in model_data.columns
    ]

    st.download_button(
        "Download Explainable AI results",
        data=(
            model_data[
                download_columns
            ]
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        ),
        file_name=(
            "geoinsightlab_explainable_ai.csv"
        ),
        mime="text/csv",
    )
