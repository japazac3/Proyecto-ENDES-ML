import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import seaborn as sns
import streamlit as st
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAS = ["EDAD", "RIQUEZA", "peso", "talla", "porciones_frutas"]
CATEGORICAS = ["QSSEXO", "AUTOIDENTIFICACION", "SHREGION", "diabetes", "exceso_peso", "obesidad"]
TARGET_REGRESION = "pasistolica"
TARGET_HTA = "HTAcomb"
TARGET_DIABETES = "diabetes"


def build_preprocessor(numericas, categoricas):
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numericas),
            ("cat", encoder, categoricas),
        ]
    )


@st.cache_data(show_spinner=False)
def load_data_from_path(path):
    df, _meta = pyreadstat.read_sav(path)
    return df


def load_data_from_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    return load_data_from_path(tmp_path)


def validate_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    return missing


def train_regression(df, numericas, categoricas, target):
    X = df.loc[:, numericas + categoricas]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(numericas, categoricas)

    models = {
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(random_state=42, n_jobs=1),
        "LinearRegression": LinearRegression(),
    }

    results = []
    best_name = None
    best_rmse = None
    best_preds = None
    best_pipeline = None
    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocesador", preprocessor),
                ("modelo", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append({"modelo": name, "mae": mae, "rmse": rmse, "r2": r2})

        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_preds = preds
            best_pipeline = pipeline

    results_df = (
        pd.DataFrame(results)
        .sort_values("rmse", ascending=True)
        .reset_index(drop=True)
    )
    return results_df, best_name, y_test, best_preds, best_pipeline


def train_classification(df, numericas, categoricas, target, use_smote):
    X = df.loc[:, numericas + categoricas]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(numericas, categoricas)

    models = {
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42, n_jobs=1),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    }

    results = []
    best_name = None
    best_f1 = None
    best_preds = None
    best_scores = None
    best_pipeline = None
    for name, model in models.items():
        steps = [("preprocesador", preprocessor)]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=42)))
        steps.append(("modelo", model))

        pipeline = ImbPipeline(steps=steps)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        if hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(X_test)[:, 1]
        else:
            scores = pipeline.decision_function(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        results.append(
            {
                "modelo": name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
            }
        )

        if best_f1 is None or f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_preds = preds
            best_scores = scores
            best_pipeline = pipeline

    results_df = (
        pd.DataFrame(results)
        .sort_values("f1_score", ascending=False)
        .reset_index(drop=True)
    )
    return results_df, best_name, y_test, best_preds, best_scores, best_pipeline


def feature_importance_htacomb(df, numericas, categoricas, target):
    X = df.loc[:, numericas + categoricas]
    y = df[target].astype(int)

    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    importances = pd.DataFrame(
        {"feature": model.feature_names_in_, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importances


def highlight_best(df, metric, ascending):
    best_value = df[metric].min() if ascending else df[metric].max()

    def _highlight(row):
        return [
            "background-color: #e8f5e9; font-weight: 600" if row[metric] == best_value else ""
            for _ in row
        ]

    return df.style.apply(_highlight, axis=1)


def plot_regression_results(y_true, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red")
    axes[0].set_title("Real vs Predicho")
    axes[0].set_xlabel("Real")
    axes[0].set_ylabel("Predicho")

    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Residuales")
    axes[1].set_xlabel("Predicho")
    axes[1].set_ylabel("Residual")

    plt.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_roc(y_true, scores, title):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


st.set_page_config(page_title="MachineEndes", layout="wide")
st.title("MachineEndes - Analisis y Modelos")
st.caption(
    "Proyecto de regresion y clasificacion con enfoque reproducible, comparacion de modelos y visualizacion de resultados."
)

with st.sidebar:
    st.header("Fuente de datos")
    use_local = st.checkbox("Usar base.sav local", value=True)
    uploaded = st.file_uploader("O cargar archivo .sav", type=["sav"])
    st.divider()
    st.header("Modelos")
    save_models = st.checkbox("Guardar modelos con joblib", value=False)
    models_dir = st.text_input("Carpeta de salida", value="models")

df = None
if use_local:
    try:
        df = load_data_from_path("base.sav")
    except Exception as exc:
        st.error(f"No se pudo leer base.sav: {exc}")
        df = None
elif uploaded is not None:
    try:
        df = load_data_from_upload(uploaded)
    except Exception as exc:
        st.error(f"No se pudo leer el archivo subido: {exc}")
        df = None

if df is None:
    st.stop()

if save_models:
    os.makedirs(models_dir, exist_ok=True)

st.subheader("Resumen del dataset")
st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
st.dataframe(df.head(10), use_container_width=True)

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if not missing.empty:
    st.write("Valores faltantes por columna")
    st.dataframe(missing.to_frame("faltantes"), use_container_width=True)

with st.expander("Metodologia profesional", expanded=False):
    st.write(
        "1. Seleccion de variables numericas y categoricas segun el notebook original.\n"
        "2. Particion train/test con random_state=42 para reproducibilidad.\n"
        "3. Preprocesamiento: escalado de numericas y OneHot para categoricas.\n"
        "4. Comparacion de modelos con metricas estandar.\n"
        "5. Balanceo con SMOTE en HTAcomb (solo en entrenamiento).\n"
        "6. Visualizacion de desempeño con graficos y matrices de confusion."
    )

tabs = st.tabs(
    [
        "Correlacion",
        "Regresion (pasistolica)",
        "Clasificacion HTAcomb",
        "Clasificacion diabetes",
    ]
)

with tabs[0]:
    st.subheader("Matriz de correlacion")
    cols = [c for c in (CATEGORICAS + NUMERICAS) if c in df.columns]
    if len(cols) < 2:
        st.warning("No hay suficientes columnas para correlacion.")
    else:
        corr = df.loc[:, cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, annot=False)
        st.pyplot(fig)

with tabs[1]:
    st.subheader("Modelos de regresion")
    required = NUMERICAS + CATEGORICAS + [TARGET_REGRESION]
    missing_cols = validate_columns(df, required)
    if missing_cols:
        st.error(f"Faltan columnas: {missing_cols}")
    else:
        if st.button("Entrenar modelos de regresion"):
            results, best_name, y_test, best_preds, best_pipeline = train_regression(
                df, NUMERICAS, CATEGORICAS, TARGET_REGRESION
            )

            st.write("Tabla comparativa (mejor modelo resaltado)")
            st.dataframe(highlight_best(results, "rmse", ascending=True), use_container_width=True)

            best_row = results.iloc[0]
            st.metric("Mejor modelo", best_name, help="Seleccionado por menor RMSE")
            st.write(
                f"MAE: {best_row['mae']:.4f} | RMSE: {best_row['rmse']:.4f} | R2: {best_row['r2']:.4f}"
            )

            st.write("Graficos de regresion")
            fig = plot_regression_results(y_test, best_preds)
            st.pyplot(fig)

            if save_models and best_pipeline is not None:
                model_path = os.path.join(models_dir, "modelo_regresion.joblib")
                joblib.dump(best_pipeline, model_path)
                st.success(f"Modelo guardado: {model_path}")

with tabs[2]:
    st.subheader("Modelos de clasificacion HTAcomb")
    required = NUMERICAS + CATEGORICAS + [TARGET_HTA]
    missing_cols = validate_columns(df, required)
    if missing_cols:
        st.error(f"Faltan columnas: {missing_cols}")
    else:
        use_smote = st.checkbox("Usar SMOTE (balanceo)", value=True)
        if st.button("Entrenar modelos HTAcomb"):
            results, best_name, y_test, best_preds, best_scores, best_pipeline = train_classification(
                df, NUMERICAS, CATEGORICAS, TARGET_HTA, use_smote
            )

            st.write("Tabla comparativa (mejor modelo resaltado)")
            st.dataframe(highlight_best(results, "f1_score", ascending=False), use_container_width=True)

            best_row = results.iloc[0]
            st.metric("Mejor modelo", best_name, help="Seleccionado por mayor F1")
            st.write(
                f"Accuracy: {best_row['accuracy']:.4f} | Precision: {best_row['precision']:.4f} | "
                f"Recall: {best_row['recall']:.4f} | F1: {best_row['f1_score']:.4f}"
            )

            st.write("Matriz de confusion y curva ROC")
            col1, col2 = st.columns(2)
            with col1:
                fig_cm = plot_confusion(y_test, best_preds, "Matriz de confusion - HTAcomb")
                st.pyplot(fig_cm)
            with col2:
                fig_roc = plot_roc(y_test, best_scores, "ROC - HTAcomb")
                st.pyplot(fig_roc)

            importances = feature_importance_htacomb(df, NUMERICAS, CATEGORICAS, TARGET_HTA)
            st.write("Importancia de variables (GradientBoostingClassifier)")
            st.dataframe(importances.head(15), use_container_width=True)

            if save_models and best_pipeline is not None:
                model_path = os.path.join(models_dir, "modelo_clasificacion_hta.joblib")
                joblib.dump(best_pipeline, model_path)
                st.success(f"Modelo guardado: {model_path}")

with tabs[3]:
    st.subheader("Modelos de clasificacion diabetes")
    categoricas_diabetes = [c for c in CATEGORICAS if c != TARGET_DIABETES]
    if "diagnosticoHTA" in df.columns and "diagnosticoHTA" not in categoricas_diabetes:
        categoricas_diabetes.append("diagnosticoHTA")

    required = NUMERICAS + categoricas_diabetes + [TARGET_DIABETES]
    missing_cols = validate_columns(df, required)
    if missing_cols:
        st.error(f"Faltan columnas: {missing_cols}")
    else:
        if st.button("Entrenar modelos diabetes"):
            results, best_name, y_test, best_preds, best_scores, best_pipeline = train_classification(
                df, NUMERICAS, categoricas_diabetes, TARGET_DIABETES, use_smote=False
            )

            st.write("Tabla comparativa (mejor modelo resaltado)")
            st.dataframe(highlight_best(results, "f1_score", ascending=False), use_container_width=True)

            best_row = results.iloc[0]
            st.metric("Mejor modelo", best_name, help="Seleccionado por mayor F1")
            st.write(
                f"Accuracy: {best_row['accuracy']:.4f} | Precision: {best_row['precision']:.4f} | "
                f"Recall: {best_row['recall']:.4f} | F1: {best_row['f1_score']:.4f}"
            )

            st.write("Matriz de confusion y curva ROC")
            col1, col2 = st.columns(2)
            with col1:
                fig_cm = plot_confusion(y_test, best_preds, "Matriz de confusion - Diabetes")
                st.pyplot(fig_cm)
            with col2:
                fig_roc = plot_roc(y_test, best_scores, "ROC - Diabetes")
                st.pyplot(fig_roc)

            if save_models and best_pipeline is not None:
                model_path = os.path.join(models_dir, "modelo_clasificacion_diabetes.joblib")
                joblib.dump(best_pipeline, model_path)
                st.success(f"Modelo guardado: {model_path}")
