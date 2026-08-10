import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from cargar_datos import cargarDatos
from ft_engineering import (
    TARGET,
    COLUMNAS_NUMERIC,
    COLUMNAS_CATEGORIC,
    COLUMNAS_ORDINAL,
    limpiar_categoricas_con_ruido_numerico,
    build_preprocessor,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_pipeline.pkl")
RANDOM_STATE = 1234
TEST_FRAC = 0.2

def get_candidate_models():
    """
    Devuelve un diccionario con los modelos candidatos a comparar.
    """
    return {
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }

def train_and_select_best_model():
    # 1. Cargar datos y preprocesar listas de columnas
    df = cargarDatos()
    df = limpiar_categoricas_con_ruido_numerico(
        df, "tendencia_ingresos", ["Creciente", "Decreciente", "Estable"]
    )

    feature_cols = COLUMNAS_NUMERIC + COLUMNAS_CATEGORIC + COLUMNAS_ORDINAL
    X = df[feature_cols]
    y = df[TARGET]

    # 2. Dividir train/test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Construir preprocesador
    preprocessor = build_preprocessor(COLUMNAS_NUMERIC, COLUMNAS_CATEGORIC, COLUMNAS_ORDINAL)

    # 4. Iterar sobre los modelos candidatos y evaluarlos
    candidates = get_candidate_models()
    best_score = -1.0
    best_pipeline = None
    best_model_name = ""

    from sklearn.metrics import f1_score

    print("--- Evaluando Modelos Candidatos ---")
    for name, model in candidates.items():
        # Crear pipeline individual
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        # Entrenar
        pipeline.fit(x_train, y_train)
        
        # Predecir en Test para seleccionar el mejor
        y_pred = pipeline.predict(x_test)
        score = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        
        print(f"Modelo: {name:<20} | F1-Score (no pago): {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_model_name = name

    print("-" * 42)
    print(f"--> Modelo seleccionado: {best_model_name} con F1-Score: {best_score:.4f}")

    # 5. Guardar el mejor pipeline
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Guardado el mejor pipeline en {MODEL_PATH}")

if __name__ == "__main__":
    train_and_select_best_model()