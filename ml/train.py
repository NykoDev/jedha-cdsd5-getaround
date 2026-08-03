import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

DATA_PATH = "data/get_around_pricing_project.csv"
MODEL_OUTPUT_PATH = "api/model/model.joblib"
RANDOM_STATE = 42

XGB_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [3, 5, 7, 9],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
}


def load_and_clean_data(path=DATA_PATH):
    df = pd.read_csv(path, index_col=0)
    df = df[(df["mileage"] > 0) & (df["mileage"] < 500000)]
    df = df[df["engine_power"] > 0]
    return df


def build_preprocessing(X):
    cat_columns = X.select_dtypes(include=["object"]).columns
    num_columns = X.select_dtypes(include=["int64", "float64"]).columns
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_columns),
        ("num", StandardScaler(), num_columns),
    ], remainder="passthrough")


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
    }


def tune_and_log(preprocessing, X_train, y_train, X_test, y_test, n_iter=30):
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", XGBRegressor(random_state=RANDOM_STATE)),
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=XGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=5,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    metrics = evaluate(best_pipeline, X_test, y_test)
    metrics["best_cv_r2"] = search.best_score_

    with mlflow.start_run(run_name="xgboost_tuned"):
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_pipeline, "model")

    print("xgboost_tuned", search.best_params_)
    print(metrics)
    return best_pipeline, metrics


def main():
    df = load_and_clean_data()
    y = df["rental_price_per_day"]
    X = df.drop(columns=["rental_price_per_day"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    preprocessing = build_preprocessing(X)

    mlflow.set_experiment("getaround-pricing")

    best_pipeline, best_metrics = tune_and_log(preprocessing, X_train, y_train, X_test, y_test)
    print(f"\nModele retenu : xgboost_tuned {best_metrics}")

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(best_pipeline, MODEL_OUTPUT_PATH)
    print(f"Pipeline exporte vers {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
