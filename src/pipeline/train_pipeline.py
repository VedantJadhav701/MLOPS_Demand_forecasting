import yaml
import logging
import pickle
import numpy as np
import os
from pathlib import Path

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

# Trigger run: 1
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc

from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.ingestion import load_data
from src.features.feature_engineering import build_features
from src.monitoring.monitor import run_drift_check


logger = logging.getLogger(__name__)
MODEL_NAME = "demand_forecasting_model"


# ----------------------
# CHAMPION VS CHALLENGER
# ----------------------
def is_better_model(new_metrics, old_metrics):
    """
    Decide if challenger is better than champion
    """
    # primary metric: MAPE
    if new_metrics["mape"] < old_metrics["mape"]:
        return True

    # secondary fallback
    if new_metrics["rmse"] < old_metrics["rmse"]:
        return True

    return False


# ----------------------
# TIME SPLIT
# ----------------------
def time_split(df, date_col="date"):
    df = df.sort_values(date_col).reset_index(drop=True)

    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size + val_size]
    test = df.iloc[train_size + val_size:]

    logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


# ----------------------
# STACKED MODEL WRAPPER (for MLflow pyfunc)
# ----------------------
class StackedModel(mlflow.pyfunc.PythonModel):
    def __init__(self, lgb_model, cb_model, lgb_weight, cb_weight):
        self.lgb_model = lgb_model
        self.cb_model = cb_model
        self.lgb_weight = lgb_weight
        self.cb_weight = cb_weight

    def predict(self, context, model_input):
        lgb_preds = self.lgb_model.predict(model_input)
        cb_preds = self.cb_model.predict(model_input)
        return self.lgb_weight * lgb_preds + self.cb_weight * cb_preds


# ----------------------
# MAIN PIPELINE
# ----------------------
def run_pipeline(is_retraining=False):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s"
    )

    mlflow.set_experiment("demand_forecasting")

    with mlflow.start_run(nested=is_retraining):

        # ----------------------
        # LOAD CONFIG
        # ----------------------
        config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Config loaded from {config_path}")

        # ----------------------
        # LOAD DATA
        # ----------------------
        data_path = os.getenv("DATA_PATH", config["data"]["path"])
        df = load_data(
            data_path,
            config["data"]["date_column"]
        )
        logger.info(f"Data loaded from {data_path}: {len(df)} records")

        # ----------------------
        # FEATURE ENGINEERING
        # ----------------------
        df = build_features(df, config)
        logger.info(f"Features engineered: {len(df.columns)} columns")

        date_col = config["data"]["date_column"]

        # ----------------------
        # SPLIT
        # ----------------------
        train, val, test = time_split(df, date_col=date_col)

        # ----------------------
        # DRIFT MONITORING
        # ----------------------
        drift_detected = run_drift_check(train, test)

        if drift_detected:
            print("[WARNING] Drift detected -> triggering retraining")
            if not is_retraining:
                run_pipeline(is_retraining=True)  # simple loop (for now)
                return
            else:
                print("[INFO] Executing retraining pipeline...")

        target = config["data"]["target"]
        features = [col for col in df.columns if col not in [target, date_col, "record_ID"]]

        X_train, y_train = train[features], train[target]
        X_val, y_val = val[features], val[target]
        X_test, y_test = test[features], test[target]

        # handle missing safely
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)

        logger.info(f"Features: {len(features)}")

        # categorical features
        categorical_features = [
            "product_id", "store_id", "category_id",
            "day_of_week", "month", "quarter", "week_of_year"
        ]
        categorical_features = [f for f in categorical_features if f in features]
        cat_indices = [features.index(f) for f in categorical_features]

        logger.info(f"Categorical features: {categorical_features}")

        # ==============================
        # TRAIN LIGHTGBM
        # ==============================
        lgb_params = config["model"]["lgb_params"]
        mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})

        train_ds = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features
        )

        val_ds = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=categorical_features,
            reference=train_ds
        )

        lgb_callbacks = [
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(50),
        ]

        lgb_model = lgb.train(
            lgb_params,
            train_ds,
            num_boost_round=3000,
            valid_sets=[val_ds],
            callbacks=lgb_callbacks,
        )

        logger.info("[LightGBM] Training complete")

        lgb_val_preds = lgb_model.predict(X_val)
        lgb_test_preds = lgb_model.predict(X_test)

        lgb_val_mape = (abs((y_val - lgb_val_preds) / (y_val + 1e-8))).mean() * 100
        lgb_test_mape = (abs((y_test - lgb_test_preds) / (y_test + 1e-8))).mean() * 100

        print(f"[LightGBM] Val MAPE: {lgb_val_mape:.2f}% | Test MAPE: {lgb_test_mape:.2f}%")

        # ==============================
        # TRAIN CATBOOST
        # ==============================
        cb_params = config["model"]["cb_params"]
        mlflow.log_params({f"cb_{k}": v for k, v in cb_params.items()})

        # CatBoost needs categorical indices
        train_pool = Pool(
            X_train, label=y_train,
            cat_features=cat_indices
        )
        val_pool = Pool(
            X_val, label=y_val,
            cat_features=cat_indices
        )

        cb_model = CatBoostRegressor(**cb_params)
        cb_model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=100,
            verbose=50,
        )

        logger.info("[CatBoost] Training complete")

        cb_val_preds = cb_model.predict(X_val)
        cb_test_preds = cb_model.predict(X_test)

        cb_val_mape = (abs((y_val - cb_val_preds) / (y_val + 1e-8))).mean() * 100
        cb_test_mape = (abs((y_test - cb_test_preds) / (y_test + 1e-8))).mean() * 100

        print(f"[CatBoost] Val MAPE: {cb_val_mape:.2f}% | Test MAPE: {cb_test_mape:.2f}%")

        # ==============================
        # STACKED PREDICTIONS
        # ==============================
        stacking_cfg = config["model"]["stacking"]
        w_lgb = stacking_cfg["lgb_weight"]
        w_cb = stacking_cfg["cb_weight"]

        mlflow.log_params({"lgb_weight": w_lgb, "cb_weight": w_cb})

        val_preds = w_lgb * lgb_val_preds + w_cb * cb_val_preds
        test_preds = w_lgb * lgb_test_preds + w_cb * cb_test_preds

        # ----------------------
        # EVALUATION (STACKED)
        # ----------------------
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        val_mae = mean_absolute_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)
        val_mape = (abs((y_val - val_preds) / (y_val + 1e-8))).mean() * 100

        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        test_mape = (abs((y_test - test_preds) / (y_test + 1e-8))).mean() * 100

        mlflow.log_metrics({
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "val_mape": val_mape,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "test_mape": test_mape,
        })

        print("=" * 50)
        print("[STACKED ENSEMBLE] LightGBM + CatBoost")
        print("=" * 50)
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Validation MAE:  {val_mae:.4f}")
        print(f"Validation R2:   {val_r2:.4f}")
        print(f"Validation MAPE: {val_mape:.2f}%")
        print("-" * 50)
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE:  {test_mae:.4f}")
        print(f"Test R2:   {test_r2:.4f}")
        print(f"Test MAPE: {test_mape:.2f}%")
        print("=" * 50)

        # ----------------------
        # LOG STACKED MODEL
        # ----------------------
        stacked = StackedModel(lgb_model, cb_model, w_lgb, w_cb)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=stacked,
        )

        # ----------------------
        # REGISTER MODEL
        # ----------------------
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        client = MlflowClient()

        if MODEL_NAME not in [m.name for m in client.search_registered_models()]:
            client.create_registered_model(MODEL_NAME)

        model_version = client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run_id
        )

        print(f"Model registered: {MODEL_NAME} v{model_version.version}")

        # ----------------------
        # LOG PREDICTIONS
        # ----------------------
        sample_input = X_test.iloc[0].to_dict()
        sample_pred = float(test_preds[0])
        mlflow.log_dict(sample_input, "input.json")
        mlflow.log_dict({"prediction": sample_pred}, "output.json")

        # Log Git Commit if available
        git_commit = os.getenv("GITHUB_SHA")
        if git_commit:
            mlflow.set_tag("git_commit", git_commit)
            logger.info(f"MLflow tag set: git_commit={git_commit}")

        # ----------------------
        # QUALITY GATES & MODEL COMPARISON
        # ----------------------
        rmse_th = config["model"]["rmse_threshold"]
        mape_th = config["model"]["mape_threshold"]
        r2_th = config["model"]["r2_threshold"]

        if val_rmse < rmse_th and val_mape < mape_th and val_r2 > r2_th:
            # get current production model
            prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            
            current_prod_metrics = None
            
            if prod_versions:
                prod_run_id = prod_versions[0].run_id
                prod_run = mlflow.get_run(prod_run_id)
            
                current_prod_metrics = {
                    "mape": prod_run.data.metrics.get("val_mape", 999),
                    "rmse": prod_run.data.metrics.get("val_rmse", 999),
                    "r2": prod_run.data.metrics.get("val_r2", 0),
                }
            
                print(f"[Champion] Current Production Model -> MAPE: {current_prod_metrics['mape']:.2f}")

            new_metrics = {
                "mape": val_mape,
                "rmse": val_rmse,
                "r2": val_r2,
            }

            promote = False
            
            if current_prod_metrics is None:
                print("No existing production model -> promoting")
                promote = True
            else:
                if is_better_model(new_metrics, current_prod_metrics):
                    print("[SUCCESS] Challenger is better -> promoting")
                    promote = True
                else:
                    print("[REJECT] Challenger worse than champion -> reject")
            
            if promote:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=model_version.version,
                    stage="Production"
                )
                print("[PROMOTED] Model promoted to PRODUCTION")

                # Write deployment flag for CI/CD
                with open("deploy_flag.txt", "w") as f:
                    f.write("promote_to_production")
                print("[CD GATE] deploy_flag.txt created")
            else:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=model_version.version,
                    stage="Staging"
                )
                print("[STAGING] Model kept in STAGING")
        else:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=model_version.version,
                stage="Staging"
            )
            print("[FAILED] Model moved to STAGING (Failed Quality Gates)")


if __name__ == "__main__":
    run_pipeline()