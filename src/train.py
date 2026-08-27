import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pathlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from mlflow.models import infer_signature

# Import the data loader from our new module
from data import load_data

# Point to shared MLflow database in project folder root
project_root = pathlib.Path(__file__).resolve().parent.parent
db_path = project_root / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("LNP_Encapsulation_Prediction")

def train_model():

    #Load full dataset (no splitting)
    X, y = load_data()
    print("Loading Data...")

    # Define Model and parameters
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None, 
        min_samples_split=10, 
        random_state=126
    )

    # Start the tracking context
    with mlflow.start_run(run_name="RandomForest_Tuned"):

        # --- STEP 1: EVALUATION (For MLFlow Tracking) ---
        kf = KFold(n_splits=5, shuffle=True, random_state=126)
        mae_scores = -cross_val_score(rf_model, X, y, cv=kf, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='r2')

        # Log metrics to MLFlow
        mlflow.log_metric("cv_mean_mae", np.mean(mae_scores))
        mlflow.log_metric("cv_mean_r2", np.mean(r2_scores))

        # Log parameters to MLFlow
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_params(rf_model.get_params())
        mlflow.log_param("cv_folds", 5)

        #Log dataset attribute           
        dataset_df = X.copy()
        dataset_df["encapsulation_efficiency"] = y
        dataset = mlflow.data.from_pandas(dataset_df, targets="encapsulation_efficiency", name="advanced_all_features")
        mlflow.log_input(dataset, context="training")
    
        # --- STEP 2: DEPLOYMENT (For FastAPI) --- 
        # Train model on full dataset
        rf_model.fit(X, y)
        print("Training Random Forest model...")

        #Log model attribute
        y_pred = rf_model.predict(X)
        signature = infer_signature(X, y_pred)
        mlflow.sklearn.log_model(sk_model=rf_model,name="RF_Tuned", signature=signature, registered_model_name="RF_Tuned")
                
        # Save the model and feature names artifacts locally for the FastAPI app
        os.makedirs("../models", exist_ok=True)
        joblib.dump(rf_model, "../models/random_forest_tuned.pkl")
        joblib.dump(list(X.columns), "../models/feature_names_tuned.pkl")
        
        print(f"\nRun completed! MAE: {np.mean(mae_scores):.4f}, R2: {np.mean(r2_scores):.4f}")

if __name__ == "__main__":
    # Ensure the script is run from the src/ directory
    # so the relative paths to ../data/ and ../models/ work correctly.
    train_model()