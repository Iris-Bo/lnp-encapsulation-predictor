import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Import the data loader from our new module
from data import load_and_split_data

def train_model():
    # 1. Set the MLflow Experiment
    mlflow.set_experiment("LNP_Encapsulation_Prediction")

    # 2. Start the tracking context
    with mlflow.start_run(run_name="Baseline_RandomForest_Model2.2"):
        
        # Load the actual processed dataset
        print("Loading data...")
        X_train, X_test, y_train, y_test = load_and_split_data(filepath="../data/processed/df_final_num_cat.csv")
        
        # Define and log parameters
        random_state = 126
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("features", "Numerical + Ionizable Lipid")
        
        # Train model
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(random_state=random_state)
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Calculate Feature Importance
        importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        print("\nTop 3 Important Features:")
        print(importance_df.head(3).to_string(index=False))
        
        # Save the model and feature names artifacts locally for the FastAPI app
        os.makedirs("../models", exist_ok=True)
        joblib.dump(rf_model, "../models/random_forest_baseline.pkl")
        joblib.dump(list(X_train.columns), "../models/feature_names.pkl")
        
        # Log the model to MLflow's artifact store
        mlflow.sklearn.log_model(rf_model, artifact_path="model")
        
        print(f"\nRun completed! MAE: {mae:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    # Ensure the script is run from the src/ directory
    # so the relative paths to ../data/ and ../models/ work correctly.
    train_model()