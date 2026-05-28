
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import shap
import matplotlib.pyplot as plt

def train_potential_model():
    print("Loading city potential data...")
    df = pd.read_csv('data/city_potential_data.csv')
    
    # All weather averages are used as features
    features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
    ]
    
    X = df[features]
    y = df['TargetPVOUT']
    
    print(f"Training XGBoost on {len(df)} cities...")
    # Use a small model to prevent overfitting on 49 rows
    model = xgb.XGBRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Evaluate fit
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    
    print(f"Model Fit - MAE: {mae:.4f}, R2: {r2:.4f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/city_potential_model.joblib')
    
    # Save feature names for the predictor
    joblib.dump(features, 'models/feature_names.joblib')
    
    print("City Potential Model saved.")

    # SHAP Analysis
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('models/potential_shap.png')
    print("Potential SHAP plot saved.")

if __name__ == "__main__":
    train_potential_model()
