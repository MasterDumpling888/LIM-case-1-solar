
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def test_performance():
    print("Loading test data and model...")
    df_test = pd.read_csv('data/Weather Test Data.csv')
    df_solar = pd.read_csv('data/location_monthly_solar.csv')
    model = joblib.load('models/city_potential_model.joblib')
    features = joblib.load('models/feature_names.joblib')
    
    # Clean location names
    df_test['Location'] = df_test['Location'].str.strip('"')
    df_solar['Location'] = df_solar['Location'].str.strip('"')
    
    # Aggregate test weather profile per city
    print("Aggregating test weather data...")
    df_test_avg = df_test.groupby('Location')[features].mean().reset_index()
    
    # Prepare GSA Targets
    df_solar['TargetPVOUT'] = df_solar[[f'PVOUT_{m:02d}' for m in range(1, 13)]].mean(axis=1)
    
    # Merge
    df_merged = df_test_avg.merge(df_solar[['Location', 'TargetPVOUT']], on='Location', how='left')
    df_merged = df_merged.dropna(subset=['TargetPVOUT'])
    
    X_test = df_merged[features]
    y_true = df_merged['TargetPVOUT']
    
    # Predict
    print(f"Predicting solar potential for {len(df_merged)} cities in the test set...")
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print("\n--- TEST PERFORMANCE RESULTS ---")
    print(f"Number of Cities Tested: {len(df_merged)}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} kWh/kWp/day")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} kWh/kWp/day")
    print(f"R-squared (R2): {r2:.4f}")
    
    # Show comparison for top 5 cities
    df_results = df_merged[['Location', 'TargetPVOUT']].copy()
    df_results['Predicted'] = y_pred
    df_results['Error'] = df_results['Predicted'] - df_results['TargetPVOUT']
    print("\nSample Predictions:")
    print(df_results.head(10).to_string(index=False))

if __name__ == "__main__":
    test_performance()
