
import joblib
import pandas as pd
import numpy as np

class PotentialPredictor:
    def __init__(self, model_path='models/city_potential_model.joblib', features_path='models/feature_names.joblib'):
        self.model = joblib.load(model_path)
        self.features = joblib.load(features_path)
        # Training set means for fallback
        self.feature_means = {
            "MinTemp": 12.16,
            "MaxTemp": 23.42,
            "Rainfall": 2.31,
            "Evaporation": 5.50,
            "Sunshine": 7.66,
            "WindGustSpeed": 40.02,
            "WindSpeed9am": 13.95,
            "WindSpeed3pm": 18.54,
            "Humidity9am": 68.62,
            "Humidity3pm": 50.81,
            "Pressure9am": 1017.63,
            "Pressure3pm": 1015.19,
            "Cloud9am": 4.57,
            "Cloud3pm": 4.60,
            "Temp9am": 16.99,
            "Temp3pm": 21.91
        }

    def predict(self, weather_profile):
        """
        weather_profile: dict with mean weather values.
        """
        df = pd.DataFrame([weather_profile])
        
        # Fill missing features with average from training data
        for col in self.features:
            if col not in df.columns or pd.isna(df[col].iloc[0]):
                df[col] = self.feature_means.get(col, 0.0)
        
        df = df[self.features]
        prediction = self.model.predict(df)[0]
        return float(prediction)

if __name__ == "__main__":
    predictor = PotentialPredictor()
    # Test with Sydney-like average profile
    test_profile = {
        'Sunshine': 7.6,
        'MaxTemp': 23.0,
        'Cloud3pm': 4.0,
        'Rainfall': 2.8
    }
    result = predictor.predict(test_profile)
    print(f"Predicted Annual Avg Solar Potential: {result:.4f} kWh/kWp/day")
