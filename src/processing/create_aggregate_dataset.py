
import pandas as pd
import numpy as np

def create_aggregate_dataset():
    print("Loading data...")
    df = pd.read_csv('data/Weather Training Data.csv')
    df_solar = pd.read_csv('data/location_monthly_solar.csv')
    
    # Clean location names
    df['Location'] = df['Location'].str.strip('"')
    df_solar['Location'] = df_solar['Location'].str.strip('"')
    
    # Define numeric features to aggregate
    numeric_features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
    ]
    
    # Calculate average weather profile per city (Using ALL data)
    print("Calculating average weather profiles per city...")
    df_city_avg = df.groupby('Location')[numeric_features].mean().reset_index()
    
    # Calculate Annual Average PVOUT from Global Solar Atlas (Our direct target)
    print("Calculating Annual Average PVOUT from Global Solar Atlas...")
    df_solar['TargetPVOUT'] = df_solar[[f'PVOUT_{m:02d}' for m in range(1, 13)]].mean(axis=1)
    
    # Merge weather and solar potential
    df_merged = df_city_avg.merge(df_solar[['Location', 'TargetPVOUT']], on='Location', how='left')
    
    # Drop cities with missing solar data
    df_merged = df_merged.dropna(subset=['TargetPVOUT'])
    
    df_merged.to_csv('data/city_potential_data.csv', index=False)
    print(f"City potential data saved for {len(df_merged)} cities.")

if __name__ == "__main__":
    create_aggregate_dataset()
