
import rasterio
import pandas as pd
import os
from processing.coords import LOCATION_COORDS

def get_raster_value(raster_path, lat, lon):
    with rasterio.open(raster_path) as dataset:
        try:
            row, col = dataset.index(lon, lat)
            if row < 0 or row >= dataset.height or col < 0 or col >= dataset.width:
                return None
            value = dataset.read(1)[row, col]
            # Handle NoData values (often very small negative numbers or specific values)
            if value < 0:
                return None
            return value
        except Exception as e:
            return None

def extract_all_monthly_values(base_dir):
    results = []
    for loc, (lat, lon) in LOCATION_COORDS.items():
        loc_values = {'Location': loc}
        for month in range(1, 13):
            month_str = f"{month:02d}"
            tiff_path = os.path.join(base_dir, f"PVOUT_{month_str}.tif")
            if os.path.exists(tiff_path):
                val = get_raster_value(tiff_path, lat, lon)
                loc_values[f"PVOUT_{month_str}"] = val
            else:
                print(f"Warning: {tiff_path} not found")
        results.append(loc_values)
    return pd.DataFrame(results)

if __name__ == "__main__":
    base_dir = "data/Australia_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/monthly"
    df_solar = extract_all_monthly_values(base_dir)
    df_solar.to_csv("data/location_monthly_solar.csv", index=False)
    print("Extracted monthly solar values saved to data/location_monthly_solar.csv")
