import pandas as pd
import numpy as np

def clean_and_prepare_data(df):
    """
    Cleans the infrastructure data and performs feature engineering.
    """
    # 1. Data Cleaning - Remove placeholders (999999) and extreme outliers
    initial_count = len(df)
    df = df[df['TRAFFIC_VOLUME'] < 500000]
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"   Note: Removed {removed_count} rows with outlier TRAFFIC_VOLUME (likely placeholders).")

    # Mapping for categorical features to make them more descriptive
    nh_map = {
        0: 'Downtown', 1: 'Nutana', 2: 'Varsity View', 3: 'Caswell Hill', 4: 'City Park',
        5: 'Riversdale', 6: 'Pleasant Hill', 7: 'Westmount', 8: 'Kelsey-Woodlawn',
        9: 'Mayfair', 10: 'North Park', 11: 'Richmond Heights', 12: 'River Heights',
        13: 'Silverwood Heights', 14: 'Lawson Heights', 15: 'Hudson Bay Park',
        16: 'Mount Royal', 17: 'Meadowgreen', 18: 'Holiday Park', 19: 'Montgomery Place',
        20: 'Fairhaven', 21: 'Confederation Park', 22: 'Pacific Heights', 25: 'Massey Place',
        26: 'Dundonald', 27: 'Parkridge', 28: 'King George', 29: 'Exhibition',
        30: 'Queen Elizabeth', 32: 'Avalon', 33: 'Adelaide/Churchill', 35: 'Nutana Park',
        36: 'Eastview', 37: 'Brevoort Park', 38: 'Grosvenor Park', 39: 'Greystone Heights',
        40: 'College Park', 41: 'College Park East', 42: 'Wildwood', 43: 'Lakeview',
        44: 'Lakeridge', 45: 'Lakewood', 46: 'Briarwood', 47: 'Sutherland',
        48: 'Forest Grove', 49: 'Arbor Creek', 50: 'Erindale', 51: 'Silverspring',
        52: 'Willowgrove', 53: 'Stonebridge', 54: 'Hampton Village', 55: 'Rosewood',
        56: 'Evergreen', 57: 'Kensington', 58: 'Aspen Ridge', 59: 'Brighton',
        60: 'The Willows', 61: 'Blairmore', 62: 'Elk Point', 63: 'Brighton',
        64: 'Holmwood', 67: 'University', 68: 'Management Area', 100: 'North Industrial',
        101: 'Airport', 102: 'West Industrial', 103: 'CN Yards', 105: 'South West Industrial',
        106: 'AgPro Industrial', 107: 'C.N. Industrial', 108: 'South West Industrial',
        109: 'Marquis Industrial', 111: 'University Heights', 112: 'Blairmore Urban Centre',
        113: 'Stonebridge Urban Centre', 711: 'CN Yards MA', 712: 'SaskPower MA',
        713: 'Gordie Howe MA', 714: 'U of S Lands North MA', 715: 'U of S MA',
        716: 'U of S Lands South MA', 717: 'Airport MA', 903: 'Blairmore SDA',
        904: 'Holmwood SDA', 905: 'North SDA', 906: 'North West SDA'
    }
    
    # Map Neighbourhood_Name to actual names
    df['Neighbourhood_Name'] = df['Neighbourhood_Name'].map(nh_map).fillna(df['Neighbourhood_Name'].astype(str))
    
    # Map Maintenance_Group
    maint_map = {0: 'Priority 1', 1: 'Priority 2', 2: 'Priority 3', 3: 'Unclassified'}
    df['Maintenance_Group'] = df['Maintenance_Group'].map(maint_map).fillna(df['Maintenance_Group'].astype(str))
    
    # Map Snow_Route
    snow_map = {0: 'No', 1: 'Yes', 2: 'Emergency'}
    df['Snow_Route'] = df['Snow_Route'].map(snow_map).fillna(df['Snow_Route'].astype(str))

    # Map Road_Surface_Type
    surface_map = {1: 'Asphalt', 5: 'Gravel', 6: 'Concrete'}
    df['Road_Surface_Type'] = df['Road_Surface_Type'].map(surface_map).fillna(df['Road_Surface_Type'].astype(str))

    # 2. Feature Engineering - One-Hot Encoding for small categories
    # This prevents the model from assuming a linear relationship where none exists
    categorical_cols = ['Snow_Route', 'Snow_Removal_Designate', 'Maintenance_Group', 
                        'Road_Surface_Type', 'Road_Structure_Type', 'route_id', 'Neighbourhood_Name']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 3. Ensure all feature columns are numeric (float) for NumPy processing
    # One-Hot Encoding can create boolean columns, which we convert to floats
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(float)
        elif df[col].dtype == 'int64' or df[col].dtype == 'int32':
            df[col] = df[col].astype(float)
    
    # 4. De-fragment the DataFrame after type conversion to avoid performance warnings
    df = df.copy()

    # 5. Classification target for Logistic Regression
    df['HIGH_TRAFFIC'] = (
        df['TRAFFIC_VOLUME'] > df['TRAFFIC_VOLUME'].median()
    ).astype(int)
    
    return df

def get_ml_features(df):
    """
    Identifies feature columns for machine learning.
    """
    non_feature_cols = ['TRAFFIC_VOLUME', 'STREET_NAME', 'OBJECTID', 'ROAD_ID', 'HIGH_TRAFFIC']
    ml_features = [col for col in df.columns if col not in non_feature_cols]
    return ml_features
