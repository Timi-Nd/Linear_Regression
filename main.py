# Machine Learning Journey: Linear Regression from Scratch
# Developed by: Timi Ndubuisi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set professional style for plots
sns.set_theme(style="whitegrid", context="talk")

def run_pipeline():
    print("--- Timi Ndubuisis ML Journey: Saskatoon Predictive Maintenance ---")
    
    # 1. LOAD & CLEAN
    print("\n1. Loading and cleaning data...")
    df_roads = pd.read_csv('Road_-_Divided.csv')
    df_routes = pd.read_csv('Transit_Routes.csv')
    
    # Remove rows missing critical data
    critical_columns = ['TRAFFIC_VOLUME', 'ROAD_LENGTH']
    df_roads = df_roads.dropna(subset=critical_columns)
    
    # 2. FEATURE ENGINEERING
    print("2. Engineering features...")
    # Road Age calculation
    df_roads['YEAR_CONSTRUCTED'] = pd.to_numeric(df_roads['YEAR_CONSTRUCTED'], errors='coerce')
    df_roads.loc[(df_roads['YEAR_CONSTRUCTED'] > 2026) | (df_roads['YEAR_CONSTRUCTED'] < 1800), 'YEAR_CONSTRUCTED'] = np.nan
    df_roads['ROAD_AGE'] = 2026 - df_roads['YEAR_CONSTRUCTED']
    df_roads['ROAD_AGE'] = df_roads['ROAD_AGE'].fillna(df_roads['ROAD_AGE'].median())
    
    # Priority Flag
    df_roads['IS_PRIORITY'] = df_roads['PRIORITY_MAINTENANCE'].apply(lambda x: 1 if x == 4 else 0)
    
    # Transit Integration (Substring matching logic)
    road_streets = df_roads[['STREET_NAME']].dropna().drop_duplicates()
    road_streets['STREET_NAME_CLEAN'] = road_streets['STREET_NAME'].astype(str).str.lower().str.strip()
    df_routes['route_desc_clean'] = df_routes['route_long_name'].astype(str).str.lower()
    
    mapping = []
    for _, road_row in road_streets.iterrows():
        street = road_row['STREET_NAME_CLEAN']
        if len(street) < 3: continue
        for _, route_row in df_routes.iterrows():
            if street in route_row['route_desc_clean']:
                mapping.append({'STREET_NAME': road_row['STREET_NAME'], 'route_id': route_row['route_id']})
                break
    
    df_mapping = pd.DataFrame(mapping)
    final_df = pd.merge(df_roads, df_mapping, on='STREET_NAME', how='left')
    
    # Impute missing transit and category values
    cols_to_fill = ['route_id', 'Snow_Route', 'Snow_Removal_Designate', 'Maintenance_Group', 
                    'Neighbourhood_Name', 'Road_Surface_Type', 'Road_Structure_Type']
    for col in cols_to_fill:
        final_df[col] = final_df[col].fillna(0).astype(int)
    
    # Select features for ML
    ml_features = ['ROAD_AGE', 'LANE_COUNT', 'SPEED_LIMIT', 'Snow_Route', 
                   'Snow_Removal_Designate', 'Maintenance_Group', 'Neighbourhood_Name',
                   'route_id', 'Road_Surface_Type', 'Road_Structure_Type']
    target = 'TRAFFIC_VOLUME'
    identifiers = ['STREET_NAME', 'OBJECTID', 'ROAD_ID']
    
    final_df = final_df[ml_features + [target] + identifiers]
    final_df.to_csv('Saskatoon_Infrastructure_Ready.csv', index=False)
    print(f"Dataset finalized: {final_df.shape[0]} rows, {final_df.shape[1]} columns.")
    
    # 3. MACHINE LEARNING (Multiple Linear Regression)
    print("\n3. Training model (Gradient Descent)...")
    X = final_df[ml_features].values
    y = final_df[target].values
    
    # Scaling
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    y_mean, y_std = np.mean(y), np.std(y)
    X_scaled = (X - X_mean) / X_std
    y_scaled = (y - y_mean) / y_std
    
    # Gradient Descent
    w = np.zeros(len(ml_features))
    b = 0.0
    lr = 0.01
    m = len(y)
    for i in range(5000):
        error = (np.dot(X_scaled, w) + b) - y_scaled
        w -= lr * (np.dot(X_scaled.T, error) / m)
        b -= lr * (np.sum(error) / m)
        if i % 1000 == 0:
            cost = (1/(2*m)) * np.sum(error**2)
            print(f"   Iteration {i}: Cost {cost:.6f}")
            
    # Predictions
    y_pred_scaled = np.dot(X_scaled, w) + b
    y_pred = (y_pred_scaled * y_std) + y_mean
    
    # 4. VISUALIZATION & EXPORT
    print("\n4. Generating insights and saving plots...")
    
    # Feature Importance
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(np.abs(w))
    plt.barh([ml_features[i] for i in sorted_idx], w[sorted_idx], 
             color=['skyblue' if x >= 0 else 'salmon' for x in w[sorted_idx]])
    plt.title("Feature Importance (Relative Impact on Traffic)")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Actual vs Predicted
    plt.figure(figsize=(8, 8))
    sns.regplot(x=y, y=y_pred, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.xlabel("Actual Traffic")
    plt.ylabel("Predicted Traffic")
    plt.title("Model Accuracy: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    
    # Residuals
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=(y - y_pred), alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residual Analysis (Prediction Errors)")
    plt.tight_layout()
    plt.savefig('residual_analysis.png')
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(final_df[ml_features + [target]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    
    # Save results
    final_df['PREDICTED_TRAFFIC'] = y_pred
    final_df['RESIDUALS'] = y - y_pred
    final_df.to_csv('Saskatoon_Predictions.csv', index=False)
    
    with open('Model_Weights.txt', 'w') as f:
        f.write("--- Model weights ---\n")
        for f_name, weight in zip(ml_features, w):
            f.write(f"{f_name}: {weight:.6f}\n")
            
    print("\nPipeline complete! All files, plots, and insights are ready.")

if __name__ == "__main__":
    run_pipeline()
