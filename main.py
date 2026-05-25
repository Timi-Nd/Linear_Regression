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
    print("--- Timi Ndubuisi's ML Journey: Practice & Critical Analysis ---")
    print("Goal: Demonstrate how data can lead to insights—and misguided decisions.")
    
    # 1. LOAD PREPARED DATA
    print("\n1. Loading prepared infrastructure data...")
    final_df = pd.read_csv('Saskatoon_Infrastructure_Ready.csv')
    
    # Select features for ML
    ml_features = ['ROAD_AGE', 'LANE_COUNT', 'SPEED_LIMIT', 'Snow_Route', 
                   'Snow_Removal_Designate', 'Maintenance_Group', 'Neighbourhood_Name',
                   'route_id', 'Road_Surface_Type', 'Road_Structure_Type']
    target = 'TRAFFIC_VOLUME'
    
    print(f"Dataset loaded: {final_df.shape[0]} rows, {final_df.shape[1]} columns.")
    
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
    # A heatmap shows how much each feature relates to others and the target.
    # Numbers near 1.0 (Darker Red) = Strong Positive relationship.
    # Numbers near -1.0 (Darker Blue) = Strong Negative relationship.
    # Numbers near 0 (White/Light) = Weak or no relationship.
    #
    # TIMI'S INSIGHT: The heatmap shows that most features correlate weakly with TRAFFIC_VOLUME.
    # This is a major lesson: the code works, but the data needs better 'clues' to be truly accurate!
    # Notable findings: Surface/Structure redundancy (0.88), Maintenance/Age (0.26), Speed/Age (-0.25).
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
