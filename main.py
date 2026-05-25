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
    
    # Classification target for Logistic Regression
    final_df['HIGH_TRAFFIC'] = (
        final_df['TRAFFIC_VOLUME'] > final_df['TRAFFIC_VOLUME'].median()
    ).astype(int)
    target_class = 'HIGH_TRAFFIC'
    
    print(f"Dataset loaded: {final_df.shape[0]} rows, {final_df.shape[1]} columns.")
    
    # 2. MACHINE LEARNING (Multiple Linear Regression)
    print("\n2. Training Linear Regression (Gradient Descent)...")
    X = final_df[ml_features].values
    y = final_df[target].values
    
    # Scaling
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    y_mean, y_std = np.mean(y), np.std(y)
    X_scaled = (X - X_mean) / X_std
    y_scaled = (y - y_mean) / y_std
    
    # Linear Regression Gradient Descent
    w_lin = np.zeros(len(ml_features))
    b_lin = 0.0
    lr = 0.01
    m = len(y)
    lin_cost_history = []
    for i in range(5000):
        error = (np.dot(X_scaled, w_lin) + b_lin) - y_scaled
        w_lin -= lr * (np.dot(X_scaled.T, error) / m)
        b_lin -= lr * (np.sum(error) / m)
        cost = (1/(2*m)) * np.sum(error**2)
        lin_cost_history.append(cost)
        if i % 1000 == 0:
            print(f"   Iteration {i}: Cost {cost:.6f}")
            
    # Predictions
    y_pred_scaled = np.dot(X_scaled, w_lin) + b_lin
    y_pred = (y_pred_scaled * y_std) + y_mean

    # 3. MACHINE LEARNING (Logistic Regression)
    print("\n3. Training Logistic Regression (Gradient Descent)...")
    y_class = final_df[target_class].values
    
    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # Logistic Regression Gradient Descent
    w_log = np.zeros(len(ml_features))
    b_log = 0.0
    log_cost_history = []
    for i in range(5000):
        z = np.dot(X_scaled, w_log) + b_log
        h = sigmoid(z)
        error = h - y_class
        w_log -= lr * (np.dot(X_scaled.T, error) / m)
        b_log -= lr * (np.sum(error) / m)
        cost = (-1/m) * np.sum(y_class * np.log(h + 1e-15) + (1 - y_class) * np.log(1 - h + 1e-15))
        log_cost_history.append(cost)
        if i % 1000 == 0:
            print(f"   Iteration {i}: Cost {cost:.6f}")
            
    # Classification Predictions
    y_prob = sigmoid(np.dot(X_scaled, w_log) + b_log)
    y_pred_class = (y_prob >= 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y_class)
    print(f"Logistic Regression Accuracy: {accuracy:.2%}")
    
    # 4. VISUALIZATION & EXPORT
    print("\n4. Generating insights and saving plots...")

    # Gradient Descent Convergence
    plt.figure(figsize=(10, 6))
    plt.plot(lin_cost_history, label='Linear Regression Cost', color='blue')
    plt.plot(log_cost_history, label='Logistic Regression Cost', color='green')
    plt.xlabel("Iterations")
    plt.ylabel("Cost (Error)")
    plt.title("Gradient Descent Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig('gd_convergence.png')

    # Classification Results (Confusion Matrix style)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_class, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Traffic', 'High Traffic'], 
                yticklabels=['Low Traffic', 'High Traffic'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Logistic Regression: Classification Results (Acc: {accuracy:.2%})')
    plt.tight_layout()
    plt.savefig('classification_matrix.png')
    
    # Feature Importance (Linear Regression)
    plt.figure(figsize=(10, 6))
    sorted_idx_lin = np.argsort(np.abs(w_lin))
    plt.barh([ml_features[i] for i in sorted_idx_lin], w_lin[sorted_idx_lin], 
             color=['skyblue' if x >= 0 else 'salmon' for x in w_lin[sorted_idx_lin]])
    plt.title("Linear Regression: Feature Importance")
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    # Feature Importance (Logistic Regression)
    plt.figure(figsize=(10, 6))
    sorted_idx_log = np.argsort(np.abs(w_log))
    plt.barh([ml_features[i] for i in sorted_idx_log], w_log[sorted_idx_log], 
             color=['teal' if x >= 0 else 'orange' for x in w_log[sorted_idx_log]])
    plt.title("Logistic Regression: Feature Importance")
    plt.tight_layout()
    plt.savefig('feature_importance_logistic.png')
    
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
    final_df['HIGH_TRAFFIC_PROB'] = y_prob
    final_df['HIGH_TRAFFIC_PRED'] = y_pred_class
    final_df['RESIDUALS'] = y - y_pred
    final_df.to_csv('Saskatoon_Predictions.csv', index=False)
    
    with open('Model_Weights.txt', 'w') as f:
        f.write("--- Linear Regression Weights ---\n")
        for f_name, weight in zip(ml_features, w_lin):
            f.write(f"{f_name}: {weight:.6f}\n")
        f.write("\n--- Logistic Regression Weights ---\n")
        for f_name, weight in zip(ml_features, w_log):
            f.write(f"{f_name}: {weight:.6f}\n")
        f.write(f"\nClassification Accuracy: {accuracy:.2%}\n")
            
    print("\nPipeline complete! All files, plots, and insights are ready.")

if __name__ == "__main__":
    run_pipeline()
