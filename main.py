# Machine Learning Journey: Linear Regression from Scratch
# Developed by: Timi Ndubuisi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_processor import clean_and_prepare_data, get_ml_features

# Set professional style for plots
sns.set_theme(style="white", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def run_pipeline():
    print("--- Timi Ndubuisi's ML Journey: Practice & Critical Analysis ---")
    print("Goal: Demonstrate how data can lead to insights—and misguided decisions.")
    
    # 1. LOAD & PREPARE DATA
    print("\n1. Loading and improving prepared infrastructure data...")
    final_df = pd.read_csv('Saskatoon_Infrastructure_Ready.csv')
    
    # Use improved logic from data_processor.py
    final_df = clean_and_prepare_data(final_df)
    ml_features = get_ml_features(final_df)
    
    target = 'TRAFFIC_VOLUME'
    target_class = 'HIGH_TRAFFIC'
    
    print(f"Dataset ready: {final_df.shape[0]} rows, {final_df.shape[1]} columns.")
    print(f"Features used: {len(ml_features)}")
    
    # 2. MACHINE LEARNING (Multiple Linear Regression)
    print("\n2. Training Linear Regression (Gradient Descent)...")
    X = final_df[ml_features].values
    y = final_df[target].values
    
    # Scaling
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    # Avoid division by zero for columns with no variance
    X_std[X_std == 0] = 1.0
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
    plt.plot(lin_cost_history, label='Linear Regression', color='#3498db', linewidth=2.5)
    plt.plot(log_cost_history, label='Logistic Regression', color='#2ecc71', linewidth=2.5)
    plt.fill_between(range(len(lin_cost_history)), lin_cost_history, color='#3498db', alpha=0.1)
    plt.fill_between(range(len(log_cost_history)), log_cost_history, color='#2ecc71', alpha=0.1)
    plt.xlabel("Iterations", fontsize=12, fontweight='bold')
    plt.ylabel("Cost (Error)", fontsize=12, fontweight='bold')
    plt.title("Model Convergence: Gradient Descent Optimization", fontsize=16, pad=20, fontweight='bold')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('gd_convergence.png', dpi=300)

    # Classification Results (Confusion Matrix style)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_class, y_pred_class)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', 
                cbar=False, annot_kws={"size": 16, "fontweight": 'bold'},
                xticklabels=['Low Traffic', 'High Traffic'], 
                yticklabels=['Low Traffic', 'High Traffic'])
    plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Category', fontsize=12, fontweight='bold')
    plt.title(f'Logistic Regression Performance\nAccuracy: {accuracy:.2%}', fontsize=16, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('classification_matrix.png', dpi=300)
    
    # Feature Importance (Linear Regression) - Top 15
    plt.figure(figsize=(12, 8))
    top_n = 15
    sorted_idx_lin = np.argsort(np.abs(w_lin))[-top_n:]
    importances = w_lin[sorted_idx_lin]
    feature_names = [ml_features[i] for i in sorted_idx_lin]
    
    colors = ['#3498db' if x >= 0 else '#e74c3c' for x in importances]
    plt.barh(feature_names, importances, color=colors, alpha=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(f"Linear Regression: Top {top_n} Impactful Features", fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Weight Magnitude", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)

    # Feature Importance (Logistic Regression) - Top 15
    plt.figure(figsize=(12, 8))
    sorted_idx_log = np.argsort(np.abs(w_log))[-top_n:]
    importances_log = w_log[sorted_idx_log]
    feature_names_log = [ml_features[i] for i in sorted_idx_log]
    
    colors_log = ['#1abc9c' if x >= 0 else '#f39c12' for x in importances_log]
    plt.barh(feature_names_log, importances_log, color=colors_log, alpha=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(f"Logistic Regression: Top {top_n} Impactful Features", fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Weight Magnitude", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance_logistic.png', dpi=300)
    
    # Actual vs Predicted
    plt.figure(figsize=(9, 9))
    sns.regplot(x=y, y=y_pred, scatter_kws={'alpha':0.4, 'color':'#34495e', 's':30}, 
                line_kws={'color':'#e74c3c', 'linewidth':3})
    plt.xlabel("Actual Traffic Volume", fontsize=12, fontweight='bold')
    plt.ylabel("Predicted Traffic Volume", fontsize=12, fontweight='bold')
    plt.title("Regression Accuracy: Actual vs Predicted", fontsize=16, pad=20, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300)
    
    # Residuals
    plt.figure(figsize=(11, 6))
    sns.scatterplot(x=y_pred, y=(y - y_pred), alpha=0.4, color='#3498db', s=40)
    plt.axhline(0, color='#e74c3c', linestyle='--', linewidth=2)
    plt.title("Residual Analysis: Prediction Errors", fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Predicted Traffic Volume", fontsize=12, fontweight='bold')
    plt.ylabel("Residual (Actual - Predicted)", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=300)
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr_features = ['ROAD_AGE', 'LANE_COUNT', 'SPEED_LIMIT', 'TRAFFIC_VOLUME']
    mask = np.triu(np.ones_like(final_df[corr_features].corr(), dtype=bool))
    sns.heatmap(final_df[corr_features].corr(), mask=mask, annot=True, 
                cmap='RdBu_r', fmt=".2f", center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .8}, annot_kws={"size": 14})
    plt.title("Core Feature Correlation Matrix", fontsize=16, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)

    # 5. DECISION BOUNDARY VISUALIZATION
    # 5. DECISION BOUNDARY VISUALIZATION
    # Since we have 100+ features, we'll pick the two most important numeric features
    # for a 2D visualization of the decision boundary.
    print('\n5. Generating Decision Boundary plot...')
    # Identify top numeric features (avoiding OHE categories for a smooth boundary plot)
    numeric_features = ['LANE_COUNT', 'SPEED_LIMIT', 'ROAD_AGE']
    feat_idx = [ml_features.index(f) for f in numeric_features]
    # Let's use the two with highest absolute weights among numeric
    top_two_idx = np.argsort(np.abs(w_log[feat_idx]))[-2:]
    idx1, idx2 = feat_idx[top_two_idx[1]], feat_idx[top_two_idx[0]]
    feat1_name, feat2_name = ml_features[idx1], ml_features[idx2]
    
    # Create a mesh grid
    x_min, x_max = X_scaled[:, idx1].min() - 0.5, X_scaled[:, idx1].max() + 0.5
    y_min, y_max = X_scaled[:, idx2].min() - 0.5, X_scaled[:, idx2].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # To predict on the grid, we need a full feature vector (108 cols)
    # We'll use the mean values for all other features
    grid_data = np.zeros((xx.ravel().shape[0], len(ml_features)))
    grid_data[:, :] = np.mean(X_scaled, axis=0)
    grid_data[:, idx1] = xx.ravel()
    grid_data[:, idx2] = yy.ravel()
    
    z_vals = np.dot(grid_data, w_log) + b_log
    Z_prob = 1 / (1 + np.exp(-np.clip(z_vals, -500, 500)))
    Z = Z_prob.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    # Plot probability regions
    contour = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.3)
    plt.colorbar(contour, label='Probability of High Traffic')
    # Plot the 0.5 boundary
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    # Overlay actual data points (subset for clarity)
    np.random.seed(42)
    sample_idx = np.random.choice(len(y_class), 500, replace=False)
    plt.scatter(X_scaled[sample_idx, idx1], X_scaled[sample_idx, idx2], 
                c=y_class[sample_idx], cmap='RdYlGn', edgecolors='k', alpha=0.6, s=40)
    
    plt.xlabel(f'{feat1_name} (Standardized)', fontweight='bold')
    plt.ylabel(f'{feat2_name} (Standardized)', fontweight='bold')
    plt.title('Logistic Regression Decision Boundary', fontsize=16, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('decision_boundary.png', dpi=300)

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
