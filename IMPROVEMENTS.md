# Project Improvements: Saskatoon Traffic ML Journey

This document outlines the major improvements made to the Saskatoon Infrastructure Machine Learning project to enhance data quality, model performance, and code maintainability.

## 1. Data Cleaning & Quality Assurance
- **Outlier Removal:** Identified and removed 19 rows where `TRAFFIC_VOLUME` was recorded as `999,999`. These were determined to be placeholder values (missing data) rather than actual measurements. Removing them significantly reduced noise and improved the reliability of both models.
- **Robust Feature Scaling:** Updated the scaling logic to handle columns with zero variance (no change in value), preventing division-by-zero errors.

## 2. Advanced Feature Engineering
- **One-Hot Encoding (OHE):** Categorical variables like `Snow_Route`, `Maintenance_Group`, `route_id`, and `Neighbourhood_Name` were previously treated as numeric labels. They have been converted into binary (0/1) features. This prevents the model from assuming a mathematical order or rank where none exists.
- **Dimensionality Expansion:** The feature set was expanded from ~10 basic columns to 108 features, capturing the unique impact of each neighborhood and infrastructure type on traffic volume.
- **Type Safety:** Automated the conversion of boolean and integer features to float format to ensure compatibility with NumPy-based mathematical operations.

## 3. Model Enhancements
- **Logistic Regression Implementation:** Added a binary classification model (Logistic Regression) from scratch to predict "High Traffic" vs. "Low Traffic" (above/below median).
- **Optimization Tracking:** Integrated cost history tracking for Gradient Descent, allowing for the analysis of model convergence over 5,000 iterations.
- **Accuracy Boost:** These improvements increased the Logistic Regression classification accuracy from **67.5%** to **83.3%**.

## 4. Enhanced Visualizations
New analytical plots were added to provide deeper insights into model behavior:
- **`gd_convergence.png`**: Visualizes the cost reduction over time, showing how the models "learn."
- **`classification_matrix.png`**: A confusion matrix heatmap showing the precision and recall of the Logistic Regression model.
- **`feature_importance_logistic.png`**: Ranks the features that have the strongest impact on classifying a road as "High Traffic."
- **`correlation_heatmap.png`**: Streamlined to focus on core relationships, making it easier to read after the introduction of many OHE features.

## 5. Architectural Refactoring
- **Modular Design:** Created `data_processor.py` to separate data preparation logic from the machine learning pipeline. 
- **Cleaner `main.py`**: The main script now focuses strictly on model training and evaluation, making the codebase easier to read, maintain, and extend.

## 6. Aesthetic & Elegant Visualizations
The output plots have been overhauled to be more visually appealing and professional:
- **Consistent Modern Theme:** Applied a custom clean theme using Seaborn and Matplotlib with bold labels and refined color palettes (`mako`, `RdBu_r`).
- **Enhanced Convergence Plot:** Added shaded areas and refined line weights to better visualize the optimization process.
- **Top Feature Focus:** Bar charts now focus on the Top 15 most impactful features, significantly improving readability given the high dimensionality (108 features).
- **High-Resolution Outputs:** All plots are now saved at 300 DPI for crystal-clear quality.
- **Improved Clarity:** Added grid lines, removed redundant spines, and used clearer annotations in the confusion matrix and correlation heatmap.

## 7. Descriptive Labeling
Improved the interpretability of machine learning insights by replacing cryptic numeric codes with human-readable labels:
- **Neighbourhood Mapping:** Categorical ID codes for all 84+ neighborhoods (e.g., `12`, `15`) were mapped to their actual names (e.g., `River Heights`, `Hudson Bay Park`).
- **Feature Sanitization:** Updated `Maintenance_Group`, `Snow_Route`, and `Road_Surface_Type` to use descriptive labels (e.g., `Priority 1`, `Emergency Route`, `Asphalt`) instead of raw database integers.
- **Readable Insights:** This change ensures that the Feature Importance plots and Model Weights are immediately understandable by stakeholders without needing a data dictionary.

## 8. Data-Driven Strategic Insights
The optimized pipeline revealed specific trends that provide actionable intelligence for urban planning:
- **Key Traffic Drivers:** Identified **Snow Routes** (Emergency status), **Lane Count**, and **Speed Limits** as the primary indicators of traffic density.
- **Geographic Hubs:** Successfully identified high-volume industrial corridors (e.g., **AgPro Industrial**, **West Industrial**) versus low-volume residential zones (e.g., **The Willows**).
- **Placeholder Identification:** Discovered and purged 19 placeholder values (999,999 volume), ensuring the model reflects the actual city state.
- **Model Confidence:** Integrated **Decision Boundary** visualizations and **Confusion Matrices** to prove that the 83.72% accuracy is backed by mathematically sound separation of classes.

## 9. Decision Boundary Visualization
Added a sophisticated visualization to illustrate the Logistic Regression model's classification logic:
- **decision_boundary.png**: A 2D plot showing how the model separates "High Traffic" from "Low Traffic" roads.
- **Dimensionality Handling:** Since the model uses 106 features, the plot focuses on the two most impactful numeric features (SPEED_LIMIT and LANE_COUNT) while holding others at their mean.
- **Probabilistic Regions:** Uses a color gradient (Red-Yellow-Green) to show the probability of high traffic across the feature space, with a dashed line representing the 0.5 decision threshold.

---
**Author:** Timi Ndubuisi
**Status:** Pipeline Optimized & Verified
