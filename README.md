# Saskatoon Infrastructure: My Machine Learning Journey

By: **Timi Ndubuisi**

Welcome to my first project exploring the world of Machine Learning (ML). This repository documents my journey into understanding the math and algorithms that power AI, starting with the fundamentals of **Multiple Linear Regression**.

## 🎯 Project Goal
My goal isn't just to build a model; it's to practice **Data Analysis and Machine Learning**. Specifically, I want to explore how data can be used to understand urban infrastructure—but also to show how **poor or incomplete data analysis can lead to misguided decisions**.

# Saskatoon Predictive Maintenance: A Learning Journey

## Overview
This project is an end-to-end Machine Learning pipeline developed to analyze and predict traffic patterns for the City of Saskatoon's road infrastructure. By integrating road datasets with transit information, the model identifies high-wear segments and provides data-driven insights for urban maintenance prioritization.

This project represents my journey into understanding **ML and AI**, moving from raw data processing to predictive modeling and insight generation.

---

## The Pipeline Workflow

### 0. Data Sources
The data used in this project is sourced from the **[City of Saskatoon Open Data Portal](https://opendata-saskatoon.hub.arcgis.com/)**.
- **Road Infrastructure**: `Road_-_Divided.csv`
- **Transit Data**: `Transit_Routes.csv`

---

## 🏗️ The Pipeline Workflow (Details)
### 1. Data Integration & Cleaning
- **Multi-Source Loading**: Ingests raw CSV data from the City of Saskatoon.
- **Quality Assurance**: Automatically removes segments with missing critical metrics like `TRAFFIC_VOLUME` or `ROAD_LENGTH`.
- **Intelligent Merging**: Bridges the gap between infrastructure maps and transit routes using substring matching logic on street names.

### 2. Feature Engineering
- **ROAD_AGE**: Calculated historical depth of infrastructure.
- **IS_PRIORITY**: Binary classification of high-maintenance zones.
- **Categorical Context**: Encoding for road types, surface materials, and maintenance groups.

### 3. Machine Learning Models
- **Multiple Linear Regression**: Implemented a vectorized **Gradient Descent** model from scratch using NumPy to forecast traffic intensity.
- **Logistic Regression**: Added a binary classification model to predict "High Traffic" vs. "Low Traffic" (above/below median).
- **Optimization**: The models use 108 features (including One-Hot Encoded neighborhoods and infrastructure types) to generate predictions.
- **Accuracy Boost**: Reached a classification accuracy of **83.72%** through advanced data cleaning and feature engineering.

---

## 📊 Visual Insights & Deliverables
The pipeline generates high-resolution diagnostic charts to explain the model's logic:
- **Feature Importance**: Visualizes which factors most significantly impact traffic volume (e.g., Snow Routes, Lanes).
- **Model Accuracy**: A regression plot showing how closely my predictions match real-world data.
- **Decision Boundary**: A 2D visualization showing exactly where the model switches between "Low" and "High" traffic categories.
- **Convergence Plot**: Shows the "Gradient Descent" process and how the model reached stability.
- **Classification Matrix**: A confusion matrix showing the reliability of the Logistic Regression model.
- **Residual Analysis**: Evaluates prediction errors to ensure model reliability.
- **Correlation Heatmap**: Maps the relationships between infrastructure attributes.

---

## 📂 Project Structure
- `main.py`: The "Brain" of the project. Run this to clean data, train the model, and generate charts.
- `PROJECT_GUIDE.md`: **Start here!** A simple guide explaining the learning journey and results.
- `Saskatoon_Infrastructure_Ready.csv`: The cleaned data used for training.
- `Saskatoon_Predictions.csv`: Final outputs with predictions and residuals.
- `*.png`: Professional charts visualizing the findings.

---

## 🛠️ How to Run
1. Ensure you have Python installed.
2. Run the main script: `python main.py`
3. Read the `PROJECT_GUIDE.md` to understand the insights and limitations.

---

## 💡 Key Takeaways & Strategic Insights
The data tells a story of a well-organized city where infrastructure capacity and maintenance priority align with actual usage:

- **Primary Drivers:** **Snow Routes** (especially Emergency status), **Lane Count**, and **Speed Limits** are the strongest indicators of high traffic volume.
- **Neighborhood Hubs:** We identified "Traffic Hubs" like **AgPro Industrial** and **West Industrial** as major transit corridors, while areas like **The Willows** remain quiet residential zones.
- **Data Quality:** We uncovered and removed 19 "placeholder" outliers (999,999 volume), which allowed the model to see real trends.
- **The "OHE" Effect:** By using One-Hot Encoding and mapping numeric codes to real names, the model's accuracy jumped from 67.5% to **83.72%**.

This project proves that **Machine Learning without context is limited**. It’s a journey from "messy numbers" to "smart decisions"—and knowing when to question the math.

---

## 🌍 Real-World Impact
The insights from this model aren't just numbers—they are a roadmap for action. By predicting traffic stress, I can help the city transition from **Reactive Maintenance** to **Proactive Maintenance**. This saves taxpayer dollars and ensures that Saskatoon's infrastructure keeps up with its growth.

**#MachineLearning #DataScience #Saskatoon #SmartCity #Python #AI #TimiNdubuisi**