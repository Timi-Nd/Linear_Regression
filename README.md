# Saskatoon Infrastructure: My Machine Learning Journey

By: **Timi Ndubuisi**

Welcome to my first project exploring the world of Machine Learning (ML). This repository documents my journey into understanding the math and algorithms that power AI, starting with the fundamentals of **Multiple Linear Regression**.

# Saskatoon Predictive Maintenance & Optimization Pipeline

## Overview
This project is an end-to-end Machine Learning pipeline developed to analyze and predict traffic patterns for the City of Saskatoon's road infrastructure. By integrating road datasets with transit information, the model identifies high-wear segments and provides data-driven insights for urban maintenance prioritization.

This project represents my journey into understanding **ML and AI**, moving from raw data processing to predictive modeling and insight generation.

---

## The Pipeline Workflow

### 0. Data Sources
The data used in this project is sourced from the **[City of Saskatoon Open Data Portal](https://opendata-saskatoon.hub.arcgis.com/)**.
- **Road Infrastructure**: `Road_-_Divided.csv`
- **Transit Data**: `Transit_Routes.csv`, `Transit_Shapes.csv`, `Transit_Stop_Times.csv`
- **Demographics & Environment**: Census and Weather datasets were also integrated for broader context.

### 1. Data Integration & Cleaning
- **Multi-Source Loading**: Ingests raw CSV data from the City of Saskatoon (Roads, Transit, Shapes).
- **Quality Assurance**: Automatically removes segments with missing critical metrics like `TRAFFIC_VOLUME` or `ROAD_LENGTH`.
- **Intelligent Merging**: Bridges the gap between infrastructure maps and transit routes using substring matching logic on street names.

### 2. Feature Engineering
- **ROAD_AGE**: Calculated historical depth of infrastructure.
- **IS_PRIORITY**: Binary classification of high-maintenance zones.
- **Categorical Context**: Encoding for road types, surface materials, and maintenance groups.

### 3. Machine Learning (Multiple Linear Regression)
- **Algorithm**: Implemented a vectorized **Gradient Descent** model from scratch using NumPy.
- **Scaling**: Applied Z-score standardization to ensure balanced feature influence.
- **Optimization**: The model uses 10+ features simultaneously (Age, Lanes, Speed, Transit) to forecast traffic intensity.

---

## Visual Insights & Deliverables
The pipeline generates high-resolution diagnostic charts to explain the model's logic:
- **Feature Importance**: Visualizes which physical factors (like `LANE_COUNT` or `ROAD_AGE`) most significantly impact road wear.
- **Model Accuracy**: A regression plot showing how closely my predictions match real-world data.
- **Residual Analysis**: Evaluates prediction errors to ensure model reliability.
- **Correlation Heatmap**: Maps the complex relationships between infrastructure attributes.

---

## Project Structure
- `main.py`: The "Brain" of the project. Run this to do everything (clean data, train model, make charts).
- `PROJECT_GUIDE.md`: **Start here!** A simple, easy-to-read guide explaining the whole project.
- `Saskatoon_Infrastructure_Ready.csv`: The cleaned data the computer uses to learn.
- `Saskatoon_Predictions.csv`: The final list of roads with our computer's predictions.
- `*.png`: Professional charts that visualize our findings.

---

## How to Run
1. Ensure you have Python installed.
2. Run the main script: `python main.py`
3. Check the `PROJECT_GUIDE.md` to understand what the results mean.

---

## Key Takeaway
This project proves that raw city data can be turned into a smart tool to help save money and keep roads safe. It’s a journey from "messy numbers" to "smart decisions."

**#MachineLearning #DataScience #Saskatoon #SmartCity #Python #AI**
