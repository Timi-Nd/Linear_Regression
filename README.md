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

### 3. Machine Learning (Multiple Linear Regression)
- **Algorithm**: Implemented a vectorized **Gradient Descent** model from scratch using NumPy.
- **Scaling**: Applied Z-score standardization to ensure balanced feature influence.
- **Optimization**: The model uses 10+ features simultaneously (Age, Lanes, Speed, Transit) to forecast traffic intensity.

---

## 📊 Visual Insights & Deliverables
The pipeline generates high-resolution diagnostic charts to explain the model's logic:
- **Feature Importance**: Visualizes which physical factors most significantly impact road wear.
- **Model Accuracy**: A regression plot showing how closely my predictions match real-world data.
- **Residual Analysis**: Evaluates prediction errors to ensure model reliability.
- **Correlation Heatmap**: Maps the relationships between infrastructure attributes (revealing weak predictors and redundancies).

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

## 💡 Key Takeaway: The "Neglect Gap" & Misguided Decisions
This project is a lesson in critical thinking. While the computer found that older roads have less traffic, my own experience shows those roads are often the most neglected. 

Furthermore, the **Correlation Heatmap** revealed that most physical features are actually weak predictors of traffic volume. This proves that **Machine Learning without context is dangerous**. It’s a journey from "messy numbers" to "smart decisions"—and knowing when to question the math.

---

## 🌍 Real-World Impact
The insights from this model aren't just numbers—they are a roadmap for action. By predicting traffic stress, I can help the city transition from **Reactive Maintenance** to **Proactive Maintenance**. This saves taxpayer dollars and ensures that Saskatoon's infrastructure keeps up with its growth.

**#MachineLearning #DataScience #Saskatoon #SmartCity #Python #AI #TimiNdubuisi**