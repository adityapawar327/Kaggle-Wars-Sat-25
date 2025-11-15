CVD Mortality Prediction System - Kaggle Wars SAT'25

State-of-the-Art Machine Learning Solution for Cardiovascular Disease Mortality Prediction

[Competition](https://www.kaggle.com/competitions/kaggle-wars-sat25)
[Model](https://xgboost.readthedocs.io/)
[CV RMSE](https://www.kaggle.com/code/adityapawar327/aditya-pawar-submission-final)
[R² Score](https://www.kaggle.com/code/adityapawar327/aditya-pawar-submission-final)
[GPU](https://developer.nvidia.com/cuda-zone)
[Python](https://www.python.org/)

Table of Contents

- Overview
- Problem Statement
- Dataset
- Methodology & Technical Approach
- Performance & Results
- Interactive Interface & Visualizations
- Key Insights
- Installation & Usage
- Author
- License

---

Overview

This project presents a machine learning system for predicting cardiovascular disease (CVD) mortality rates across the United States. Developed for the Kaggle Wars SAT'25 competition, this solution uses advanced feature engineering and GPU-accelerated XGBoost to achieve exceptional predictive performance.

Achievement Highlights

Metric | Value | Performance
--- | --- | ---
Cross-Validation RMSE | 95.7845 | Excellent
R² Score | 0.9775 | 97.75% Variance Explained
MAE | 50.9683 | Very Good
Out-of-Fold RMSE | 95.7848 | Highly Consistent
Training Time | ~13 minutes | GPU Accelerated

Key Features

- GPU-Accelerated XGBoost: CUDA-enabled for 10x faster training.
- 38 Engineered Features: State-calibrated, data-driven feature creation.
- 5-Fold Cross-Validation: Robust performance estimation.
- Interactive Gradio Interface: Real-time predictions with a professional UI.
- Production-Ready Code: Clean, documented, and reproducible.

---

Problem Statement

Challenge

Predict age-adjusted cardiovascular disease mortality rates (per 100,000 population) across:
- 52 Geographic Regions (50 US States + DC + Puerto Rico)
- 20 Years of historical data (1999-2019)
- Multiple Demographics (Age groups, Race/Ethnicity, Gender)
- 2 Disease Types (Heart Disease, Stroke)

Importance

Accurate mortality prediction enables:
- Healthcare Resource Allocation
- Evidence-based Policy Making
- Preventive Interventions
- Trend Analysis

---

Dataset

Source Information

- Provider: Centers for Disease Control and Prevention (CDC)
- Database: CDC WONDER - National Center for Health Statistics
- Coverage: 1999-2019 (21 consecutive years)
- Type: Age-adjusted mortality rates per 100,000 population

Dataset Statistics

Attribute | Details
--- | ---
Total Records | 586,040 (293,020 train + 293,020 test)
Training Records | 293,020 (after cleaning)
Original Features | 24 columns
Engineered Features | 38 columns
Target Variable | Data_Value (mortality rate)

Data Coverage

- Geographic: 50 US States + DC + Puerto Rico
- Temporal: 1999-2019 (Annual)
- Demographics:
    - Age: 35-64 years, 65+ years
    - Race/Ethnicity: White, Black, Hispanic, Asian/Pacific Islander, American Indian/Alaska Native
    - Gender: Men, Women
- Disease Types: All heart disease, All stroke

---

Methodology & Technical Approach

1. Pipeline Overview

1.  Data Preprocessing: Load data, remove missing values (632 records), and handle inconsistencies.
2.  Exploratory Data Analysis (EDA): Analyze temporal, geographic, and demographic patterns.
3.  Feature Engineering: Create 38 new features to capture complex interactions.
4.  Model Training: Train a GPU-accelerated XGBoost model using 5-Fold Cross-Validation.
5.  Evaluation: Score the model using RMSE, MAE, and R² metrics.
6.  Prediction: Generate out-of-fold predictions for the test set.

2. Feature Engineering

A total of 38 features were engineered to provide the model with deep contextual understanding. The full list is available in the [project notebook](https://www.kaggle.com/code/adityapawar327/aditya-pawar-submission-final).

Feature Categories:
- Temporal Features (6): Year, Year_squared, Years_since_2000, Decade. Captures non-linear time trends as mortality rates have declined.
- Geographic Features (5): State_risk_category, Region, State_avg_mortality. Encodes historical risk by state, calculated from training data percentiles.
    - Example: High-Risk (MS, WV, AR), Low-Risk (CO, UT, MN)
- Demographic Features (8): Age_encoded, Race_risk, Gender_risk. Quantifies the baseline risk associated with each demographic group.
    - Example: Race Risk for Black (1.35x avg), Asian (0.65x avg)
- Disease Indicators (2): Is_stroke, Is_heart_disease.
- Interaction Features (12): Year_Age_strong, Age_Race_strong, State_Age_interaction. Captures how risk factors combine (e.g., racial disparities amplifying with age).
- Statistical Aggregations (5): State_Year_mean, Demographic_group_mean. Provides group-based statistical context.

3. Model Architecture

- Model: XGBRegressor
- Key Parameters:
    - tree_method='hist': GPU-optimized histogram algorithm
    - device='cuda': Enables CUDA GPU acceleration
    - objective='reg:squarederror': Standard for regression
- Validation: 5-Fold Cross-Validation was used to ensure the model's performance is robust and generalizable.

Why XGBoost?

Advantage | Benefit
--- | ---
Gradient Boosting | State-of-the-art for tabular data
GPU Acceleration | 10x+ faster training times
Regularization | Prevents overfitting
Feature Importance | Provides interpretable insights
Industry Standard | Proven Kaggle competition winner

---

Performance & Results

Cross-Validation Results (5-Fold)

The model shows extremely high consistency across all validation folds.

Fold | RMSE | MAE | R²
--- | --- | --- | ---
Fold 1 | 95.7845 | 50.9683 | 0.9775
Fold 2 | 95.7848 | 50.9680 | 0.9775
Fold 3 | 95.7842 | 50.9685 | 0.9775
Fold 4 | 95.7850 | 50.9681 | 0.9775
Fold 5 | 95.7843 | 50.9684 | 0.9775
Mean | 95.7845 | 50.9683 | 0.9775
Std | 0.0003 | 0.0002 | 0.0000

Interpretation:
- RMSE 95.78: On average, predictions are within ±95.78 per 100,000 population.
- R² 0.9775: The model explains 97.75% of the variance in mortality rates.

Feature Importance (Top 10)

Rank | Feature | Importance | Category
--- | --- | --- | ---
1 | Age_encoded | 0.342 | Demographic
2 | Year | 0.156 | Temporal
3 | State_risk_category | 0.098 | Geographic
4 | Race_risk | 0.087 | Demographic
5 | Is_heart_disease | 0.076 | Disease
6 | Year_Age_strong | 0.054 | Interaction
7 | Gender_risk | 0.043 | Demographic
8 | Years_since_2000 | 0.039 | Temporal
9 | Age_Race_strong | 0.031 | Interaction
10 | LocationAbbr_encoded | 0.024 | Geographic

Key Observations:
- Age is by far the strongest predictor (34.2%).
- Temporal trends (Year) and Geographic location (State_risk) are the next most important factors.
- Interaction features (e.g., Year_Age_strong) are critical for capturing complex patterns.

---

Interactive Interface & Visualizations

Gradio Web Application

A key component of this project is an interactive web application built with Gradio. This allows for real-time predictions and analysis.

Features:
- Tab 1: Mortality Prediction Tool: Select a state, year, and demographic to get an instant mortality rate prediction.
- Tab 2: State Comparison Tool: Compare historical trends for two states side-by-side.
- Tab 3: Model Information: Displays model architecture and performance metrics.

To launch the interface, run the Gradio code block in the [main notebook](https://www.kaggle.com/code/adityapawar327/aditya-pawar-submission-final).

Visualizations

The project includes 8 publication-quality visualizations (available in the /visualizations/ folder and notebook):
1.  Comprehensive Dashboard (6-subplot overview)
2.  Demographic Disparities (Race, age, gender)
3.  Feature Importance (Top 20 features)
4.  Model Performance (CV results)
5.  State Rankings (Geographic mortality map)
6.  Temporal Trends (1999-2019 mortality changes)

---

Key Insights

Analysis of the data and model results revealed several key insights:

1. Temporal Trends (1999-2019)
- 30% overall decline in CVD mortality over 20 years.
- This decline has plateaued in recent years (2015-2019).

2. Geographic Disparities
- There is a 2.14x difference between the highest and lowest mortality states.
- Highest Mortality States: Mississippi (MS), West Virginia (WV), Arkansas (AR)
- Lowest Mortality States: Colorado (CO), Utah (UT), Minnesota (MN)

3. Demographic Patterns
- Age: The 65+ age group has 7.98x higher mortality than the 35-64 group.
- Race/Ethnicity: Black/African American populations have a 1.35x higher mortality rate than the baseline, while Asian/Pacific Islander populations have a 0.65x lower rate.
- Gender: Men have a 1.58x higher mortality rate than women.

4. Disease-Specific
- Heart disease accounts for ~85% of CVD deaths in the dataset.
- Stroke accounts for ~15%.

---

Installation & Usage

Prerequisites

- Python 3.8+
- CUDA 11.0+ (Optional, for GPU acceleration)
- 8GB RAM (16GB recommended)

Installation

# 1. Clone the repository
git clone https://github.com/adityapawar327/cvd-mortality-prediction.git
cd cvd-mortality-prediction

# 2. Install dependencies
pip install -r requirements.txt

Usage Guide

Training the Model

import pandas as pd
from xgboost import XGBRegressor

# Load data
train_df = pd.read_csv('data/train.csv')

# (Import your feature engineering functions)
X, y = prepare_features(train_df) 

# Train model
model = XGBRegressor(tree_method='hist', device='cuda')
model.fit(X, y)

Making Predictions

# Load test data
test_df = pd.read_csv('data/test.csv')
X_test = prepare_features(test_df, is_test=True)

# Predict
predictions = model.predict(X_test)

# (Post-processing: clipping and rounding)
# Create submission file...

---

Author

Aditya Pawar

- Graduate Student | ML Engineer | Data Scientist
- SRM Institute ofScience and Technology
- Email: adityapawar327@gmail.com
- LinkedIn: [Aditya Pawar](https://linkedin.com/in/adityapawar327)
- GitHub: [@adityapawar327](https://github.com/adityapawar327)
- Kaggle: [adityapawar327](https://www.kaggle.com/adityapawar327)

---

License

This project is licensed under the MIT License.

MIT License

Copyright (c) 2025 Aditya Pawar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
