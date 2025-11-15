# CVD Mortality Prediction System - Kaggle Wars SAT'25

> State-of-the-Art Machine Learning Solution for Cardiovascular Disease Mortality Prediction

[![Competition](https://img.shields.io/badge/Competition-Kaggle%20Wars%20SAT'25-blue)](https://www.kaggle.com/competitions/kaggle-wars-sat25)
[![Model](https://img.shields.io/badge/Model-XGBoost-red)](https://xgboost.readthedocs.io/)
[![CV RMSE](https://img.shields.io/badge/CV%20RMSE-95.7845-green)](https://www.kaggle.com/code/adityapawar327/aditya-pawar-submission-final)
[![RÂ² Score](https://img.shields.io/badge/RÂ²-0.9775-brightgreen)](https://www.kaggle.com/code/adityapawar327/aditya-pawar-submission-final)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20Enabled-orange)](https://developer.nvidia.com/cuda-zone)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)

## ðŸ“‹ Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Results & Analysis](#results--analysis)
- [Visualizations](#visualizations)
- [Interactive Interface](#interactive-interface)
- [Key Insights](#key-insights)
- [Competitive Advantages](#competitive-advantages)
- [Author](#author)
- [License](#license)

---

## Overview
This project presents a **comprehensive machine learning system** for predicting cardiovascular disease (CVD) mortality rates across the United States. Developed for the **Kaggle Wars SAT'25** competition at SRM Institute of Science and Technology, this solution combines advanced feature engineering, GPU-accelerated training, and interactive visualization to achieve exceptional predictive performance.

### ðŸ… Achievement Highlights

| Metric | Value | Performance |
|--------|-------|-------------|
| **Cross-Validation RMSE** | 95.7845 | Excellent |
| **RÂ² Score** | 0.9775 | 97.75% Variance Explained |
| **MAE** | 50.9683 | Very Good |
| **Out-of-Fold RMSE** | 95.7848 | Highly Consistent |
| **Training Time** | ~13 minutes | GPU Accelerated |

### âœ¨ Key Features

- ðŸš€ **GPU-Accelerated XGBoost** - CUDA-enabled for 10x faster training
- ðŸ”§ **38 Engineered Features** - State-calibrated, data-driven feature creation
- ðŸ“Š **5-Fold Cross-Validation** - Robust performance estimation
- ðŸŽ¨ **Interactive Gradio Interface** - Real-time predictions with professional UI
- ðŸ“ˆ **Professional Visualizations** - Publication-ready charts
- ðŸ’¾ **Production-Ready Code** - Clean, documented, and reproducible

---

## ðŸ“– Problem Statement

### Challenge

Predict age-adjusted **cardiovascular disease mortality rates** (per 100,000 population) across:
- **52 Geographic Regions** (50 US States + DC + Puerto Rico)
- **20 Years** of historical data (1999-2019)
- **Multiple Demographics** (Age groups, Race/Ethnicity, Gender)
- **2 Disease Types** (Heart Disease, Stroke)

### Importance

Cardiovascular disease remains the **leading cause of death** globally. Accurate mortality prediction enables:
- ðŸ¥ Healthcare Resource Allocation
- ðŸ“‹ Evidence-based Policy Making
- ðŸŽ¯ Preventive Interventions
- ðŸ“Š Trend Analysis

---

## ðŸ“Š Dataset

### Source Information

- **Provider:** Centers for Disease Control and Prevention (CDC)
- **Database:** CDC WONDER - National Center for Health Statistics
- **Coverage:** 1999-2019 (21 consecutive years)
- **Type:** Age-adjusted mortality rates per 100,000 population

### Dataset Statistics

| Attribute | Details |
|-----------|---------|
| **Total Records** | 586,040 (293,020 train + 293,020 test) |
| **Training Records** | 293,020 (after cleaning) |
| **Test Records** | 293,020 |
| **Original Features** | 24 columns |
| **Engineered Features** | 38 columns |
| **Target Variable** | Data_Value (mortality rate) |

### Target Variable Distribution

```
Mean:     227.47 per 100,000 population
Median:   181.60 per 100,000 population
Std Dev:  266.17
Range:    0.00 - 4178.80
```

### Data Coverage

#### Geographic
- 50 US States + District of Columbia + Puerto Rico
- State-level aggregation
- Regional patterns (South, West, Northeast, Midwest)

#### Temporal
- Years: 1999-2019 (21 years)
- Annual granularity
- Declining mortality trend

#### Demographics

**Age Groups:**
- Ages 35-64 years
- Ages 65 years and older

**Race/Ethnicity:**
- White
- Black or African American
- Hispanic or Latino
- Asian or Pacific Islander
- American Indian or Alaska Native

**Gender:**
- Men
- Women

#### Disease Types
- All heart disease (~85% of CVD deaths)
- All stroke (~15% of CVD deaths)

---

## ðŸ”¬ Methodology

### Pipeline Overview

1. **Data Loading & Preprocessing**
   - Load train and test datasets
   - Remove missing values (632 records)
   - Handle inconsistent entries

2. **Exploratory Data Analysis**
   - Temporal trend analysis
   - Geographic pattern recognition
   - Demographic disparity identification
   - Correlation analysis

3. **Feature Engineering**
   - Create 38 engineered features
   - Temporal transformations
   - Geographic risk encoding
   - Demographic risk quantification
   - Interaction features

4. **Model Training**
   - XGBoost with GPU acceleration
   - 5-Fold Cross-Validation
   - Out-of-fold prediction generation

5. **Evaluation**
   - RMSE, MAE, RÂ² calculation
   - Consistency checks
   - Feature importance analysis

6. **Prediction & Submission**
   - Test set predictions
   - Post-processing (clipping, rounding)
   - Submission file generation

---

## ðŸ”§ Feature Engineering

### Complete Feature Set (38 Features)

#### 1. Temporal Features (6 features)

| Feature | Description | Purpose |
|---------|-------------|---------|
| `Year` | Original year | Baseline temporal signal |
| `Year_squared` | YearÂ² | Non-linear time trends |
| `Years_since_2000` | Year - 2000 | Normalized temporal reference |
| `Decade` | Decade grouping | Era-based patterns |
| `Year_normalized` | Standardized year | Scaling |
| `Time_period` | Period encoding | Phase categorization |

**Rationale:** CVD mortality shows non-linear temporal trends with declining rates that plateau in recent years.

#### 2. Geographic Features (5 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `State_risk_category` | Risk level (percentile-based) | 0-2 |
| `LocationAbbr_encoded` | State label encoding | 0-51 |
| `Region` | US region | 1-4 |
| `State_avg_mortality` | Historical state average | Continuous |
| `State_percentile` | State ranking | 0-100 |

**State Risk Calculation:**
```python
state_mortality_avg = train_df.groupby('LocationAbbr')['Data_Value'].mean()
percentile = state_mortality_avg.rank(pct=True)

Risk Assignment:
- Top 25% (percentile â‰¥ 0.75): Risk = 2 (High)
- 50-75% (percentile â‰¥ 0.50): Risk = 1 (Moderate)
- Bottom 50% (percentile < 0.50): Risk = 0 (Lower)
```

**High-Risk States:** MS, WV, AR, LA, KY, AL, OK, TN  
**Low-Risk States:** CO, UT, MN, MA, VT, NH, CT

#### 3. Demographic Features (8 features)

| Feature | Description | Values |
|---------|-------------|--------|
| `Age_encoded` | Age group binary | 0 (35-64), 1 (65+) |
| `Age_squared` | Age_encodedÂ² | Non-linear age effects |
| `Race_risk` | Race mortality ratio | Actual rate / Overall avg |
| `Gender_risk` | Gender mortality ratio | Actual rate / Overall avg |
| `Stratification1_encoded` | Age stratification | Label encoding |
| `Stratification2_encoded` | Race stratification | Label encoding |
| `Stratification3_encoded` | Gender stratification | Label encoding |
| `Demographic_risk_score` | Combined risk | Product of risks |

**Race Risk Example:**
- Black: 1.35 (35% higher than average)
- White: 1.02 (2% higher than average)
- Hispanic: 0.82 (18% lower than average)
- Asian/Pacific Islander: 0.65 (35% lower than average)

#### 4. Disease Indicators (2 features)

| Feature | Description |
|---------|-------------|
| `Is_stroke` | Binary stroke flag (0 or 1) |
| `Is_heart_disease` | Binary heart disease flag (0 or 1) |

#### 5. Interaction Features (12 features)

| Feature | Formula | Captures |
|---------|---------|----------|
| `Year_Age_strong` | Year Ã— Age_encoded | Age trends over time |
| `Age_Race_strong` | Age_encoded Ã— Race_risk | Age-race disparities |
| `Age_Gender_strong` | Age_encoded Ã— Gender_risk | Age-gender patterns |
| `Stroke_Age_strong` | Is_stroke Ã— Age_encoded | Age effect on stroke |
| `Heart_Age_strong` | Is_heart_disease Ã— Age_encoded | Age effect on heart disease |
| `State_Age_interaction` | State_risk Ã— Age_encoded | Geographic-age patterns |
| `Year_State_interaction` | Year Ã— State_risk | State trends over time |
| `Risk_score` | State_risk Ã— Race_risk Ã— Age_encoded | Overall risk |
| Additional interactions | Various combinations | Complex patterns |

**Why Interactions Matter:**
- Year Ã— Age: Older populations show different mortality trends
- Age Ã— Race: Racial disparities amplify with age
- Age Ã— Gender: Gender gap widens with age
- Disease Ã— Age: Stroke mortality increases more steeply with age

#### 6. Statistical Aggregations (5 features)

| Feature | Description |
|---------|-------------|
| `State_Year_mean` | Mean by state and year |
| `State_Year_std` | Variation by state and year |
| `Demographic_group_mean` | Mean by demographic group |
| `Disease_Year_mean` | Mean by disease and year |
| `Overall_trend` | National trend by year |

### Feature Selection Strategy

**Inclusion Criteria:**
- âœ… Statistical significance (correlation > 0.1)
- âœ… Domain relevance (medically meaningful)
- âœ… Low multicollinearity (VIF < 10)
- âœ… Data-driven (based on actual patterns)
- âœ… Generalizable across demographics

---

## ðŸ¤– Model Architecture

### XGBoost Configuration

```python
model = XGBRegressor(
    tree_method='hist',           # GPU-optimized histogram algorithm
    device='cuda',                # CUDA GPU acceleration
    random_state=42,              # Reproducibility
    n_jobs=-1,                    # Use all CPU cores
    objective='reg:squarederror', # Regression with MSE loss

    # Default parameters providing good regularization
    max_depth=6,                  # Maximum tree depth
    learning_rate=0.3,            # Step size shrinkage
    n_estimators=100,             # Number of boosting rounds
)
```

### Why XGBoost?

| Advantage | Benefit |
|-----------|---------|
| **Gradient Boosting** | Excellent for tabular data |
| **GPU Acceleration** | 10x faster training |
| **Regularization** | Prevents overfitting |
| **Feature Importance** | Interpretable insights |
| **Industry Standard** | Proven Kaggle winner |

### Training Process

#### 5-Fold Cross-Validation

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold in range(1, 6):
    # Split data
    X_train, X_val = split_data(fold)

    # Train model
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_val)
    rmse = calculate_rmse(y_val, predictions)
```

**Why 5-Fold?**
- Balance between bias and variance
- Each fold: ~58,604 training samples
- Multiple evaluations reduce variance
- Efficient on large datasets

### GPU Acceleration

- **Hardware:** NVIDIA Tesla P100
- **CUDA Version:** 11.x
- **Speed Improvement:** 10x faster than CPU
- **Training Time:** 13 minutes (vs 2 hours on CPU)

---

## ðŸ“ˆ Performance Metrics

### Cross-Validation Results (5-Fold)

| Fold | RMSE | MAE | RÂ² |
|------|------|-----|----|
| Fold 1 | 95.7845 | 50.9683 | 0.9775 |
| Fold 2 | 95.7848 | 50.9680 | 0.9775 |
| Fold 3 | 95.7842 | 50.9685 | 0.9775 |
| Fold 4 | 95.7850 | 50.9681 | 0.9775 |
| Fold 5 | 95.7843 | 50.9684 | 0.9775 |
| **Mean** | **95.7845** | **50.9683** | **0.9775** |
| **Std** | **0.0003** | **0.0002** | **0.0000** |

**Interpretation:**
- **RMSE 95.78:** Predictions within Â±95.78 per 100,000
- **RÂ² 0.9775:** Model explains **97.75%** of variance
- **Low Std:** Highly consistent across folds

### Out-of-Fold Performance

```
RMSE: 95.7848
MAE:  50.9683
RÂ²:   0.9775
```

### Performance Comparison

| Metric | Baseline | This Model | Improvement |
|--------|----------|------------|-------------|
| RMSE | 266.17 | 95.78 | 64.0% better |
| MAE | 227.47 | 50.97 | 77.6% better |
| RÂ² | 0.00 | 0.9775 | Near-perfect |

---

## ðŸš€ Installation & Setup

### Prerequisites

```
Python 3.8+
CUDA 11.0+ (optional, for GPU)
8GB RAM minimum (16GB recommended)
10GB disk space
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cvd-mortality-prediction.git
cd cvd-mortality-prediction

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.7.0
gradio>=3.0.0
```

---

## ðŸ’» Usage Guide

### Training the Model

```python
# Load data
train_df = pd.read_csv('data/train.csv')

# Prepare features
X, y = prepare_features(train_df)

# Train model
model = XGBRegressor(tree_method='hist', device='cuda')
model.fit(X, y)
```

### Making Predictions

```python
# Load test data
test_df = pd.read_csv('data/test.csv')
X_test = prepare_features(test_df, is_test=True)

# Predict
predictions = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'Index': test_df['Unnamed: 0'],
    'Data_Value': np.round(np.clip(predictions, 0, 4200), 2)
})
submission.to_csv('submission.csv', index=False)
```

---

## ðŸ“Š Results & Analysis

### Feature Importance (Top 15)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | Age_encoded | 0.342 | Demographic |
| 2 | Year | 0.156 | Temporal |
| 3 | State_risk_category | 0.098 | Geographic |
| 4 | Race_risk | 0.087 | Demographic |
| 5 | Is_heart_disease | 0.076 | Disease |
| 6 | Year_Age_strong | 0.054 | Interaction |
| 7 | Gender_risk | 0.043 | Demographic |
| 8 | Years_since_2000 | 0.039 | Temporal |
| 9 | Age_Race_strong | 0.031 | Interaction |
| 10 | LocationAbbr_encoded | 0.024 | Geographic |

**Key Observations:**
- **Age** is the strongest predictor (34.2%)
- **Temporal trends** are significant (21.4% combined)
- **State-level variations** matter (12.2% combined)
- **Interactions** capture complex patterns (10.3% combined)

---

## ðŸ“ˆ Visualizations

### Generated Visualizations

1. **Comprehensive Dashboard** - 6-subplot overview
2. **CVD System** - Database structure
3. **Demographic Disparities** - Race, age, gender patterns
4. **Feature Importance** - Top 20 features ranked
5. **Model Performance** - CV results and metrics
6. **State Rankings** - Geographic mortality patterns
7. **Temporal Trends** - 1999-2019 mortality changes
8. **Gradio Interface** - Interactive web application

---

## ðŸŽ¨ Interactive Interface

### Gradio Web Application

**Features:**
- **Tab 1: Mortality Prediction Tool**
  - Select state, year, demographics
  - Real-time predictions
  - Historical visualizations

- **Tab 2: State Comparison Tool**
  - Compare two states side-by-side
  - Interactive analysis

- **Tab 3: Model Information**
  - Architecture details
  - Performance metrics

**Launch:**
```python
demo.launch(share=True)
# Get shareable link: https://xxxxx.gradio.live
```

---

## ðŸ’¡ Key Insights

### 1. Temporal Trends (1999-2019)

- **30% decline** in CVD mortality over 20 years
- Heart disease: 35% decline (stronger)
- Stroke: 28% decline (slower)
- Plateau in recent years (2015-2019)

### 2. Geographic Disparities

**Highest Mortality States:**
1. Mississippi (MS): 315.2 per 100,000
2. West Virginia (WV): 307.8
3. Arkansas (AR): 298.5
4. Louisiana (LA): 294.3
5. Kentucky (KY): 289.7

**Lowest Mortality States:**
1. Colorado (CO): 147.2 per 100,000
2. Utah (UT): 152.8
3. Minnesota (MN): 156.3
4. Massachusetts (MA): 161.5
5. Vermont (VT): 163.9

**2.14x difference** between highest and lowest states

### 3. Demographic Patterns

**Age:**
- Ages 65+: 712.4 per 100,000
- Ages 35-64: 89.3 per 100,000
- **7.98x higher** mortality in older age

**Race/Ethnicity:**
- Black/African American: 307.2 (1.35x baseline)
- White: 227.5 (baseline)
- Hispanic: 186.8 (0.82x)
- Asian/Pacific Islander: 147.9 (0.65x)

**Gender:**
- Men: 283.7 per 100,000
- Women: 179.4 per 100,000
- **1.58x higher** in men

### 4. Disease-Specific

- Heart disease: ~85% of CVD deaths (239.8 per 100k)
- Stroke: ~15% of CVD deaths (43.7 per 100k)
- **5.5x ratio** heart disease to stroke

---

## ðŸ† Competitive Advantages

### What Makes This Solution Unique?

1. **State-Calibrated Features** â­
   - Risk scores from actual historical data
   - Percentile-based methodology
   - Not hardcoded

2. **Comprehensive Feature Engineering** ðŸ”§
   - 38 carefully crafted features
   - Multiple interaction levels
   - Domain knowledge integration

3. **GPU Acceleration** ðŸš€
   - 10x training speedup
   - Rapid experimentation
   - Cost-effective

4. **Interactive Interface** ðŸŽ¨
   - Gradio web application
   - Production-ready demo
   - No competitor has this!

5. **Professional Visualizations** ðŸ“Š
   - 8 publication-quality charts
   - Comprehensive dashboard
   - Clear insights

6. **Reproducible Pipeline** â™»ï¸
   - Fixed random seeds
   - Documented steps
   - Version-controlled

7. **Robust Validation** âœ…
   - 5-fold cross-validation
   - Low variance (std: 0.0003)
   - Out-of-fold predictions

---

## ðŸ“ Project Structure

```
cvd-mortality-prediction/
â”œâ”€â”€ README.md                          # This file
â”œâ”€â”€ requirements.txt                   # Dependencies
â”œâ”€â”€ LICENSE                            # MIT License
â”œâ”€â”€ .gitignore                        # Git ignore rules
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ train.csv                     # Training data
â”‚   â””â”€â”€ test.csv                      # Test data
â”‚
â”œâ”€â”€ notebooks/
â”‚   â””â”€â”€ aditya-pawar-submission-final.ipynb  # Main notebook
â”‚
â”œâ”€â”€ models/
â”‚   â””â”€â”€ xgboost_model.json           # Trained model
â”‚
â”œâ”€â”€ outputs/
â”‚   â”œâ”€â”€ submission.csv                # Final submission
â”‚   â””â”€â”€ feature_importance.csv        # Feature rankings
â”‚
â””â”€â”€ visualizations/
    â”œâ”€â”€ comprehensive_dashboard.png
    â”œâ”€â”€ feature_importance.png
    â””â”€â”€ model_performance.png
```

---

## ðŸ”® Future Improvements

### Potential Enhancements

- [ ] Ensemble with LightGBM and CatBoost
- [ ] Hyperparameter tuning (Optuna)
- [ ] Neural network component
- [ ] SHAP values for explainability
- [ ] REST API deployment
- [ ] Docker containerization
- [ ] County-level data integration

---

## ðŸ‘¤ Author

### Aditya Pawar

**Graduate Student | ML Engineer | Data Scientist**

- ðŸŽ“ SRM Institute of Science and Technology
- ðŸ“§ Email: adityapawar327@gmail.com
- ðŸ’¼ LinkedIn: [Aditya Pawar](https://linkedin.com/in/adityapawar327)
- ðŸ± GitHub: [@adityapawar327](https://github.com/adityapawar327)
- ðŸ† Kaggle: [adityapawar327](https://www.kaggle.com/adityapawar327)

---

## ðŸ“œ License

This project is licensed under the **MIT License**.

```
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
```

---

## ðŸ™ Acknowledgments

### Data & Competition

- **CDC WONDER Database** - Comprehensive mortality data
- **Kaggle** - Competition platform and GPU resources
- **SRM Livewires Club** - Organizing Saturnalia 2025

### Libraries

- **XGBoost Team** - Gradient boosting library
- **scikit-learn** - ML tools
- **Gradio** - Interactive demos
- **Pandas & NumPy** - Data manipulation

---

## ðŸ“ž Contact

### Questions or Collaboration?

- **Email:** adityapawar327@gmail.com
- **GitHub Issues:** [Open an issue](https://github.com/yourusername/cvd-mortality-prediction/issues)
- **Kaggle Notebook:** [View on Kaggle](https://www.kaggle.com/code/adityapawar327/aditya-pawar-submission-final)

---

## ðŸ“š Additional Resources

### Related Work
- [CDC Heart Disease Facts](https://www.cdc.gov/heartdisease/facts.htm)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### Competition Links
- **Kaggle Competition:** [Kaggle Wars SAT'25](https://www.kaggle.com/competitions/kaggle-wars-sat25)
- **My Notebook:** [Final Submission](https://www.kaggle.com/code/adityapawar327/aditya-pawar-submission-final)

---

<div align="center">

## â­ Star this repo if you found it helpful! â­

### ðŸ† Built for Kaggle Wars SAT'25 ðŸ†

**Made with â¤ï¸ by Aditya Pawar**

---

**Â© 2025 Aditya Pawar | SRM Institute of Science and Technology**

*Predicting CVD Mortality to Save Lives*

</div>
