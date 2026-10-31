# Campus Placement Prediction & Career Outcome Analytics Platform
## Project Plan & Architecture Document

---

## 1. Problem Statement

Campus placements are a critical milestone for students and institutions. This platform aims to:
- **Predict** whether a student will be placed or not (binary classification)
- **Predict** the expected salary/CTC for placed students (regression)
- **Analyze** career outcome trends across departments, streams, and academic performance bands
- **Provide** actionable recommendations to students and placement officers

---

## 2. Objectives

| # | Objective | Type |
|---|-----------|------|
| 1 | Placement likelihood prediction | Binary Classification |
| 2 | Expected CTC prediction | Regression |
| 3 | Department-wise placement analytics | EDA / Visualization |
| 4 | Student profile risk scoring | Scoring Model |
| 5 | Interactive career analytics dashboard | Streamlit App |

---

## 3. Dataset Description

### Source
Synthetic dataset generated to simulate real placement data with statistically realistic distributions. Inspired by the standard Kaggle MBA placement dataset.

### Features

#### Academic Performance
| Feature | Type | Description |
|---------|------|-------------|
| `ssc_percentage` | Float | 10th grade marks (%) |
| `hsc_percentage` | Float | 12th grade marks (%) |
| `degree_percentage` | Float | Graduation marks (%) |
| `mba_percentage` | Float | MBA/PG marks (%) |
| `etest_percentage` | Float | Employability test score (%) |

#### Demographics & Background
| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Categorical | Male / Female |
| `ssc_board` | Categorical | Central / Others |
| `hsc_board` | Categorical | Central / Others |
| `hsc_subject` | Categorical | Commerce / Science / Arts |
| `degree_type` | Categorical | Comm&Mgmt / Sci&Tech / Others |
| `work_experience` | Categorical | Yes / No |
| `mba_specialization` | Categorical | Mkt&HR / Mkt&Fin |

#### Target Variables
| Feature | Type | Description |
|---------|------|-------------|
| `placement_status` | Binary | Placed (1) / Not Placed (0) |
| `salary` | Float | CTC in LPA (only for placed students) |

### Dataset Size
- **1,000 student records** (synthetic, realistic distributions)
- ~65% placement rate (industry-realistic)

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Campus Placement Analytics Platform             │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
   │  Data Layer   │  │  Model Layer  │  │  App Layer    │
   │               │  │               │  │               │
   │ • Raw CSV     │  │ • Classifiers │  │ • Streamlit   │
   │ • Preprocessor│  │ • Regressors  │  │   Dashboard   │
   │ • Feature Eng │  │ • Ensembles   │  │ • Predictions │
   └───────────────┘  └───────────────┘  │ • Analytics   │
           │                  │          └───────────────┘
           └──────────────────┘
                     │
            ┌────────────────┐
            │  Saved Models  │
            │  (models/*.pkl)│
            └────────────────┘
```

---

## 5. ML Pipeline

```
Raw Data
   │
   ▼
[1. Data Generation]
   generate_data.py → placement_data.csv
   │
   ▼
[2. Exploratory Data Analysis]
   notebooks/EDA.ipynb
   - Distribution analysis
   - Correlation heatmaps
   - Placement rate by segment
   │
   ▼
[3. Preprocessing]
   src/data_preprocessing.py
   - Handle missing values (salary for unplaced)
   - Encode categoricals (Label + One-Hot)
   - Scale numerics (StandardScaler)
   - Train/test split (80/20)
   │
   ▼
[4. Feature Engineering]
   src/feature_engineering.py
   - Academic composite score
   - Consistency index (std of academic scores)
   - Performance tier (Low/Mid/High)
   - Experience-specialization interaction
   │
   ▼
[5. Model Training]
   train.py → models/
   ┌─────────────────────────────────────┐
   │  Classification (Placement Status)  │
   │  • Logistic Regression (baseline)  │
   │  • Random Forest Classifier        │
   │  • XGBoost Classifier              │
   │  • Gradient Boosting Classifier    │
   │  • SVM Classifier                  │
   └─────────────────────────────────────┘
   ┌─────────────────────────────────────┐
   │  Regression (Salary Prediction)    │
   │  • Linear Regression (baseline)    │
   │  • Random Forest Regressor         │
   │  • XGBoost Regressor               │
   │  • Ridge Regression                │
   └─────────────────────────────────────┘
   │
   ▼
[6. Evaluation]
   src/evaluation.py
   - Classification: Accuracy, Precision, Recall, F1, ROC-AUC
   - Regression: MAE, RMSE, R²
   - Confusion matrices, ROC curves, Feature importance
   │
   ▼
[7. Deployment]
   app.py (Streamlit)
   - Student prediction interface
   - Batch upload predictions
   - Analytics dashboards
   - Model comparison page
```

---

## 6. Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.9+ |
| ML Framework | scikit-learn, XGBoost |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Model Persistence | joblib |
| Notebook | Jupyter |

---

## 7. Project File Structure

```
Campus Placement Prediction.../
│
├── Plan/
│   └── PROJECT_PLAN.md           ← This document
│
└── Code/
    ├── data/
    │   └── placement_data.csv    ← Generated dataset
    │
    ├── models/                   ← Saved trained models (.pkl)
    │
    ├── notebooks/
    │   └── EDA.ipynb             ← Exploratory Data Analysis
    │
    ├── src/
    │   ├── __init__.py
    │   ├── data_generator.py     ← Synthetic data creation
    │   ├── data_preprocessing.py ← Cleaning, encoding, scaling
    │   ├── feature_engineering.py← Derived features
    │   ├── models.py             ← Model definitions
    │   └── evaluation.py         ← Metrics & plots
    │
    ├── app.py                    ← Streamlit dashboard
    ├── train.py                  ← Full training pipeline
    └── requirements.txt          ← Dependencies
```

---

## 8. Key ML Models & Rationale

### Classification Models
| Model | Rationale |
|-------|-----------|
| **Logistic Regression** | Interpretable baseline; good for understanding feature importance linearly |
| **Random Forest** | Handles non-linearity, robust to outliers, built-in feature importance |
| **XGBoost** | State-of-the-art gradient boosting; best overall performance expected |
| **Gradient Boosting** | Compare with XGBoost for consistency |
| **SVM** | Effective in high-dimensional spaces |

### Regression Models (Salary)
| Model | Rationale |
|-------|-----------|
| **Linear Regression** | Interpretable baseline |
| **Ridge Regression** | Handles multicollinearity |
| **Random Forest Regressor** | Captures non-linear salary relationships |
| **XGBoost Regressor** | Best performance expected |

---

## 9. Expected Results

### Classification Performance Targets
| Metric | Target |
|--------|--------|
| Accuracy | ≥ 85% |
| F1 Score | ≥ 0.83 |
| ROC-AUC | ≥ 0.88 |

### Regression Performance Targets
| Metric | Target |
|--------|--------|
| R² Score | ≥ 0.75 |
| RMSE | ≤ 1.5 LPA |

---

## 10. Dashboard Features

### Page 1: Home / Overview
- Project summary and KPIs
- Quick statistics from the dataset

### Page 2: Predict Placement
- Input student profile via form
- Real-time placement probability
- Expected salary if placed
- Risk score and recommendations

### Page 3: Analytics Dashboard
- Placement rate by gender, stream, specialization
- Salary distribution plots
- Academic performance vs placement correlation
- Department-wise trends

### Page 4: Model Performance
- Accuracy comparison across all models
- Confusion matrices
- ROC-AUC curves
- Feature importance charts

### Page 5: Batch Prediction
- Upload CSV with student data
- Download predictions as CSV

---

## 11. Insights the Platform Reveals

1. **Academic impact**: Which academic metric (10th/12th/degree/MBA) most influences placement
2. **Gender disparity**: Placement rates and salary gaps across genders
3. **Specialization value**: Mkt&Fin vs Mkt&HR salary differences
4. **Experience premium**: Impact of prior work experience on placement and salary
5. **Board effect**: Central vs State board influence
6. **At-risk identification**: Students with low placement probability flagged early

---

## 12. Future Enhancements

- Real-time data pipeline integration with college ERP
- NLP-based resume scoring module
- LinkedIn profile analysis for career tracking
- Time-series analysis of placement trends across years
- Recommendation engine for skill gap bridging
