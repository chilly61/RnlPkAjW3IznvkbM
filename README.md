# Apziva Project A - Customer Happiness

## Project Overview

This project focuses on predicting customer satisfaction (happiness) based on survey data from ACME Corporation. The goal is to build a classification model that identifies satisfied vs. dissatisfied customers, enabling businesses to take proactive measures to improve customer experience.

The project implements a sophisticated ensemble learning approach with:
- **Stacking Ensemble**: Combines Logistic Regression, Random Forest, and SVM
- **RFE Analysis**: Recursive Feature Elimination for feature importance
- **Hyperopt Optimization**: Automated hyperparameter tuning
- **Threshold Optimization**: Balances accuracy and recall for business needs

This project was developed as part of the Apziva program, demonstrating advanced machine learning techniques for customer satisfaction prediction.

> **Note:** Please read `RnlPkAjW3IznvkbM.pdf` for comprehensive results and analysis.
📊 **Visualization**: See [ProjectB_Visualization.pdf](RnlPkAjW3IznvkbM.pdf)

## Core Features

### 1. Stacking Ensemble Classifier

The model combines three diverse base learners with a meta-learner:

**Base Models:**
- **Logistic Regression (LR)**: Linear classifier for probabilistic outputs
- **Support Vector Machine (SVM)**: RBF kernel for non-linear boundaries
- **Random Forest (RF)**: Ensemble of decision trees

**Meta-Learner:**
- **Random Forest**: Combines base model predictions for final classification

**Architecture:**
```
StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(...)),
        ('svm', SVC(...)),
        ('rf', RandomForestClassifier(...))
    ],
    final_estimator=RandomForestClassifier(...),
    cv=5,
    passthrough=False
)
```

### 2. Recursive Feature Elimination (RFE)

- **Feature Ranking**: Evaluates feature importance through iterative elimination
- **Model-Based Selection**: Uses the trained model to rank features
- **Visualization**: Bar plots and correlation analysis
- **Top Feature Identification**: Automatically identifies most predictive features

### 3. Hyperopt Hyperparameter Optimization

- **Bayesian Optimization**: Uses Tree of Parzen Estimators (TPE) algorithm
- **Search Space**: Defines ranges for all model hyperparameters
- **Balanced Objective**: Optimizes for both accuracy and class recall
- **Constraint Handling**: Prevents models from being overly biased toward one class

**Optimized Parameters:**
- LR: `C`, `max_iter`
- SVM: `C`, `gamma`
- RF: `n_estimators`, `max_depth`, `min_samples_split`
- Meta-RF: `n_estimators`, `max_depth`

### 4. Threshold Optimization Strategies

Multiple weighting strategies for threshold selection:

| Strategy | Weight (Acc) | Weight (Rec0) | Weight (Rec1) |
|----------|--------------|---------------|---------------|
| Balanced | 0.4 | 0.3 | 0.3 |
| Accuracy Focused | 0.6 | 0.2 | 0.2 |
| Class 0 Focused | 0.4 | 0.4 | 0.2 |
| Class 1 Focused | 0.3 | 0.2 | 0.5 |

## Technical Architecture

### Dataset Summary

| Metric | Value |
|--------|-------|
| Target Variable | `Y` (customer satisfaction: 0/1) |
| Features | X1, X2, X3, X4, X5, X6 |
| Data Source | ACME Happiness Survey 2020 |

### Data Preprocessing Pipeline

1. **Feature Selection**: SelectKBest with f_classif (k=10 or min available)
2. **Standardization**: StandardScaler for all features
3. **Train/Test Split**: 70/30 with stratification (random_state=124)

### Model Configuration

**Base Models:**
```python
# Logistic Regression
LogisticRegression(C=0.0690578407, max_iter=1100, random_state=124)

# SVM
SVC(C=16.895294109, kernel='rbf', gamma=0.3508198145, probability=True)

# Random Forest
RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_split=3)
```

**Meta-Learner:**
```python
RandomForestClassifier(n_estimators=100, max_depth=14)
```

### Cross-Validation Strategy

- **5-Fold Stratified Cross-Validation**
- **Deterministic Split**: Fixed random state (124) for reproducibility
- **Probability Estimation**: cross_val_predict for CV probabilities

### Feature Selection

Based on RFE and analysis, the optimal feature combination is:
- **X1, X4, X5** (as identified in the final model)

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/chilly61/RnlPkAjW3IznvkbM
cd RnlPkAjW3IznvkbM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn hyperopt joblib
```

### Running the Pipeline

```bash
# Main Stacking Model
python "Project A - Customer Happiness/Main.py"
```

The Main.py contains three main sections:

1. **Stacking Accuracy Optimization** (Lines 39-373)
   - Trains the stacking ensemble
   - Performs threshold search
   - Evaluates multiple weighting strategies

2. **RFE Feature Analysis** (Lines 380-536)
   - Analyzes feature importance
   - Ranks features by relevance
   - Generates visualization plots

3. **Hyperopt Optimization** (Lines 546+)
   - Tunes hyperparameters
   - Finds optimal model configuration
   - Generates optimization progress plots

### Output Files

The code generates various visualizations:
- Model comparison charts
- Threshold vs. accuracy/recall curves
- ROC curves (CV and Test)
- RFE feature rankings
- Hyperopt optimization progress

## Project Structure

```
RnlPkAjW3IznvkbM/
├── README.md                                    # This file
├── RnlPkAjW3IznvkbM.pdf                         # Full analysis report
└── Project A - Customer Happiness/
    └── Main.py                                   # Complete ML pipeline
```

### Code Sections in Main.py

| Section | Lines | Description |
|---------|-------|-------------|
| Stacking Optimization | 39-373 | Main model training with CV and threshold tuning |
| RFE Analysis | 380-536 | Feature importance analysis |
| Hyperopt Optimization | 546+ | Hyperparameter tuning |

## Performance Evaluation

### Final Model Performance

| Metric | Value |
|--------|-------|
| **CV Accuracy** | 0.659 |
| **CV AUC** | ~0.70+ |
| **Recall (Class 0)** | 0.875 |
| **Recall (Class 1)** | Variable based on threshold |

### Model Comparison

The stacking ensemble outperforms individual models:

| Model | Best CV Accuracy |
|-------|------------------|
| Logistic Regression | ~0.60 |
| SVM | ~0.62 |
| Random Forest | ~0.64 |
| **Stacking Ensemble** | **0.659** |

### Key Findings

1. **Stacking Improves Performance**: Combining multiple models yields better results than individual classifiers
2. **Optimal Features**: X1, X4, X5 provide the best predictive power
3. **Threshold Matters**: Different business objectives require different classification thresholds
4. **Balanced Trade-off**: The model balances recall across both classes

### Weighting Strategy Results

| Strategy | Threshold | Test Accuracy | Test Recall 0 | Test Recall 1 |
|----------|-----------|---------------|---------------|---------------|
| Balanced | ~0.3-0.4 | ~0.65 | ~0.85 | ~0.45 |
| Class 0 Focused | ~0.2-0.3 | ~0.62 | ~0.90 | ~0.35 |

## Challenges and Solutions

### Challenge 1: Small Dataset Size

**Problem**: Limited sample size makes model training challenging.

**Solution:**
- Used cross-validation for robust evaluation
- Applied regularization to prevent overfitting
- Selected minimal effective feature set (X1, X4, X5)

### Challenge 2: Feature Selection

**Problem**: Determining which features are most predictive.

**Solution:**
- Used SelectKBest for initial feature screening
- Applied RFE for detailed feature ranking
- Identified optimal combination: [X1, X4, X5]

### Challenge 3: Class Imbalance

**Problem**: Uneven distribution of satisfied/dissatisfied customers.

**Solution:**
- Multiple threshold optimization strategies
- Weighted scoring approaches
- Balanced accuracy as evaluation metric

### Challenge 4: Hyperparameter Tuning

**Problem**: Large search space makes manual tuning inefficient.

**Solution:**
- Hyperopt with TPE algorithm
- 50-100 evaluations for optimal results
- Balanced objective function

### Challenge 5: Reproducibility

**Problem**: Randomized algorithms may produce different results.

**Solution:**
- Fixed random seeds (SEED=124)
- Deterministic train/test splits
- Single-threaded operations (n_jobs=1)

## Tech Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Hyperparameter Optimization**: hyperopt
- **Visualization**: matplotlib
- **Parallel Processing**: joblib

## Key Takeaways

1. **Ensemble Methods Work**: Stacking combines the strengths of different algorithms
2. **Feature Quality Over Quantity**: Three well-chosen features outperform six poorly selected ones
3. **Threshold Selection is Business Decision**: The "best" threshold depends on business priorities
4. **Automation Saves Time**: Hyperopt efficiently explores the hyperparameter space

## Contributors

Thanks to the Apziva team for their support and guidance.

## License

This project is for internal use only.
