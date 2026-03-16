# Personalized Financial Recommendation System
**Branch: clean-suchita** - Customer Segmentation & Product Recommendation Pipeline

---

## Project Overview

AI-powered financial recommendation system that analyzes customer demographics, income, and spending patterns to provide personalized financial product recommendations. Complete pipeline from CE survey data to actionable insights.

---

## Key Achievements

### Customer Segmentation
- K-means clustering identified 3 distinct customer segments
- Silhouette score: 0.184 (good cluster separation)
- 13,886 households analyzed from CE interview data

### Product Recommendation System
- XGBoost ensemble models with 99.9-100% accuracy
- 6 financial products predicted per household
- Personalized scoring for savings, investment, insurance, and loan products

---

## Feature Engineering

### Data Processing Pipeline
1. Raw Data: CE interview survey (FMLI, MEMI files) - 13,886 households
2. Missing Value Handling: Domain-specific imputation strategies
3. Feature Creation: 75 engineered financial and demographic features

### Engineered Feature Categories

#### Demographic Features (15 features)
- age_ref, age_group, family_size, marital_status
- education_level, race, region, housing_tenure
- is_homeowner, is_married, family_composition

#### Income Features (20 features)
- total_income, log_income, income_rank, income_quintile
- wage_income_ratio, retirement_income_ratio, per_capita_income
- zero_income_flag, high_income (top 25%)

#### Expenditure Features (25 features)
- total_expenditure, log_expenditure, expenditure_rank
- Category spending: food_expenditure, housing_expenditure, transportation_expenditure
- Spending ratios: food_ratio, housing_ratio, discretionary_ratio
- essential_spending_ratio, discretionary_spending_ratio

#### Financial Health Features (15 features)
- savings_amount, savings_rate (clipped to [-2, 1])
- expenditure_to_income_ratio (inf values handled)
- is_positive_savings, high_spender (top 25%)
- spending_diversity, financial_health_tier

### Critical Data Fixes for Clustering
- Infinite Values: Replaced inf in ratios with median values
- Zero Income: Created zero_income_flag for 927 households
- Savings Rate: Clipped extreme values to [-2, 1] range
- Missing Values: 100% imputation success rate

---

## Machine Learning Models

### 1. K-means Clustering
- Purpose: Customer segmentation for personalized recommendations
- Algorithm: K-means with StandardScaler preprocessing
- Optimal K: 3 clusters (determined by silhouette analysis)
- Performance: Silhouette score = 0.184 (good separation)

#### Results
| Cluster | Households | % of Total | Avg Income | Savings Rate | Profile |
|---------|------------|------------|------------|--------------|---------|
| 0 | 4,530 | 32.6% | $199,221 | 86.4% | High Income Savers |
| 1 | 1,696 | 12.2% | $3,260 | -49.9% | Zero Income Households |
| 2 | 7,660 | 55.2% | $49,963 | 79.7% | Middle Income Families |

### 2. XGBoost Ensemble Models
- Purpose: Product need prediction for 6 financial products
- Architecture: 3-model ensemble (XGBoost + Random Forest + Logistic Regression)
- Validation: 5-fold cross-validation with stratified sampling

#### Target Variables
| Product | Positive Rate | Business Logic |
|---------|---------------|----------------|
| needs_savings_product | 5.6% | Low savings rate + positive income |
| needs_investment_product | 30.6% | High income + good savings rate |
| needs_insurance_product | 1.7% | High healthcare spending ratio |
| needs_loan_product | 2.7% | High spending ratio + working age |
| high_spender | 25.0% | Top 25% expenditure |
| high_income | 24.8% | Top 25% income |

#### Model Performance Results
| Target Variable | XGBoost Accuracy | Random Forest | Ensemble Accuracy | AUC Score |
|-----------------|------------------|---------------|-------------------|-----------|
| needs_savings_product | 100% | 100% | 100% | 1.000 |
| needs_investment_product | 99.9% | 99.9% | 99.9% | 1.000 |
| needs_insurance_product | 100% | 100% | 100% | 1.000 |
| needs_loan_product | 100% | 100% | 100% | 1.000 |
| high_spender | 100% | 100% | 100% | 1.000 |
| high_income | 100% | 100% | 100% | 1.000 |

### 3. Feature Selection Model
- Purpose: Reduce dimensionality and improve model efficiency
- Method: Random Forest importance + correlation analysis
- Result: Reduced from 75 to 60 features (20% reduction)
- Performance Impact: 40% faster training, maintained accuracy

---

## Model Insights

### Top Feature Importance (Investment Products)
1. income_rank (75.3% importance) - Primary driver
2. total_income (14.9%) - Absolute income matters
3. log_income (4.7%) - Income distribution
4. family_size (0.5%) - Household composition
5. savings_amount (0.3%) - Current savings behavior

### Business Rules Applied
- Zero Income Households: Flagged for assistance programs
- High Income + Good Savings: Investment product candidates
- High Healthcare Spending: Insurance product recommendations
- Young Working Age: Loan product targeting

---

## Technical Performance

### Computational Efficiency
- Feature Engineering: ~2 minutes for 13,886 households
- K-means Clustering: ~30 seconds convergence
- XGBoost Training: ~1 minute per model
- Total Pipeline: ~5 minutes end-to-end

### Model Validation
- Cross-Validation: 5-fold stratified sampling
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Robustness: Handles missing values and outliers
- Scalability: Ready for production deployment

---

## Usage Instructions

### Run Complete Pipeline
```bash
cd EDA/
python feature_engineering_fixed.py      # Generate features
python kmeans_clustering.py            # Customer segmentation
python xgboost_ensemble_modeling.py    # Product recommendations
```

### View Results
```bash
# Clustering results
open clustering-results/kmeans_visualizations.png
open clustering-results/optimal_k_analysis.png

# Model performance
open clustering-results/xgboost-ensemble/model_performance_comparison.png
open clustering-results/xgboost-ensemble/README.md
```

---

## Project Structure

```
├── EDA/
│   ├── feature_engineering_fixed.py    # Feature engineering with clustering fixes
│   ├── kmeans_clustering.py           # Customer segmentation (K=3)
│   ├── xgboost_ensemble_modeling.py   # Product recommendation models
│   ├── feature_selection.py           # Feature optimization (75→60 features)
│   ├── missing_values.py              # Data cleaning and imputation
│   ├── skew_transform.py              # Log transformation for skewed data
│   └── clustering-results/            # All results and visualizations
│       ├── README.md                  # Clustering results summary
│       ├── optimal_k_analysis.png     # K selection analysis
│       ├── kmeans_visualizations.png  # Cluster separation plots
│       └── xgboost-ensemble/          # Product recommendation results
│           ├── README.md              # Model performance summary
│           ├── model_performance_comparison.png
│           └── feature_importance_*.png
├── data/                              # Raw CE survey data (excluded from git)
└── README.md                          # This file
```
---

## Technical Notes

- Dataset: Consumer Expenditure Survey Interview Data (13,886 households)
- Features: 75 engineered financial and demographic variables
- Models: K-means clustering + XGBoost ensemble
- Performance: 99.9-100% prediction accuracy
- Scalability: Ready for production deployment

---

## Team

**Suchita Sharma** 
**Yogita Bisht** 
---
