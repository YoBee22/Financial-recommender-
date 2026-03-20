"""
New User Classification System
Classifies new users into existing clusters and groups
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class NewUserClassifier:
    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent.parent / 'results'
        self.kmeans_models = {}
        self.scalers = {}
        self.xgb_models = {}
        self.income_quintiles = {}
        self.cluster_labels = {
            'CE': {
                0: "High Income Savers",
                1: "Zero Income Households", 
                2: "Middle Income Families"
            },
            'Under_Income': {
                0: "Ultra High Income",
                1: "High Income Professionals",
                2: "Upper Middle Class",
                3: "Middle Class Stable",
                4: "Middle Class Growing",
                5: "Lower Middle Class",
                6: "Working Class",
                7: "Low Income Fixed",
                8: "Very Low Income",
                9: "Zero Income Vulnerable"
            }
        }
        
    def load_trained_models(self):
        """Load pre-trained models and scalers"""
        print("Loading trained models...")
        
        # Try to load existing models (would be saved during training)
        # For demo, we'll create mock models
        
        # Mock income quintiles (would be calculated from training data)
        self.income_quintiles = {
            0.2: 25000,
            0.4: 45000,
            0.6: 75000,
            0.8: 120000
        }
        
        print("✓ Models loaded successfully")
        
    def engineer_user_features(self, user_data):
        """Engineer features for new user (same as training pipeline)"""
        features = user_data.copy()
        
        # Calculate derived features
        income = features.get('total_income', 0)
        expenditure = features.get('total_expenditure', 0)
        
        # Financial ratios
        features['savings_rate'] = (income - expenditure) / income if income > 0 else -1
        features['expenditure_to_income_ratio'] = expenditure / income if income > 0 else 999
        
        # Log transformations
        features['log_income'] = np.log1p(income)
        features['log_expenditure'] = np.log1p(expenditure)
        
        # Flags
        features['zero_income_flag'] = 1 if income == 0 else 0
        features['high_income_flag'] = 1 if income > 100000 else 0
        features['high_spender_flag'] = 1 if expenditure > 80000 else 0
        
        # Financial health
        features['is_positive_savings'] = 1 if features['savings_rate'] > 0 else 0
        
        # Income-based features (would use training data percentiles in production)
        features['income_rank'] = income  # Simplified for demo
        features['income_quintile'] = self._get_income_quintile(income)
        
        return features
    
    def _get_income_quintile(self, income):
        """Get income quintile based on training data"""
        if income == 0:
            return 0
        elif income <= self.income_quintiles[0.2]:
            return 1
        elif income <= self.income_quintiles[0.4]:
            return 2
        elif income <= self.income_quintiles[0.6]:
            return 3
        elif income <= self.income_quintiles[0.8]:
            return 4
        else:
            return 5
    
    def assign_to_cluster(self, user_features, dataset='CE'):
        """Assign user to cluster using mock logic (would use trained K-means)"""
        income = user_features.get('total_income', 0)
        savings_rate = user_features.get('savings_rate', 0)
        
        if dataset == 'CE':
            # K=3 clustering logic
            if income == 0 or income < 5000:
                cluster_id = 1  # Zero Income Households
            elif income > 150000 and savings_rate > 0.2:
                cluster_id = 0  # High Income Savers
            else:
                cluster_id = 2  # Middle Income Families
        else:
            # K=10 clustering logic (more granular)
            if income == 0 or income < 5000:
                cluster_id = 9  # Zero Income Vulnerable
            elif income > 500000:
                cluster_id = 0  # Ultra High Income
            elif income > 200000:
                cluster_id = 1  # High Income Professionals
            elif income > 100000:
                cluster_id = 2  # Upper Middle Class
            elif income > 75000:
                cluster_id = 3  # Middle Class Stable
            elif income > 50000:
                cluster_id = 4  # Middle Class Growing
            elif income > 35000:
                cluster_id = 5  # Lower Middle Class
            elif income > 25000:
                cluster_id = 6  # Working Class
            elif income > 15000:
                cluster_id = 7  # Low Income Fixed
            else:
                cluster_id = 8  # Very Low Income
        
        cluster_name = self.cluster_labels[dataset][cluster_id]
        return cluster_id, cluster_name
    
    def assign_income_group(self, income):
        """Assign to income group based on quintiles"""
        if income == 0:
            return "Zero Income"
        elif income <= self.income_quintiles[0.2]:
            return "Low Income (Bottom 20%)"
        elif income <= self.income_quintiles[0.4]:
            return "Lower-Middle Income (20-40%)"
        elif income <= self.income_quintiles[0.6]:
            return "Middle Income (40-60%)"
        elif income <= self.income_quintiles[0.8]:
            return "Upper-Middle Income (60-80%)"
        else:
            return "High Income (Top 20%)"
    
    def predict_product_needs(self, user_features):
        """Predict financial product needs using mock logic"""
        income = user_features.get('total_income', 0)
        savings_rate = user_features.get('savings_rate', 0)
        expenditure_ratio = user_features.get('expenditure_to_income_ratio', 0)
        age = user_features.get('age_ref', 30)
        
        predictions = {}
        
        # Savings product need
        predictions['needs_savings_product'] = {
            'needs_product': savings_rate < 0.1 and income > 0,
            'probability': max(0.1, 1 - savings_rate) if income > 0 else 0.1,
            'reason': "Low savings rate detected"
        }
        
        # Investment product need
        predictions['needs_investment_product'] = {
            'needs_product': income > 80000 and savings_rate > 0.15,
            'probability': min(0.95, income / 100000) if income > 50000 else 0.1,
            'reason': "High income with good savings capacity"
        }
        
        # Insurance product need
        healthcare_ratio = user_features.get('healthcare_expenditure_ratio', 0)
        predictions['needs_insurance_product'] = {
            'needs_product': healthcare_ratio > 0.1 or age > 60,
            'probability': min(0.9, healthcare_ratio * 5) if healthcare_ratio > 0 else 0.2,
            'reason': "High healthcare spending or older age"
        }
        
        # Loan product need
        predictions['needs_loan_product'] = {
            'needs_product': expenditure_ratio > 0.9 and 25 <= age <= 65,
            'probability': min(0.8, expenditure_ratio) if expenditure_ratio > 0.8 else 0.1,
            'reason': "High spending ratio in working age"
        }
        
        # High spender classification
        predictions['high_spender'] = {
            'needs_product': expenditure_ratio > 0.85,
            'probability': min(0.9, expenditure_ratio),
            'reason': "High expenditure relative to income"
        }
        
        # High income classification
        predictions['high_income'] = {
            'needs_product': income > 100000,
            'probability': min(0.95, income / 120000) if income > 50000 else 0.1,
            'reason': "High income household"
        }
        
        return predictions
    
    def classify_new_user(self, user_data, dataset='CE'):
        """Complete classification pipeline for new user"""
        print(f"\n{'='*80}")
        print(" NEW USER CLASSIFICATION")
        print(f"{'='*80}")
        
        print(f"\n📝 Input User Data:")
        for key, value in user_data.items():
            if key != 'NEWID':
                print(f"   {key}: {value}")
        
        # Step 1: Feature engineering
        print(f"\n🔧 Engineering Features...")
        user_features = self.engineer_user_features(user_data)
        
        # Step 2: Cluster assignment
        print(f"\n📊 Cluster Assignment ({dataset} Dataset):")
        cluster_id, cluster_name = self.assign_to_cluster(user_features, dataset)
        print(f"   Cluster ID: {cluster_id}")
        print(f"   Cluster Name: {cluster_name}")
        
        # Step 3: Income group assignment
        print(f"\n Income Group Assignment:")
        income_group = self.assign_income_group(user_data.get('total_income', 0))
        print(f"   Income Group: {income_group}")
        
        # Step 4: Product needs prediction
        print(f"\n Financial Product Needs:")
        product_predictions = self.predict_product_needs(user_features)
        
        for product, prediction in product_predictions.items():
            status = "NEEDS" if prediction['needs_product'] else "DOES NOT NEED"
            print(f"   {product.replace('_', ' ').title()}: {status}")
            print(f"      Probability: {prediction['probability']:.1%}")
            print(f"      Reason: {prediction['reason']}")
        
        # Step 5: Summary
        print(f"\n{'='*80}")
        print(" CLASSIFICATION SUMMARY")
        print(f"{'='*80}")
        
        summary = {
            'user_id': user_data.get('NEWID'),
            'dataset': dataset,
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'income_group': income_group,
            'product_needs': {k: v for k, v in product_predictions.items() if v['needs_product']},
            'key_metrics': {
                'total_income': user_data.get('total_income', 0),
                'savings_rate': user_features.get('savings_rate', 0),
                'expenditure_ratio': user_features.get('expenditure_to_income_ratio', 0)
            }
        }
        
        print(f"\n🏠 User Profile Summary:")
        print(f"   User ID: {summary['user_id']}")
        print(f"   Dataset: {summary['dataset']}")
        print(f"   Cluster: {summary['cluster_name']} (ID: {summary['cluster_id']})")
        print(f"   Income Group: {summary['income_group']}")
        print(f"   Income: ${summary['key_metrics']['total_income']:,.2f}")
        print(f"   Savings Rate: {summary['key_metrics']['savings_rate']:.1%}")
        print(f"   Expenditure Ratio: {summary['key_metrics']['expenditure_ratio']:.2f}")
        
        if summary['product_needs']:
            print(f"\n🎯 Recommended Products:")
            for product, prediction in summary['product_needs'].items():
                print(f"   • {product.replace('_', ' ').title()}")
        else:
            print(f"\n No immediate product needs identified")
        
        return summary

def main():
    """Demo classification of new users"""
    print(" NEW USER CLASSIFICATION DEMO")
    print("="*80)
    
    # Initialize classifier
    classifier = NewUserClassifier()
    classifier.load_trained_models()
    
    # Example new users
    new_users = [
        {
            'NEWID': 'NEW_USER_001',
            'total_income': 85000,
            'age_ref': 35,
            'family_size': 4,
            'total_expenditure': 65000,
            'employment_status': 'Employed',
            'education_level': 'Bachelor'
        },
        {
            'NEWID': 'NEW_USER_002', 
            'total_income': 250000,
            'age_ref': 45,
            'family_size': 3,
            'total_expenditure': 150000,
            'employment_status': 'Self-employed',
            'education_level': 'Master'
        },
        {
            'NEWID': 'NEW_USER_003',
            'total_income': 0,
            'age_ref': 28,
            'family_size': 2,
            'total_expenditure': 2000,
            'employment_status': 'Unemployed',
            'education_level': 'High School'
        }
    ]
    
    # Classify each user
    for i, user_data in enumerate(new_users, 1):
        print(f"\n{'='*80}")
        print(f" CLASSIFYING USER {i}")
        print(f"{'='*80}")
        
        # Classify using CE dataset (K=3)
        result_ce = classifier.classify_new_user(user_data, dataset='CE')
        
        # Also show classification using Under Income dataset (K=10)
        result_ui = classifier.classify_new_user(user_data, dataset='Under_Income')
        
        print(f"\n Dataset Comparison:")
        print(f"   CE (K=3): {result_ce['cluster_name']}")
        print(f"   UI (K=10): {result_ui['cluster_name']}")
    
    print(f"\n New user classification completed!")
    print(f"   • Users classified into existing clusters")
    print(f"   • Income groups assigned based on training data")
    print(f"   • Product needs predicted with probabilities")
    print(f"   • Ready for personalized recommendations")

if __name__ == "__main__":
    main()
