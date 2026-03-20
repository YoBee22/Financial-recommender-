"""
Fixed Feature Engineering for CE Interview Data
Addresses critical clustering issues:
1. Handles inf values from zero-income households
2. Fixes unreliable savings_rate with clipping and flags
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CEFeatureEngineerFixed:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.fmli_data = None
        self.memi_data = None
        self.features_df = None
        
    def load_data(self, quarters=['242', '243', '244']):
        """Load FMLI and MEMI data for specified quarters"""
        print("Loading CE Interview data")
        
        # Load FMLI (Family Characteristics)
        fmli_files = []
        for quarter in quarters:
            file_path = self.data_dir / f'fmli{quarter}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['quarter'] = quarter
                fmli_files.append(df)
                print(f"✓ Loaded fmli{quarter}.csv: {len(df):,} records")
        
        if fmli_files:
            self.fmli_data = pd.concat(fmli_files, ignore_index=True)
            print(f"✓ Combined FMLI data: {len(self.fmli_data):,} records")
        
        # Load MEMI (Member Characteristics)
        memi_files = []
        for quarter in quarters:
            file_path = self.data_dir / f'memi{quarter}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['quarter'] = quarter
                memi_files.append(df)
                print(f"✓ Loaded memi{quarter}.csv: {len(df):,} records")
        
        if memi_files:
            self.memi_data = pd.concat(memi_files, ignore_index=True)
            print(f"✓ Combined MEMI data: {len(self.memi_data):,} records")
    
    def create_demographic_features(self):
        """Create demographic-based features"""
        print("Creating demographic features")
        
        if self.fmli_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        features = self.fmli_data[['NEWID', 'quarter']].copy()
        
        # Age-based features
        features['age_ref'] = pd.to_numeric(self.fmli_data['AGE_REF'], errors='coerce')
        features['age_group'] = pd.cut(features['age_ref'], 
                                     bins=[0, 34, 49, 64, 100], 
                                     labels=['18-34', '35-49', '50-64', '65+'])
        features['age_squared'] = features['age_ref'] ** 2
        features['is_senior'] = (features['age_ref'] >= 65).astype(int)
        features['is_young_adult'] = (features['age_ref'] <= 34).astype(int)
        
        # Family composition features
        features['family_size'] = pd.to_numeric(self.fmli_data['FAM_SIZE'], errors='coerce')
        features['family_size_squared'] = features['family_size'] ** 2
        features['has_children'] = (features['family_size'] > 2).astype(int)
        features['is_single'] = (features['family_size'] == 1).astype(int)
        features['is_couple'] = (features['family_size'] == 2).astype(int)
        
        # Geographic features
        region_map = {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'}
        features['region'] = self.fmli_data['REGION'].map(region_map)
        features['is_urban'] = (self.fmli_data['BLS_URBN'] == 1).astype(int)
        
        # Housing tenure features
        tenure_map = {1: 'owned_mortgage', 2: 'owned_free', 3: 'rented', 
                    4: 'student', 5: 'other'}
        features['housing_tenure'] = self.fmli_data['CUTENURE'].map(tenure_map)
        features['is_homeowner'] = self.fmli_data['CUTENURE'].isin([1, 2]).astype(int)
        
        # Education features
        educ_map = {1: 'less_than_hs', 2: 'hs_graduate', 3: 'some_college', 
                   4: 'bachelor_degree', 5: 'graduate_degree'}
        features['education_level'] = self.fmli_data['EDUC_REF'].map(educ_map)
        
        # Marital status features
        marital_map = {1: 'married', 2: 'widowed', 3: 'divorced', 
                      4: 'separated', 5: 'never_married'}
        features['marital_status'] = self.fmli_data['MARITAL1'].map(marital_map)
        features['is_married'] = (self.fmli_data['MARITAL1'] == 1).astype(int)
        
        # Race/Ethnicity features
        race_map = {1: 'white', 2: 'black', 3: 'american_indian', 
                   4: 'asian', 5: 'hawaiian', 6: 'multiracial'}
        features['race'] = self.fmli_data['RACE2'].map(race_map)
        features['is_minority'] = (self.fmli_data['RACE2'] != 1).astype(int)
        
        print("✓ Demographic features created")
        return features
    
    def create_income_features(self, features_df):
        """Create income-related features"""
        print("Creating income features")
        
        # Income variables
        features_df['total_income'] = pd.to_numeric(self.fmli_data['FINCBTAX'], errors='coerce')
        features_df['log_income'] = np.log1p(features_df['total_income'])
        
        # Income rank and percentiles
        features_df['income_rank'] = pd.to_numeric(self.fmli_data['INC_RANK'], errors='coerce')
        features_df['income_quintile'] = pd.qcut(features_df['income_rank'], 
                                               q=5, labels=[1, 2, 3, 4, 5])
        features_df['income_decile'] = pd.qcut(features_df['income_rank'], 
                                            q=10, labels=range(1, 11))
        
        # Income source features
        features_df['wage_income'] = pd.to_numeric(self.fmli_data['FSALARYX'], errors='coerce')
        features_df['self_employment_income'] = pd.to_numeric(self.fmli_data['FRRDEDX'], errors='coerce')
        features_df['retirement_income'] = pd.to_numeric(self.fmli_data['FGOVRETX'], errors='coerce')
        features_df['investment_income'] = pd.to_numeric(self.fmli_data['FINDRETX'], errors='coerce')
        
        # Income ratios and diversity
        features_df['wage_income_ratio'] = np.where(
            features_df['total_income'] > 0,
            features_df['wage_income'] / features_df['total_income'],
            0
        )
        features_df['passive_income_ratio'] = np.where(
            features_df['total_income'] > 0,
            (features_df['retirement_income'] + features_df['investment_income']) / features_df['total_income'],
            0
        )
        
        # Count income sources
        income_sources = ['wage_income', 'self_employment_income', 'retirement_income', 'investment_income']
        features_df['income_diversity'] = (features_df[income_sources] > 0).sum(axis=1)
        
        # Per capita income
        features_df['per_capita_income'] = np.where(
            features_df['family_size'] > 0,
            features_df['total_income'] / features_df['family_size'],
            features_df['total_income']
        )
        
        # Zero income flag
        features_df['zero_income_flag'] = (features_df['total_income'] == 0).astype(int)
        
        print("✓ Income features created")
        return features_df
    
    def create_expenditure_features(self, features_df):
        """Create expenditure category features and spending ratios"""
        print("Creating expenditure features")
        
        # Total expenditure
        features_df['total_expenditure'] = pd.to_numeric(self.fmli_data['TOTEXPPQ'], errors='coerce')
        features_df['log_expenditure'] = np.log1p(features_df['total_expenditure'])
        
        # Major expenditure categories
        features_df['food_expenditure'] = pd.to_numeric(self.fmli_data['FDHOMEPQ'], errors='coerce') + \
                                        pd.to_numeric(self.fmli_data['FDAWAYPQ'], errors='coerce')
        features_df['housing_expenditure'] = pd.to_numeric(self.fmli_data['HOUSPQ'], errors='coerce')
        
        # Transportation - check available columns
        transport_cols = [col for col in self.fmli_data.columns if 'TR' in col and 'PQ' in col]
        if transport_cols:
            features_df['transportation_expenditure'] = pd.to_numeric(
                self.fmli_data[transport_cols[0]], errors='coerce'
            )
        else:
            features_df['transportation_expenditure'] = 0
            
        # Healthcare - check available columns  
        health_cols = [col for col in self.fmli_data.columns if 'HL' in col and 'PQ' in col]
        if health_cols:
            features_df['healthcare_expenditure'] = pd.to_numeric(
                self.fmli_data[health_cols[0]], errors='coerce'
            )
        else:
            features_df['healthcare_expenditure'] = 0
            
        # Entertainment - check available columns
        ent_cols = [col for col in self.fmli_data.columns if 'ENT' in col and 'PQ' in col]
        if ent_cols:
            features_df['entertainment_expenditure'] = pd.to_numeric(
                self.fmli_data[ent_cols[0]], errors='coerce'
            )
        else:
            features_df['entertainment_expenditure'] = 0
            
        # Apparel - check available columns
        app_cols = [col for col in self.fmli_data.columns if 'APP' in col and 'PQ' in col]
        if app_cols:
            features_df['apparel_expenditure'] = pd.to_numeric(
                self.fmli_data[app_cols[0]], errors='coerce'
            )
        else:
            features_df['apparel_expenditure'] = 0
        
        # Expenditure ratios
        features_df['food_ratio'] = np.where(
            features_df['total_expenditure'] > 0,
            features_df['food_expenditure'] / features_df['total_expenditure'],
            0
        )
        features_df['housing_ratio'] = np.where(
            features_df['total_expenditure'] > 0,
            features_df['housing_expenditure'] / features_df['total_expenditure'],
            0
        )
        features_df['transportation_ratio'] = np.where(
            features_df['total_expenditure'] > 0,
            features_df['transportation_expenditure'] / features_df['total_expenditure'],
            0
        )
        features_df['healthcare_ratio'] = np.where(
            features_df['total_expenditure'] > 0,
            features_df['healthcare_expenditure'] / features_df['total_expenditure'],
            0
        )
        features_df['entertainment_ratio'] = np.where(
            features_df['total_expenditure'] > 0,
            features_df['entertainment_expenditure'] / features_df['total_expenditure'],
            0
        )
        
        # Essential vs discretionary spending
        features_df['essential_spending_ratio'] = (
            features_df['food_ratio'] + features_df['housing_ratio'] + features_df['healthcare_ratio']
        )
        features_df['discretionary_spending_ratio'] = (
            features_df['entertainment_ratio'] + features_df['apparel_expenditure'] / features_df['total_expenditure'].fillna(1)
        )
        
        # Per capita expenditure
        features_df['per_capita_expenditure'] = np.where(
            features_df['family_size'] > 0,
            features_df['total_expenditure'] / features_df['family_size'],
            features_df['total_expenditure']
        )
        
        print("✓ Expenditure features created")
        return features_df
    
    def create_financial_health_features(self, features_df):
        """Create financial health indicator features with fixes for clustering"""
        print("Creating financial health features")
        
        # Savings features
        features_df['savings_amount'] = features_df['total_income'] - features_df['total_expenditure']
        features_df['is_positive_savings'] = (features_df['savings_amount'] > 0).astype(int)
        
        # FIXED: Handle inf values in expenditure_to_income_ratio
        features_df['expenditure_to_income_ratio'] = np.where(
            features_df['total_income'] > 0,
            features_df['total_expenditure'] / features_df['total_income'],
            np.nan
        )
        
        # FIXED: Replace inf values with median and create zero income flag
        features_df['expenditure_to_income_ratio'] = features_df['expenditure_to_income_ratio'].replace([np.inf, -np.inf], np.nan)
        features_df['expenditure_to_income_ratio'] = features_df['expenditure_to_income_ratio'].fillna(
            features_df['expenditure_to_income_ratio'].median()
        )
        
        # FIXED: Handle savings_rate with clipping to sensible range
        features_df['savings_rate'] = np.where(
            features_df['total_income'] > 0,
            features_df['savings_amount'] / features_df['total_income'],
            np.nan
        )
        
        # Clip savings_rate to sensible range [-2, 1]
        features_df['savings_rate'] = features_df['savings_rate'].clip(lower=-2, upper=1)
        features_df['savings_rate'] = features_df['savings_rate'].fillna(0)  # Zero income households get 0
        
        # Financial stress indicators
        features_df['high_spending_ratio'] = (features_df['expenditure_to_income_ratio'] > 0.9).astype(int)
        features_df['low_savings_rate'] = (features_df['savings_rate'] < 0.1).astype(int)
        
        # Spending diversity (number of categories with significant spending)
        categories = ['food_expenditure', 'housing_expenditure', 'transportation_expenditure', 
                     'healthcare_expenditure', 'entertainment_expenditure']
        threshold = features_df['total_expenditure'] * 0.05  # 5% of total spending
        features_df['spending_diversity'] = (
            (features_df[categories].values > threshold.values.reshape(-1, 1)).sum(axis=1)
        )
        
        print("✓ Financial health features created (with clustering fixes)")
        return features_df
    
    def create_interaction_features(self, features_df):
        """Create interaction features between demographics and spending patterns"""
        print("Creating interaction features")
        
        # Age x Income interactions
        features_df['age_x_income'] = features_df['age_ref'] * features_df['income_rank']
        features_df['age_x_log_income'] = features_df['age_ref'] * features_df['log_income']
        
        # Family size x Income interactions
        features_df['family_size_x_income'] = features_df['family_size'] * features_df['income_rank']
        
        # Age x Expenditure interactions
        features_df['age_x_expenditure'] = features_df['age_ref'] * features_df['log_expenditure']
        
        # Handle inf values in income_x_expenditure_ratio
        features_df['income_x_expenditure_ratio'] = features_df['income_rank'] * features_df['expenditure_to_income_ratio']
        features_df['income_x_expenditure_ratio'] = features_df['income_x_expenditure_ratio'].replace([np.inf, -np.inf], np.nan)
        features_df['income_x_expenditure_ratio'] = features_df['income_x_expenditure_ratio'].fillna(
            features_df['income_x_expenditure_ratio'].median()
        )
        
        # Region x Income interactions (one-hot encoded regions)
        region_dummies = pd.get_dummies(features_df['region'], prefix='region')
        for region_col in region_dummies.columns:
            features_df[f'{region_col}_x_income'] = region_dummies[region_col] * features_df['income_rank']
        
        print("✓ Interaction features created (with clustering fixes)")
        return features_df
    
    def create_target_variables(self, features_df):
        """Create target variables for product prediction models"""
        print("Creating target variables")
        
        # High spender classification (top 25%)
        features_df['high_spender'] = (features_df['total_expenditure'] > 
                                    features_df['total_expenditure'].quantile(0.75)).astype(int)
        
        # High income classification (top 25%)
        features_df['high_income'] = (features_df['total_income'] > 
                                    features_df['total_income'].quantile(0.75)).astype(int)
        
        # Financial health tier
        def classify_financial_health(row):
            if pd.isna(row['savings_rate']):
                return 'Unknown'
            elif row['savings_rate'] >= 0.2:
                return 'Excellent'
            elif row['savings_rate'] >= 0.1:
                return 'Good'
            elif row['savings_rate'] >= 0:
                return 'Fair'
            else:
                return 'Poor'
        
        features_df['financial_health_tier'] = features_df.apply(classify_financial_health, axis=1)
        
        # Primary spending category
        categories = ['food_expenditure', 'housing_expenditure', 'transportation_expenditure', 
                     'healthcare_expenditure', 'entertainment_expenditure']
        features_df['primary_spending_category'] = features_df[categories].idxmax(axis=1)
        
        # Product recommendation targets
        features_df['needs_savings_product'] = ((features_df['savings_rate'] < 0.1) & 
                                             (features_df['total_income'] > 0)).astype(int)
        features_df['needs_investment_product'] = ((features_df['income_rank'] > 0.7) & 
                                                 (features_df['savings_rate'] > 0.1)).astype(int)
        features_df['needs_insurance_product'] = (features_df['healthcare_ratio'] > 0.15).astype(int)
        features_df['needs_loan_product'] = ((features_df['expenditure_to_income_ratio'] > 1.0) & 
                                           (features_df['age_ref'] < 65)).astype(int)
        
        print("✓ Target variables created")
        return features_df
    
    def handle_missing_values(self, df):
        """Handle missing values appropriately"""
        print("Handling missing values")
        
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna('Unknown')
        
        print("✓ Missing values handled")
        return df
    
    def validate_clustering_readiness(self, df):
        """Validate that data is ready for clustering algorithms"""
        print("Validating clustering readiness")
        
        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        print(f"  Infinite values: {inf_count}")
        
        # Check for very large values
        large_values = (df.select_dtypes(include=[np.number]) > 1e6).sum().sum()
        print(f"  Values > 1M: {large_values}")
        
        # Check savings_rate range
        if 'savings_rate' in df.columns:
            min_rate = df['savings_rate'].min()
            max_rate = df['savings_rate'].max()
            print(f"  Savings rate range: [{min_rate:.2f}, {max_rate:.2f}]")
        
        # Check zero income households
        if 'zero_income_flag' in df.columns:
            zero_income_count = df['zero_income_flag'].sum()
            print(f"  Zero income households: {zero_income_count}")
        
        print("✓ Clustering validation complete")
        return df
    
    def engineer_all_features(self):
        """Main method to create all features with clustering fixes"""
        print("Starting fixed feature engineering pipeline")
        
        # Create demographic features
        features = self.create_demographic_features()
        
        # Create income features
        features = self.create_income_features(features)
        
        # Create expenditure features
        features = self.create_expenditure_features(features)
        
        # Create financial health features (with fixes)
        features = self.create_financial_health_features(features)
        
        # Create interaction features (with fixes)
        features = self.create_interaction_features(features)
        
        # Create target variables
        features = self.create_target_variables(features)
        
        # Handle missing values
        features = self.handle_missing_values(features)
        
        # Validate clustering readiness
        features = self.validate_clustering_readiness(features)
        
        self.features_df = features
        print("✓ Fixed feature engineering complete!")
        return features
    
    def save_features(self, output_path):
        """Save engineered features to CSV"""
        if self.features_df is not None:
            self.features_df.to_csv(output_path, index=False)
            print(f"✓ Features saved to {output_path}")
        else:
            print("No features to save. Run engineer_all_features() first.")
    
    def get_feature_summary(self):
        """Get summary of engineered features"""
        if self.features_df is None:
            return "No features available. Run engineer_all_features() first."
        
        summary = {
            'total_records': len(self.features_df),
            'total_features': len(self.features_df.columns),
            'numeric_features': len(self.features_df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(self.features_df.select_dtypes(include=['object']).columns),
            'missing_values': self.features_df.isnull().sum().sum(),
            'infinite_values': np.isinf(self.features_df.select_dtypes(include=[np.number])).sum().sum()
        }
        
        return summary

# Main execution function
def main():
    # Create output directory
    output_dir = Path('./feature-engineering-output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature engineer
    data_dir = '../data/intrvw24'
    engineer = CEFeatureEngineerFixed(data_dir)
    
    # Load data
    engineer.load_data(['242', '243', '244'])
    
    # Engineer features with fixes
    features = engineer.engineer_all_features()
    
    # Save features
    engineer.save_features(output_dir / 'engineered_features_fixed.csv')
    
    # Print summary
    summary = engineer.get_feature_summary()
    print("\nFixed Feature Engineering Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:,}")
    
    # Display sample of key features for clustering
    print("\nSample of clustering-ready features:")
    key_cols = ['NEWID', 'total_income', 'total_expenditure', 'savings_rate', 
                'expenditure_to_income_ratio', 'zero_income_flag', 'is_positive_savings']
    print(features[key_cols].head())
    
    return features

if __name__ == "__main__":
    features = main()
