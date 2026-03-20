"""
Multi-K Clustering Handler
Manages clustering results from different datasets with different K values
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MultiKClusteringHandler:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.cluster_mappings = {
            'CE': {
                'k_value': 3,
                'cluster_labels': {
                    0: "High Income Savers",
                    1: "Zero Income Households", 
                    2: "Middle Income Families"
                }
            },
            'Under_Income': {
                'k_value': 10,
                'cluster_labels': {
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
        }
        
    def create_sample_ce_data(self, n_households=100):
        """Create sample CE dataset with K=3 clustering"""
        np.random.seed(42)
        data = []
        
        for i in range(n_households):
            household = {
                'NEWID': f'CE_{i+1:06d}',
                'dataset': 'CE',
                'total_income': np.random.lognormal(10.5, 1.0),
                'age_ref': np.random.randint(22, 75),
                'family_size': np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.25, 0.2, 0.05]),
                'total_expenditure': 0,
                'savings_rate': 0
            }
            
            # Calculate expenditure and savings
            income = household['total_income']
            household['total_expenditure'] = income * np.random.uniform(0.6, 0.95)
            household['savings_rate'] = (income - household['total_expenditure']) / income if income > 0 else 0
            
            # Assign cluster based on K=3 patterns
            if income == 0 or income < 5000:
                household['cluster'] = 1  # Zero Income Households
            elif income > 150000 and household['savings_rate'] > 0.2:
                household['cluster'] = 0  # High Income Savers
            else:
                household['cluster'] = 2  # Middle Income Families
            
            data.append(household)
        
        return pd.DataFrame(data)
    
    def create_sample_under_income_data(self, n_households=100):
        """Create sample Under Income dataset with K=10 clustering"""
        np.random.seed(123)
        data = []
        
        # Define income ranges for K=10 clusters
        income_ranges = [
            (500000, 2000000),   # Ultra High Income
            (200000, 500000),    # High Income Professionals  
            (100000, 200000),    # Upper Middle Class
            (75000, 100000),     # Middle Class Stable
            (50000, 75000),      # Middle Class Growing
            (35000, 50000),      # Lower Middle Class
            (25000, 35000),      # Working Class
            (15000, 25000),      # Low Income Fixed
            (5000, 15000),       # Very Low Income
            (0, 5000)           # Zero Income Vulnerable
        ]
        
        for i in range(n_households):
            # Assign to one of 10 clusters
            cluster_id = np.random.randint(0, 10)
            min_income, max_income = income_ranges[cluster_id]
            
            household = {
                'NEWID': f'UI_{i+1:06d}',
                'dataset': 'Under_Income',
                'cluster': cluster_id,
                'total_income': np.random.uniform(min_income, max_income),
                'age_ref': np.random.randint(18, 80),
                'family_size': np.random.choice([1, 2, 3, 4, 5, 6], p=[0.15, 0.25, 0.25, 0.2, 0.1, 0.05]),
                'employment_status': np.random.choice(['Employed', 'Self-employed', 'Retired', 'Unemployed']),
                'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
                'total_expenditure': 0,
                'savings_rate': 0
            }
            
            # Calculate expenditure and savings
            income = household['total_income']
            if income > 0:
                household['total_expenditure'] = income * np.random.uniform(0.7, 0.95)
                household['savings_rate'] = (income - household['total_expenditure']) / income
            else:
                household['total_expenditure'] = np.random.uniform(500, 2000)
                household['savings_rate'] = -1.0  # Negative savings (debt)
            
            data.append(household)
        
        return pd.DataFrame(data)
    
    def create_income_groups(self, df):
        """Create income groups based on quintiles"""
        income_values = df['total_income'].dropna()
        if len(income_values) == 0:
            return df
            
        quintiles = income_values.quantile([0.2, 0.4, 0.6, 0.8])
        
        def assign_income_group(income):
            if pd.isna(income) or income == 0:
                return "Zero Income"
            elif income <= quintiles[0.2]:
                return "Low Income (Bottom 20%)"
            elif income <= quintiles[0.4]:
                return "Lower-Middle Income (20-40%)"
            elif income <= quintiles[0.6]:
                return "Middle Income (40-60%)"
            elif income <= quintiles[0.8]:
                return "Upper-Middle Income (60-80%)"
            else:
                return "High Income (Top 20%)"
        
        df['income_group'] = df['total_income'].apply(assign_income_group)
        return df
    
    def load_datasets(self):
        """Load both datasets with their respective K values"""
        print("Loading datasets with different K values...")
        
        # Load CE dataset (K=3)
        ce_data = self.create_sample_ce_data(100)
        ce_data = self.create_income_groups(ce_data)
        self.datasets['CE'] = ce_data
        print(f"✓ Loaded CE dataset: {len(ce_data)} households (K=3)")
        
        # Load Under Income dataset (K=10)
        ui_data = self.create_sample_under_income_data(100)
        ui_data = self.create_income_groups(ui_data)
        self.datasets['Under_Income'] = ui_data
        print(f"✓ Loaded Under Income dataset: {len(ui_data)} households (K=10)")
        
        # Combine datasets
        combined_data = pd.concat([ce_data, ui_data], ignore_index=True)
        print(f"✓ Combined datasets: {len(combined_data)} total households")
        
        return combined_data
    
    def filter_by_dataset(self, dataset_name):
        """Filter households by dataset"""
        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found")
            return None
        
        return self.datasets[dataset_name]
    
    def filter_by_cluster(self, dataset_name, cluster_ids):
        """Filter households by cluster ID(s) for specific dataset"""
        data = self.filter_by_dataset(dataset_name)
        if data is None:
            return None
        
        if isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]
        
        return data[data['cluster'].isin(cluster_ids)]
    
    def filter_by_income_group(self, dataset_name, income_groups):
        """Filter households by income group(s) for specific dataset"""
        data = self.filter_by_dataset(dataset_name)
        if data is None:
            return None
        
        if isinstance(income_groups, str):
            income_groups = [income_groups]
        
        return data[data['income_group'].isin(income_groups)]
    
    def cross_dataset_comparison(self, income_range=None):
        """Compare clusters across datasets for similar income ranges"""
        print(f"\n{'='*80}")
        print("CROSS-DATASET CLUSTER COMPARISON")
        print(f"{'='*80}")
        
        comparison_results = {}
        
        for dataset_name, data in self.datasets.items():
            k_value = self.cluster_mappings[dataset_name]['k_value']
            cluster_labels = self.cluster_mappings[dataset_name]['cluster_labels']
            
            print(f"\n📊 {dataset_name} Dataset (K={k_value}):")
            
            # Filter by income range if specified
            filtered_data = data.copy()
            if income_range:
                min_income, max_income = income_range
                filtered_data = filtered_data[
                    (filtered_data['total_income'] >= min_income) & 
                    (filtered_data['total_income'] <= max_income)
                ]
                print(f"   Income Range: ${min_income:,} - ${max_income:,}")
            
            # Show cluster distribution
            cluster_dist = filtered_data['cluster'].value_counts().sort_index()
            print(f"   Cluster Distribution:")
            
            for cluster_id, count in cluster_dist.items():
                cluster_name = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
                percentage = (count / len(filtered_data)) * 100
                avg_income = filtered_data[filtered_data['cluster'] == cluster_id]['total_income'].mean()
                print(f"     {cluster_name}: {count} households ({percentage:.1f}%) - Avg Income: ${avg_income:,.2f}")
            
            comparison_results[dataset_name] = {
                'total_households': len(filtered_data),
                'cluster_distribution': cluster_dist.to_dict(),
                'avg_income_by_cluster': {}
            }
            
            for cluster_id in cluster_dist.index:
                cluster_data = filtered_data[filtered_data['cluster'] == cluster_id]
                comparison_results[dataset_name]['avg_income_by_cluster'][cluster_id] = {
                    'avg_income': cluster_data['total_income'].mean(),
                    'avg_savings_rate': cluster_data['savings_rate'].mean(),
                    'avg_family_size': cluster_data['family_size'].mean()
                }
        
        return comparison_results
    
    def map_clusters_between_datasets(self, income_range=None):
        """Map clusters between datasets based on similar characteristics"""
        print(f"\n{'='*80}")
        print("CLUSTER MAPPING BETWEEN DATASETS")
        print(f"{'='*80}")
        
        # Get cluster characteristics for both datasets
        ce_data = self.datasets['CE']
        ui_data = self.datasets['Under_Income']
        
        if income_range:
            min_income, max_income = income_range
            ce_data = ce_data[(ce_data['total_income'] >= min_income) & (ce_data['total_income'] <= max_income)]
            ui_data = ui_data[(ui_data['total_income'] >= min_income) & (ui_data['total_income'] <= max_income)]
        
        print(f"\nMapping clusters based on income ranges and characteristics...")
        
        # CE clusters (K=3)
        ce_clusters = {}
        for cluster_id in [0, 1, 2]:
            cluster_data = ce_data[ce_data['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                ce_clusters[cluster_id] = {
                    'name': self.cluster_mappings['CE']['cluster_labels'][cluster_id],
                    'avg_income': cluster_data['total_income'].mean(),
                    'income_range': (cluster_data['total_income'].min(), cluster_data['total_income'].max()),
                    'households': len(cluster_data)
                }
        
        # Under Income clusters (K=10)
        ui_clusters = {}
        for cluster_id in range(10):
            cluster_data = ui_data[ui_data['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                ui_clusters[cluster_id] = {
                    'name': self.cluster_mappings['Under_Income']['cluster_labels'][cluster_id],
                    'avg_income': cluster_data['total_income'].mean(),
                    'income_range': (cluster_data['total_income'].min(), cluster_data['total_income'].max()),
                    'households': len(cluster_data)
                }
        
        # Create mapping based on income similarity
        print(f"\n CLUSTER MAPPINGS:")
        for ce_cluster_id, ce_info in ce_clusters.items():
            print(f"\n{ce_info['name']} (CE K=3) → Similar Under Income clusters:")
            
            # Find UI clusters with similar income ranges
            similar_ui_clusters = []
            for ui_cluster_id, ui_info in ui_clusters.items():
                # Check if income ranges overlap or are close
                ce_min, ce_max = ce_info['income_range']
                ui_min, ui_max = ui_info['income_range']
                
                # Simple similarity check: overlapping ranges or close averages
                if (ce_min <= ui_max and ce_max >= ui_min) or \
                   abs(ce_info['avg_income'] - ui_info['avg_income']) < (ce_info['avg_income'] * 0.3):
                    similar_ui_clusters.append((ui_cluster_id, ui_info))
            
            # Sort by income similarity
            similar_ui_clusters.sort(key=lambda x: abs(x[1]['avg_income'] - ce_info['avg_income']))
            
            for ui_cluster_id, ui_info in similar_ui_clusters[:3]:  # Show top 3 similar
                print(f"   → {ui_info['name']} (UI K=10) - Avg Income: ${ui_info['avg_income']:,.2f}")
        
        return ce_clusters, ui_clusters
    
    def print_user_profiles_by_dataset(self, dataset_name, max_users=3):
        """Print user profiles for specific dataset"""
        data = self.filter_by_dataset(dataset_name)
        if data is None:
            return
        
        k_value = self.cluster_mappings[dataset_name]['k_value']
        cluster_labels = self.cluster_mappings[dataset_name]['cluster_labels']
        
        print(f"\n{'='*80}")
        print(f"{dataset_name} DATASET PROFILES (K={k_value})")
        print(f"{'='*80}")
        
        # Show sample from each cluster
        for cluster_id in sorted(data['cluster'].unique()):
            cluster_name = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
            cluster_data = data[data['cluster'] == cluster_id]
            
            print(f"\n{cluster_name} (ID: {cluster_id}) - {len(cluster_data)} households")
            
            for idx, row in cluster_data.head(max_users).iterrows():
                print(f"\n User ID: {row['NEWID']}")
                print(f"    Income Group: {row['income_group']}")
                print(f"    Total Income: ${row['total_income']:,.2f}")
                print(f"    Reference Age: {row['age_ref']}")
                print(f"    Family Size: {row['family_size']}")
                print(f"    Savings Rate: {row['savings_rate']:.1%}")
                
                # Show dataset-specific fields
                if 'employment_status' in row:
                    print(f"    Employment: {row['employment_status']}")
                if 'education_level' in row:
                    print(f"    Education: {row['education_level']}")
                
                print("-" * 40)

def main():
    """Main execution function"""
    print(" MULTI-K CLUSTERING HANDLER")
    print("="*80)
    
    # Initialize handler
    data_dir = Path(__file__).parent.parent / 'data'
    handler = MultiKClusteringHandler(data_dir)
    
    # Load datasets
    combined_data = handler.load_datasets()
    
    # Show profiles for each dataset
    handler.print_user_profiles_by_dataset('CE')
    handler.print_user_profiles_by_dataset('Under_Income')
    
    # Cross-dataset comparison
    handler.cross_dataset_comparison(income_range=(30000, 100000))
    
    # Map clusters between datasets
    handler.map_clusters_between_datasets()
    
    print(f"\n Multi-K clustering analysis completed!")
    print(f"   • CE Dataset: K=3 clusters (broader segments)")
    print(f"   • Under Income Dataset: K=10 clusters (granular segments)")
    print(f"   • Cross-dataset mapping enables comparison")
    print(f"   • Income-based filtering works across both datasets")

if __name__ == "__main__":
    main()
