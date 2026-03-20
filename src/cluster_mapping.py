"""
Cluster Mapping: K=10 to K=3
Maps Under Income dataset (10 clusters) to CE dataset (3 broad categories)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ClusterMapper:
    def __init__(self):
        # Mapping from K=10 (Under Income) to K=3 (CE) clusters
        self.k10_to_k3_mapping = {
            # High Income Groups (CE Cluster 0)
            0: 0,  # Ultra High Income → High Income Savers
            1: 0,  # High Income Professionals → High Income Savers
            2: 0,  # Upper Middle Class → High Income Savers
            
            # Middle Income Groups (CE Cluster 2)
            3: 2,  # Middle Class Stable → Middle Income Families
            4: 2,  # Middle Class Growing → Middle Income Families
            5: 2,  # Lower Middle Class → Middle Income Families
            6: 2,  # Working Class → Middle Income Families
            
            # Zero Income Groups (CE Cluster 1)
            7: 1,  # Low Income Fixed → Zero Income Households
            8: 1,  # Very Low Income → Zero Income Households
            9: 1,  # Zero Income Vulnerable → Zero Income Households
        }
        
        # Cluster names for both datasets
        self.k10_names = {
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
        
        self.k3_names = {
            0: "High Income Savers",
            1: "Zero Income Households",
            2: "Middle Income Families"
        }
        
        # Income thresholds for validation
        self.k10_income_ranges = {
            0: (500000, 2000000),   # Ultra High Income
            1: (200000, 500000),    # High Income Professionals
            2: (100000, 200000),    # Upper Middle Class
            3: (75000, 100000),     # Middle Class Stable
            4: (50000, 75000),      # Middle Class Growing
            5: (35000, 50000),      # Lower Middle Class
            6: (25000, 35000),      # Working Class
            7: (15000, 25000),      # Low Income Fixed
            8: (5000, 15000),       # Very Low Income
            9: (0, 5000)           # Zero Income Vulnerable
        }
        
        self.k3_income_ranges = {
            0: (100000, 2000000),   # High Income Savers
            1: (0, 5000),          # Zero Income Households
            2: (5000, 100000)       # Middle Income Families
        }
    
    def map_k10_to_k3(self, k10_cluster_id):
        """Map single K=10 cluster to K=3 cluster"""
        return self.k10_to_k3_mapping.get(k10_cluster_id, 2)  # Default to middle income
    
    def get_mapping_summary(self):
        """Display complete mapping summary"""
        print("="*80)
        print(" CLUSTER MAPPING: K=10 to K=3")
        print("="*80)
        
        # Group K=10 clusters by K=3 categories
        k3_groups = {0: [], 1: [], 2: []}
        
        for k10_id, k3_id in self.k10_to_k3_mapping.items():
            k3_groups[k3_id].append(k10_id)
        
        for k3_id, k10_ids in k3_groups.items():
            k3_name = self.k3_names[k3_id]
            print(f"\n{k3_name} (CE Cluster {k3_id}):")
            print(f"   Income Range: ${self.k3_income_ranges[k3_id][0]:,} - ${self.k3_income_ranges[k3_id][1]:,}")
            print(f"   Mapped K=10 Clusters:")
            
            for k10_id in sorted(k10_ids):
                k10_name = self.k10_names[k10_id]
                income_min, income_max = self.k10_income_ranges[k10_id]
                print(f"     • Cluster {k10_id}: {k10_name}")
                print(f"       Income: ${income_min:,} - ${income_max:,}")
    
    def create_sample_data(self, n_households=200):
        """Create sample Under Income data with K=10 clustering"""
        np.random.seed(42)
        data = []
        
        for i in range(n_households):
            # Assign to one of 10 clusters
            k10_cluster_id = np.random.randint(0, 10)
            income_min, income_max = self.k10_income_ranges[k10_cluster_id]
            
            household = {
                'NEWID': f'UI_{i+1:06d}',
                'k10_cluster': k10_cluster_id,
                'k10_cluster_name': self.k10_names[k10_cluster_id],
                'total_income': np.random.uniform(income_min, income_max),
                'age_ref': np.random.randint(18, 80),
                'family_size': np.random.choice([1, 2, 3, 4, 5, 6], p=[0.15, 0.25, 0.25, 0.2, 0.1, 0.05]),
                'employment_status': np.random.choice(['Employed', 'Self-employed', 'Retired', 'Unemployed']),
                'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'])
            }
            
            # Map to K=3 cluster
            household['k3_cluster'] = self.map_k10_to_k3(k10_cluster_id)
            household['k3_cluster_name'] = self.k3_names[household['k3_cluster']]
            
            # Calculate expenditure and savings
            income = household['total_income']
            if income > 0:
                household['total_expenditure'] = income * np.random.uniform(0.7, 0.95)
                household['savings_rate'] = (income - household['total_expenditure']) / income
            else:
                household['total_expenditure'] = np.random.uniform(500, 2000)
                household['savings_rate'] = -1.0
            
            data.append(household)
        
        return pd.DataFrame(data)
    
    def analyze_mapping_accuracy(self, df):
        """Analyze how well the K=10 to K=3 mapping works"""
        print("\n" + "="*80)
        print(" MAPPING ACCURACY ANALYSIS")
        print("="*80)
        
        # Group by K=3 clusters
        k3_analysis = df.groupby('k3_cluster').agg({
            'NEWID': 'count',
            'total_income': ['mean', 'median', 'std'],
            'savings_rate': 'mean',
            'family_size': 'mean'
        }).round(2)
        
        k3_analysis.columns = ['Households', 'Avg Income', 'Median Income', 
                           'Income Std Dev', 'Avg Savings Rate', 'Avg Family Size']
        
        print(f"\n📊 K=3 Cluster Analysis (After Mapping from K=10):")
        for k3_id, row in k3_analysis.iterrows():
            k3_name = self.k3_names[k3_id]
            expected_min, expected_max = self.k3_income_ranges[k3_id]
            
            print(f"\n{k3_name} (Cluster {k3_id}):")
            print(f"   Households: {row['Households']:,}")
            print(f"   Avg Income: ${row['Avg Income']:,.2f}")
            print(f"   Median Income: ${row['Median Income']:,.2f}")
            print(f"   Expected Range: ${expected_min:,} - ${expected_max:,}")
            print(f"   Avg Savings Rate: {row['Avg Savings Rate']:.1%}")
            print(f"   Avg Family Size: {row['Avg Family Size']:.1f}")
            
            # Check if income range matches expectations
            if expected_min <= row['Avg Income'] <= expected_max:
                print(f"   ✓ Income range matches expectations")
            else:
                print(f"   ⚠ Income range outside expectations")
        
        # Show original K=10 distribution within each K=3 cluster
        print(f"\n📋 K=10 Distribution within K=3 Clusters:")
        for k3_id in [0, 1, 2]:
            k3_name = self.k3_names[k3_id]
            k3_data = df[df['k3_cluster'] == k3_id]
            
            print(f"\n{k3_name} (K=3 Cluster {k3_id}):")
            k10_dist = k3_data['k10_cluster'].value_counts().sort_index()
            
            for k10_id, count in k10_dist.items():
                k10_name = self.k10_names[k10_id]
                percentage = (count / len(k3_data)) * 100
                avg_income = k3_data[k3_data['k10_cluster'] == k10_id]['total_income'].mean()
                print(f"   • {k10_name}: {count} households ({percentage:.1f}%) - Avg: ${avg_income:,.2f}")
    
    def demonstrate_mapping(self, n_households=200):
        """Demonstrate the complete mapping process"""
        print(" K=10 TO K=3 CLUSTER MAPPING DEMONSTRATION")
        print("="*80)
        
        # Show mapping summary
        self.get_mapping_summary()
        
        # Create sample data
        print(f"\n📊 Creating sample data with {n_households} households...")
        df = self.create_sample_data(n_households)
        
        # Show original vs mapped distribution
        print(f"\n📈 Distribution Comparison:")
        
        print(f"\nOriginal K=10 Distribution:")
        k10_dist = df['k10_cluster'].value_counts().sort_index()
        for k10_id, count in k10_dist.items():
            k10_name = self.k10_names[k10_id]
            percentage = (count / len(df)) * 100
            print(f"   Cluster {k10_id}: {k10_name} - {count} households ({percentage:.1f}%)")
        
        print(f"\nMapped K=3 Distribution:")
        k3_dist = df['k3_cluster'].value_counts().sort_index()
        for k3_id, count in k3_dist.items():
            k3_name = self.k3_names[k3_id]
            percentage = (count / len(df)) * 100
            print(f"   Cluster {k3_id}: {k3_name} - {count} households ({percentage:.1f}%)")
        
        # Analyze mapping accuracy
        self.analyze_mapping_accuracy(df)
        
        # Show example households
        print(f"\n" + "="*80)
        print(" EXAMPLE HOUSEHOLDS - BEFORE AND AFTER MAPPING")
        print("="*80)
        
        for k3_id in [0, 1, 2]:
            k3_name = self.k3_names[k3_id]
            k3_data = df[df['k3_cluster'] == k3_id].head(3)
            
            print(f"\n{k3_name} (K=3 Cluster {k3_id}):")
            for idx, row in k3_data.iterrows():
                print(f"\n   User ID: {row['NEWID']}")
                print(f"   Original: {row['k10_cluster_name']} (K=10 Cluster {row['k10_cluster']})")
                print(f"   Mapped: {row['k3_cluster_name']} (K=3 Cluster {row['k3_cluster']})")
                print(f"   Income: ${row['total_income']:,.2f}")
                print(f"   Savings Rate: {row['savings_rate']:.1%}")
                print(f"   Family Size: {row['family_size']}")
                print("   " + "-"*40)
        
        return df

def main():
    """Main demonstration function"""
    mapper = ClusterMapper()
    
    # Demonstrate mapping
    mapped_data = mapper.demonstrate_mapping(n_households=200)
    
    print(f"\n K=10 to K=3 mapping completed!")
    print(f"   • 10 granular clusters mapped to 3 broad categories")
    print(f"   • High Income: Clusters 0, 1, 2 (Ultra High, High Prof, Upper Middle)")
    print(f"   • Middle Income: Clusters 3, 4, 5, 6 (Stable, Growing, Lower Middle, Working)")
    print(f"   • Zero Income: Clusters 7, 8, 9 (Low Fixed, Very Low, Zero Vulnerable)")
    print(f"   • Ready for unified analysis across both datasets")

if __name__ == "__main__":
    main()
