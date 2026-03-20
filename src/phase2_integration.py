"""
Phase 2: Integration
Implements unified classification system, product recommendations, and dashboard
Builds on Phase 1 foundation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from cluster_mapping import ClusterMapper
from multi_k_clustering import MultiKClusteringHandler
from phase1_foundation import Phase1Foundation
from new_user_classifier import NewUserClassifier

class UnifiedClassificationSystem:
    """Unified system for classifying users across both datasets"""
    
    def __init__(self):
        self.mapper = ClusterMapper()
        self.multi_k_handler = MultiKClusteringHandler(Path(__file__).parent.parent / 'data')
        self.ce_classifier = MockCEClassifier()
        self.ui_classifier = MockUIClassifier()
        self.phase1_foundation = Phase1Foundation()
        
    def classify_user_unified(self, user_data):
        """Classify user using both CE and UI systems, then unify results"""
        print(f"\n{'='*60}")
        print(" UNIFIED USER CLASSIFICATION")
        print(f"{'='*60}")
        
        # Get classifications from both systems
        ce_result = self.ce_classifier.classify(user_data, dataset='CE')
        ui_result = self.ui_classifier.classify(user_data, dataset='Under_Income')
        
        # Map UI result to K=3 for comparison
        ui_mapped_cluster = self.mapper.map_k10_to_k3(ui_result['cluster_id'])
        ui_mapped_result = ui_result.copy()
        ui_mapped_result['cluster_id'] = ui_mapped_cluster
        ui_mapped_result['cluster_name'] = self.mapper.k3_names[ui_mapped_cluster]
        
        # Create unified profile
        unified_profile = self._create_unified_profile(user_data, ce_result, ui_mapped_result)
        
        # Display results
        self._display_unified_results(unified_profile)
        
        return unified_profile
    
    def _create_unified_profile(self, user_data, ce_result, ui_result):
        """Create unified user profile from both classifications"""
        # Determine consensus classification
        consensus_cluster = self._get_consensus_cluster(ce_result, ui_result)
        
        unified_profile = {
            'user_id': user_data.get('NEWID', 'Unknown'),
            'input_data': user_data,
            'ce_classification': ce_result,
            'ui_classification': ui_result,
            'consensus_cluster_id': consensus_cluster,
            'consensus_cluster_name': self.mapper.k3_names[consensus_cluster],
            'classification_confidence': self._calculate_confidence(ce_result, ui_result),
            'data_sources': ['CE', 'Under_Income'],
            'classification_timestamp': pd.Timestamp.now()
        }
        
        return unified_profile
    
    def _get_consensus_cluster(self, ce_result, ui_result):
        """Determine consensus cluster from both classifications"""
        ce_cluster = ce_result['cluster_id']
        ui_cluster = ui_result['cluster_id']
        
        # If both agree, return that cluster
        if ce_cluster == ui_cluster:
            return ce_cluster
        
        # If they disagree, use income-based logic
        income = ce_result.get('total_income', 0) or ui_result.get('total_income', 0)
        
        if income == 0 or income < 5000:
            return 1  # Zero Income Households
        elif income > 150000:
            return 0  # High Income Savers
        else:
            return 2  # Middle Income Families
    
    def _calculate_confidence(self, ce_result, ui_result):
        """Calculate confidence in unified classification"""
        ce_cluster = ce_result['cluster_id']
        ui_cluster = ui_result['cluster_id']
        
        if ce_cluster == ui_cluster:
            return 0.95  # High confidence when both agree
        else:
            return 0.75  # Moderate confidence when they disagree
    
    def _display_unified_results(self, unified_profile):
        """Display unified classification results"""
        print(f"\n User ID: {unified_profile['user_id']}")
        print(f" Input Data: Income ${unified_profile['input_data'].get('total_income', 0):,.2f}")
        
        print(f"\n Classification Results:")
        print(f"   CE System: {unified_profile['ce_classification']['cluster_name']} (Cluster {unified_profile['ce_classification']['cluster_id']})")
        print(f"   UI System: {unified_profile['ui_classification']['cluster_name']} (Cluster {unified_profile['ui_classification']['cluster_id']})")
        
        print(f"\n Unified Classification:")
        print(f"   Consensus: {unified_profile['consensus_cluster_name']} (Cluster {unified_profile['consensus_cluster_id']})")
        print(f"   Confidence: {unified_profile['classification_confidence']:.1%}")
        print(f"   Data Sources: {', '.join(unified_profile['data_sources'])}")

class MockCEClassifier:
    """Mock CE classifier (K=3) - replace with real model later"""
    
    def __init__(self):
        self.cluster_names = {
            0: "High Income Savers",
            1: "Zero Income Households",
            2: "Middle Income Families"
        }
    
    def classify(self, user_data, dataset='CE'):
        """Mock classification logic"""
        income = user_data.get('total_income', 0)
        age = user_data.get('age_ref', 30)
        family_size = user_data.get('family_size', 2)
        
        # Calculate derived features
        expenditure = user_data.get('total_expenditure', income * 0.8)
        savings_rate = (income - expenditure) / income if income > 0 else -1
        
        # Mock clustering logic
        if income == 0 or income < 5000:
            cluster_id = 1  # Zero Income Households
        elif income > 150000 and savings_rate > 0.2:
            cluster_id = 0  # High Income Savers
        else:
            cluster_id = 2  # Middle Income Families
        
        return {
            'cluster_id': cluster_id,
            'cluster_name': self.cluster_names[cluster_id],
            'total_income': income,
            'savings_rate': savings_rate,
            'confidence': 0.85,
            'dataset': dataset
        }

class MockUIClassifier:
    """Mock Under Income classifier (K=10) - replace with real model later"""
    
    def __init__(self):
        self.cluster_names = {
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
    
    def classify(self, user_data, dataset='Under_Income'):
        """Mock classification logic"""
        income = user_data.get('total_income', 0)
        age = user_data.get('age_ref', 30)
        employment = user_data.get('employment_status', 'Employed')
        
        # Mock K=10 clustering logic
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
        
        return {
            'cluster_id': cluster_id,
            'cluster_name': self.cluster_names[cluster_id],
            'total_income': income,
            'confidence': 0.85,
            'dataset': dataset
        }

class UnifiedProductRecommender:
    """Unified product recommendation system"""
    
    def __init__(self):
        self.mock_xgb_models = self._create_mock_xgb_models()
        self.mapper = ClusterMapper()
        
        # Cluster-specific product prioritization
        self.cluster_product_mapping = {
            'High Income Savers': ['needs_investment_product', 'high_income'],
            'Middle Income Families': ['needs_savings_product', 'needs_insurance_product', 'needs_loan_product'],
            'Zero Income Households': ['needs_loan_product']  # Assistance programs
        }
    
    def _create_mock_xgb_models(self):
        """Create mock XGBoost models - replace with real models later"""
        return {
            'needs_savings_product': MockXGBModel('needs_savings_product'),
            'needs_investment_product': MockXGBModel('needs_investment_product'),
            'needs_insurance_product': MockXGBModel('needs_insurance_product'),
            'needs_loan_product': MockXGBModel('needs_loan_product'),
            'high_spender': MockXGBModel('high_spender'),
            'high_income': MockXGBModel('high_income')
        }
    
    def get_recommendations(self, unified_profile):
        """Get product recommendations for unified user profile"""
        print(f"\n{'='*60}")
        print(" PRODUCT RECOMMENDATIONS")
        print(f"{'='*60}")
        
        user_data = unified_profile['input_data']
        cluster_name = unified_profile['consensus_cluster_name']
        
        # Get predictions from all models
        product_predictions = {}
        
        for product_name, model in self.mock_xgb_models.items():
            prediction = model.predict(user_data)
            product_predictions[product_name] = prediction
        
        # Align with unified clusters
        recommendations = self._align_with_clusters(cluster_name, product_predictions)
        
        # Display recommendations
        self._display_recommendations(cluster_name, recommendations, product_predictions)
        
        return {
            'cluster_name': cluster_name,
            'primary_recommendations': recommendations['primary'],
            'secondary_recommendations': recommendations['secondary'],
            'all_predictions': product_predictions,
            'recommendation_confidence': recommendations['confidence']
        }
    
    def _align_with_clusters(self, cluster_name, product_predictions):
        """Align product predictions with cluster-specific priorities"""
        priority_products = self.cluster_product_mapping.get(cluster_name, [])
        
        # Separate primary and secondary recommendations
        primary = []
        secondary = []
        
        for product in priority_products:
            if product_predictions[product]['needs_product']:
                primary.append(product)
        
        # Add other high-confidence predictions as secondary
        for product, prediction in product_predictions.items():
            if product not in priority_products and prediction['needs_product']:
                secondary.append(product)
        
        # Calculate overall confidence
        all_predictions = list(product_predictions.values())
        confidence = np.mean([p['probability'] for p in all_predictions if p['needs_product']])
        
        return {
            'primary': primary,
            'secondary': secondary,
            'confidence': confidence
        }
    
    def _display_recommendations(self, cluster_name, recommendations, product_predictions):
        """Display product recommendations"""
        print(f"\n Cluster: {cluster_name}")
        
        if recommendations['primary']:
            print(f"\n Primary Recommendations:")
            for product in recommendations['primary']:
                pred = product_predictions[product]
                print(f"   • {product.replace('_', ' ').title()}")
                print(f"     Probability: {pred['probability']:.1%}")
                print(f"     Reason: {pred['reason']}")
        
        if recommendations['secondary']:
            print(f"\n Secondary Recommendations:")
            for product in recommendations['secondary']:
                pred = product_predictions[product]
                print(f"   • {product.replace('_', ' ').title()}")
                print(f"     Probability: {pred['probability']:.1%}")
        
        if not recommendations['primary'] and not recommendations['secondary']:
            print(f"\n No immediate product needs identified")
        
        print(f"\n Overall Recommendation Confidence: {recommendations['confidence']:.1%}")

class MockXGBModel:
    """Mock XGBoost model - replace with real model later"""
    
    def __init__(self, product_name):
        self.product_name = product_name
        self.business_logic = self._get_business_logic(product_name)
    
    def _get_business_logic(self, product_name):
        """Get business logic for each product"""
        logic_map = {
            'needs_savings_product': 'Low savings rate detected',
            'needs_investment_product': 'High income with good savings capacity',
            'needs_insurance_product': 'High healthcare spending or older age',
            'needs_loan_product': 'High spending ratio in working age',
            'high_spender': 'High expenditure relative to income',
            'high_income': 'High income household'
        }
        return logic_map.get(product_name, 'General financial need')
    
    def predict(self, user_data):
        """Mock prediction logic"""
        income = user_data.get('total_income', 0)
        age = user_data.get('age_ref', 30)
        family_size = user_data.get('family_size', 2)
        
        # Calculate derived features
        expenditure = user_data.get('total_expenditure', income * 0.8)
        savings_rate = (income - expenditure) / income if income > 0 else -1
        expenditure_ratio = expenditure / income if income > 0 else 1
        
        # Mock prediction logic based on product
        if self.product_name == 'needs_savings_product':
            probability = max(0.1, 1 - savings_rate) if income > 0 else 0.1
            needs_product = savings_rate < 0.1 and income > 0
            
        elif self.product_name == 'needs_investment_product':
            probability = min(0.95, income / 100000) if income > 50000 else 0.1
            needs_product = income > 80000 and savings_rate > 0.15
            
        elif self.product_name == 'needs_insurance_product':
            healthcare_ratio = user_data.get('healthcare_expenditure_ratio', 0)
            probability = min(0.9, healthcare_ratio * 5) if healthcare_ratio > 0 else 0.2
            needs_product = healthcare_ratio > 0.1 or age > 60
            
        elif self.product_name == 'needs_loan_product':
            probability = min(0.8, expenditure_ratio) if expenditure_ratio > 0.8 else 0.1
            needs_product = expenditure_ratio > 0.9 and 25 <= age <= 65
            
        elif self.product_name == 'high_spender':
            probability = min(0.9, expenditure_ratio)
            needs_product = expenditure_ratio > 0.85
            
        elif self.product_name == 'high_income':
            probability = min(0.95, income / 120000) if income > 50000 else 0.1
            needs_product = income > 100000
            
        else:
            probability = 0.5
            needs_product = False
        
        return {
            'needs_product': needs_product,
            'probability': probability,
            'reason': self.business_logic
        }

class UnifiedDashboard:
    """Unified dashboard for reporting and visualization"""
    
    def __init__(self):
        self.classification_system = UnifiedClassificationSystem()
        self.product_recommender = UnifiedProductRecommender()
        
    def generate_dashboard(self, sample_users=None):
        """Generate comprehensive dashboard"""
        print("="*80)
        print(" UNIFIED DASHBOARD")
        print("="*80)
        
        if sample_users is None:
            sample_users = self._create_sample_users()
        
        # Process all users
        user_profiles = []
        product_recommendations = []
        
        for user_data in sample_users:
            # Classify user
            profile = self.classification_system.classify_user_unified(user_data)
            user_profiles.append(profile)
            
            # Get recommendations
            recommendations = self.product_recommender.get_recommendations(profile)
            product_recommendations.append(recommendations)
        
        # Generate dashboard sections
        self._generate_classification_summary(user_profiles)
        self._generate_product_summary(product_recommendations)
        self._generate_cross_dataset_analysis(user_profiles)
        self._create_dashboard_visualizations(user_profiles, product_recommendations)
        
        return {
            'user_profiles': user_profiles,
            'product_recommendations': product_recommendations,
            'summary_metrics': self._calculate_summary_metrics(user_profiles, product_recommendations)
        }
    
    def _create_sample_users(self):
        """Create sample users for dashboard"""
        return [
            {
                'NEWID': 'DASH_USER_001',
                'total_income': 85000,
                'age_ref': 35,
                'family_size': 4,
                'total_expenditure': 65000,
                'employment_status': 'Employed',
                'education_level': 'Bachelor'
            },
            {
                'NEWID': 'DASH_USER_002',
                'total_income': 250000,
                'age_ref': 45,
                'family_size': 3,
                'total_expenditure': 150000,
                'employment_status': 'Self-employed',
                'education_level': 'Master'
            },
            {
                'NEWID': 'DASH_USER_003',
                'total_income': 0,
                'age_ref': 28,
                'family_size': 2,
                'total_expenditure': 2000,
                'employment_status': 'Unemployed',
                'education_level': 'High School'
            },
            {
                'NEWID': 'DASH_USER_004',
                'total_income': 120000,
                'age_ref': 55,
                'family_size': 2,
                'total_expenditure': 90000,
                'employment_status': 'Employed',
                'education_level': 'PhD'
            },
            {
                'NEWID': 'DASH_USER_005',
                'total_income': 45000,
                'age_ref': 32,
                'family_size': 3,
                'total_expenditure': 42000,
                'employment_status': 'Employed',
                'education_level': 'Bachelor'
            }
        ]
    
    def _generate_classification_summary(self, user_profiles):
        """Generate classification summary"""
        print(f"\n{'='*60}")
        print(" CLASSIFICATION SUMMARY")
        print(f"{'='*60}")
        
        # Cluster distribution
        cluster_counts = {}
        confidence_scores = []
        
        for profile in user_profiles:
            cluster = profile['consensus_cluster_name']
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            confidence_scores.append(profile['classification_confidence'])
        
        print(f"\n Cluster Distribution:")
        for cluster, count in cluster_counts.items():
            percentage = (count / len(user_profiles)) * 100
            print(f"   {cluster}: {count} users ({percentage:.1f}%)")
        
        print(f"\n Classification Confidence:")
        print(f"   Average: {np.mean(confidence_scores):.1%}")
        print(f"   Range: {min(confidence_scores):.1%} - {max(confidence_scores):.1%}")
        
        # Cross-dataset agreement
        agreements = []
        for profile in user_profiles:
            ce_cluster = profile['ce_classification']['cluster_id']
            ui_cluster = profile['ui_classification']['cluster_id']
            agreements.append(ce_cluster == ui_cluster)
        
        agreement_rate = np.mean(agreements)
        print(f"\n Cross-Dataset Agreement:")
        print(f"   Agreement Rate: {agreement_rate:.1%}")
        print(f"   Disagreements: {sum(not a for a in agreements)} out of {len(user_profiles)}")
    
    def _generate_product_summary(self, product_recommendations):
        """Generate product recommendation summary"""
        print(f"\n{'='*60}")
        print(" PRODUCT RECOMMENDATION SUMMARY")
        print(f"{'='*60}")
        
        # Product need rates
        product_counts = {}
        confidence_scores = []
        
        for rec in product_recommendations:
            for product in rec['all_predictions'].values():
                if product['needs_product']:
                    product_name = product['reason'].split()[0]  # Get first word as proxy
                    product_counts[product_name] = product_counts.get(product_name, 0) + 1
            
            confidence_scores.append(rec['recommendation_confidence'])
        
        print(f"\n Product Need Rates:")
        for product, count in product_counts.items():
            percentage = (count / len(product_recommendations)) * 100
            print(f"   {product}: {count} users ({percentage:.1f}%)")
        
        print(f"\n Recommendation Confidence:")
        print(f"   Average: {np.mean(confidence_scores):.1%}")
    
    def _generate_cross_dataset_analysis(self, user_profiles):
        """Generate cross-dataset analysis"""
        print(f"\n{'='*60}")
        print(" CROSS-DATASET ANALYSIS")
        print(f"{'='*60}")
        
        # Income distribution by dataset
        ce_incomes = [p['ce_classification']['total_income'] for p in user_profiles]
        ui_incomes = [p['ui_classification']['total_income'] for p in user_profiles]
        
        print(f"\n Income Distribution:")
        print(f"   CE System - Mean: ${np.mean(ce_incomes):,.2f}, Median: ${np.median(ce_incomes):,.2f}")
        print(f"   UI System - Mean: ${np.mean(ui_incomes):,.2f}, Median: ${np.median(ui_incomes):,.2f}")
        
        # Classification differences
        differences = []
        for profile in user_profiles:
            if profile['ce_classification']['cluster_id'] != profile['ui_classification']['cluster_id']:
                differences.append({
                    'user_id': profile['user_id'],
                    'ce_cluster': profile['ce_classification']['cluster_name'],
                    'ui_cluster': profile['ui_classification']['cluster_name'],
                    'consensus': profile['consensus_cluster_name']
                })
        
        if differences:
            print(f"\n Classification Differences:")
            for diff in differences:
                print(f"   User {diff['user_id']}: CE={diff['ce_cluster']}, UI={diff['ui_cluster']} → Consensus={diff['consensus']}")
        else:
            print(f"\n No classification differences found")
    
    def _create_dashboard_visualizations(self, user_profiles, product_recommendations):
        """Create dashboard visualizations"""
        viz_dir = Path(__file__).parent.parent / 'results' / 'phase2_visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dashboard plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Unified Dashboard - Phase 2 Integration', fontsize=16)
        
        # 1. Cluster distribution
        cluster_counts = {}
        for profile in user_profiles:
            cluster = profile['consensus_cluster_name']
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        
        axes[0, 0].pie(cluster_counts.values(), labels=cluster_counts.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Unified Cluster Distribution')
        
        # 2. Classification confidence
        confidences = [p['classification_confidence'] for p in user_profiles]
        axes[0, 1].hist(confidences, bins=10, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Classification Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Income distribution
        incomes = [p['input_data']['total_income'] for p in user_profiles]
        axes[1, 0].hist(incomes, bins=10, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('User Income Distribution')
        axes[1, 0].set_xlabel('Total Income ($)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Product recommendation confidence
        product_confidences = [r['recommendation_confidence'] for r in product_recommendations]
        axes[1, 1].hist(product_confidences, bins=10, alpha=0.7, color='orange')
        axes[1, 1].set_title('Product Recommendation Confidence')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'phase2_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n Dashboard visualizations saved to: {viz_dir / 'phase2_dashboard.png'}")
    
    def _calculate_summary_metrics(self, user_profiles, product_recommendations):
        """Calculate summary metrics for dashboard"""
        return {
            'total_users': len(user_profiles),
            'avg_classification_confidence': np.mean([p['classification_confidence'] for p in user_profiles]),
            'avg_recommendation_confidence': np.mean([r['recommendation_confidence'] for r in product_recommendations]),
            'cross_dataset_agreement_rate': np.mean([
                p['ce_classification']['cluster_id'] == p['ui_classification']['cluster_id'] 
                for p in user_profiles
            ]),
            'cluster_distribution': {
                cluster: sum(1 for p in user_profiles if p['consensus_cluster_name'] == cluster)
                for cluster in set(p['consensus_cluster_name'] for p in user_profiles)
            }
        }

def main():
    """Main Phase 2 execution"""
    print(" PHASE 2: INTEGRATION IMPLEMENTATION")
    print("="*80)
    print("Objectives: Unified classification, product recommendations, dashboard")
    
    # Initialize systems
    dashboard = UnifiedDashboard()
    
    # Generate comprehensive dashboard
    results = dashboard.generate_dashboard()
    
    print(f"\n{'='*80}")
    print(" PHASE 2 COMPLETION REPORT")
    print(f"{'='*80}")
    
    metrics = results['summary_metrics']
    
    print(f"\n PHASE 2 ACHIEVEMENTS:")
    print(" Unified classification system implemented")
    print(" Product recommendation system integrated")
    print(" Comprehensive dashboard created")
    print(" Cross-dataset analysis completed")
    
    print(f"\n KEY METRICS:")
    print(f"   Total Users Processed: {metrics['total_users']}")
    print(f"   Avg Classification Confidence: {metrics['avg_classification_confidence']:.1%}")
    print(f"   Avg Recommendation Confidence: {metrics['avg_recommendation_confidence']:.1%}")
    print(f"   Cross-Dataset Agreement: {metrics['cross_dataset_agreement_rate']:.1%}")
    
    print(f"\n CLUSTER DISTRIBUTION:")
    for cluster, count in metrics['cluster_distribution'].items():
        percentage = (count / metrics['total_users']) * 100
        print(f"   {cluster}: {count} users ({percentage:.1f}%)")
    
    print(f"\n PHASE 2 INTEGRATION COMPLETED!")
    
    return results

if __name__ == "__main__":
    main()
