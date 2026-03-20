"""
Interactive Financial Recommendation Demo
Asks users for basic information and provides personalized recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our systems
from cluster_mapping import ClusterMapper

class InteractiveDemo:
    """Interactive demo for financial recommendations"""
    
    def __init__(self):
        self.mapper = ClusterMapper()
        
    def get_user_input(self):
        """Get user input through interactive prompts"""
        print("="*50)
        print(" PERSONALIZED FINANCIAL RECOMMENDATION SYSTEM")
        print("="*50)
        print("Answer a few questions to get personalized financial advice")
        
        user_data = {}
        
        # Get basic information
        print(f"\n{'='*50}")
        print(" BASIC INFORMATION")
        print(f"{'='*50}")
        
        try:
            # Salary/Income
            while True:
                salary_input = input(f"\nWhat is your annual household income? (e.g., 50000, 75000, 120000): ")
                try:
                    salary = float(salary_input.replace(',', '').replace('$', ''))
                    if salary < 0:
                        print("Income cannot be negative. Please try again.")
                        continue
                    user_data['total_income'] = salary
                    break
                except ValueError:
                    print("Please enter a valid number. Try again.")
            
            # Family size
            while True:
                family_input = input(f"\nHow many people are in your household? (1-10): ")
                try:
                    family_size = int(family_input)
                    if family_size < 1 or family_size > 10:
                        print("Family size must be between 1 and 10. Please try again.")
                        continue
                    user_data['family_size'] = family_size
                    break
                except ValueError:
                    print("Please enter a valid number. Try again.")
            
            # Age
            while True:
                age_input = input(f"\nWhat is your age? (18-100): ")
                try:
                    age = int(age_input)
                    if age < 18 or age > 100:
                        print("Age must be between 18 and 100. Please try again.")
                        continue
                    user_data['age_ref'] = age
                    break
                except ValueError:
                    print("Please enter a valid number. Try again.")
            
            # Employment status
            print(f"\nEmployment Status:")
            print("1. Employed")
            print("2. Self-employed")
            print("3. Unemployed")
            print("4. Retired")
            
            while True:
                employment_input = input(f"\nSelect your employment status (1-4): ")
                if employment_input in ['1', '2', '3', '4']:
                    employment_map = {
                        '1': 'Employed',
                        '2': 'Self-employed',
                        '3': 'Unemployed',
                        '4': 'Retired'
                    }
                    user_data['employment_status'] = employment_map[employment_input]
                    break
                else:
                    print("Please select a valid option (1-4). Try again.")
            
            # Education level
            print(f"\nEducation Level:")
            print("1. High School")
            print("2. Bachelor's Degree")
            print("3. Master's Degree")
            print("4. PhD")
            
            while True:
                education_input = input(f"\nSelect your education level (1-4): ")
                if education_input in ['1', '2', '3', '4']:
                    education_map = {
                        '1': 'High School',
                        '2': 'Bachelor',
                        '3': 'Master',
                        '4': 'PhD'
                    }
                    user_data['education_level'] = education_map[education_input]
                    break
                else:
                    print("Please select a valid option (1-4). Try again.")
            
            # Monthly expenses (optional)
            expense_input = input(f"\nWhat are your monthly household expenses? (Press Enter to estimate): ")
            if expense_input.strip():
                try:
                    monthly_expenses = float(expense_input.replace(',', '').replace('$', ''))
                    user_data['total_expenditure'] = monthly_expenses * 12  # Annual
                except ValueError:
                    print("Invalid input. Using estimated expenses.")
                    user_data['total_expenditure'] = self._estimate_expenses(user_data)
            else:
                user_data['total_expenditure'] = self._estimate_expenses(user_data)
            
            # Healthcare expenses (optional)
            healthcare_input = input(f"\nWhat are your monthly healthcare expenses? (Press Enter to estimate): ")
            if healthcare_input.strip():
                try:
                    monthly_healthcare = float(healthcare_input.replace(',', '').replace('$', ''))
                    annual_healthcare = monthly_healthcare * 12
                    user_data['healthcare_expenditure_ratio'] = annual_healthcare / user_data['total_income'] if user_data['total_income'] > 0 else 0
                except ValueError:
                    print("Invalid input. Using estimated healthcare ratio.")
                    user_data['healthcare_expenditure_ratio'] = self._estimate_healthcare_ratio(user_data)
            else:
                user_data['healthcare_expenditure_ratio'] = self._estimate_healthcare_ratio(user_data)
            
            # Generate user ID
            user_data['NEWID'] = f'DEMO_USER_{np.random.randint(10000, 99999)}'
            
            return user_data
            
        except KeyboardInterrupt:
            print(f"\n\nDemo cancelled by user.")
            return None
    
    def _estimate_expenses(self, user_data):
        """Estimate monthly expenses based on income and family size"""
        income = user_data['total_income']
        family_size = user_data['family_size']
        
        # Basic expense estimation rules
        if income == 0:
            # Basic survival expenses
            base_expense = 2000
            family_expense = (family_size - 1) * 500
        else:
            # Percentage-based estimation
            if income < 30000:
                expense_ratio = 0.85  # High expense ratio for low income
            elif income < 75000:
                expense_ratio = 0.80
            elif income < 150000:
                expense_ratio = 0.75
            else:
                expense_ratio = 0.70
            
            base_expense = income * expense_ratio
            family_expense = (family_size - 2) * 3000 if family_size > 2 else 0
        
        return base_expense + family_expense
    
    def _estimate_healthcare_ratio(self, user_data):
        """Estimate healthcare expense ratio"""
        age = user_data['age_ref']
        income = user_data['total_income']
        
        if age > 65:
            return 0.12  # Higher healthcare costs for seniors
        elif age > 50:
            return 0.08
        elif age > 35:
            return 0.05
        else:
            return 0.03
    
    def classify_user(self, user_data):
        """Classify user into cluster based on their data"""
        print(f"\n{'='*60}")
        print(" USER CLASSIFICATION")
        print(f"{'='*60}")
        
        income = user_data['total_income']
        family_size = user_data['family_size']
        age = user_data['age_ref']
        expenditure = user_data['total_expenditure']
        
        # Calculate savings rate
        savings_rate = (income - expenditure) / income if income > 0 else -1
        
        # Classification logic (simplified version of our system)
        if income == 0 or income < 5000:
            cluster_id = 1
            cluster_name = "Zero Income Households"
            confidence = 0.9
        elif income > 150000 and savings_rate > 0.15:
            cluster_id = 0
            cluster_name = "High Income Savers"
            confidence = 0.85
        else:
            cluster_id = 2
            cluster_name = "Middle Income Families"
            confidence = 0.8
        
        # Create user profile
        user_profile = {
            'user_id': user_data['NEWID'],
            'input_data': user_data,
            'consensus_cluster_id': cluster_id,
            'consensus_cluster_name': cluster_name,
            'classification_confidence': confidence,
            'ce_classification': {
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'total_income': income,
                'savings_rate': savings_rate
            },
            'ui_classification': {
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'total_income': income
            }
        }
        
        # Display classification
        print(f"\nYour Financial Profile:")
        print(f"   Income: ${income:,.2f}")
        print(f"   Family Size: {family_size}")
        print(f"   Age: {age}")
        print(f"   Employment: {user_data['employment_status']}")
        print(f"   Estimated Annual Expenses: ${expenditure:,.2f}")
        print(f"   Savings Rate: {savings_rate:.1%}")
        
        print(f"\nClassification Results:")
        print(f"   Your Group: {cluster_name}")
        print(f"   Group ID: {cluster_id}")
        print(f"   Confidence: {confidence:.1%}")
        
        return user_profile
    
    def generate_product_recommendations(self, user_profile):
        """Generate product recommendations based on user profile"""
        print(f"\n{'='*60}")
        print(" PRODUCT RECOMMENDATIONS")
        print(f"{'='*60}")
        
        user_data = user_profile['input_data']
        income = user_data['total_income']
        age = user_data['age_ref']
        family_size = user_data['family_size']
        expenditure = user_data['total_expenditure']
        savings_rate = (income - expenditure) / income if income > 0 else -1
        healthcare_ratio = user_data.get('healthcare_expenditure_ratio', 0)
        
        recommendations = {}
        
        # Savings product recommendation
        if savings_rate < 0.1 and income > 0:
            recommendations['needs_savings_product'] = {
                'needs_product': True,
                'probability': max(0.1, 1 - savings_rate),
                'reason': 'Low savings rate detected',
                'urgency': 'High',
                'action': 'Build emergency fund with high-yield savings account'
            }
        else:
            recommendations['needs_savings_product'] = {
                'needs_product': False,
                'probability': 0.2,
                'reason': 'Savings rate is adequate'
            }
        
        # Investment product recommendation
        if income > 80000 and savings_rate > 0.15:
            recommendations['needs_investment_product'] = {
                'needs_product': True,
                'probability': min(0.95, income / 100000),
                'reason': 'High income with good savings capacity',
                'urgency': 'Medium',
                'action': 'Start with diversified ETF portfolio'
            }
        else:
            recommendations['needs_investment_product'] = {
                'needs_product': False,
                'probability': 0.3,
                'reason': 'Income or savings rate not sufficient for investing'
            }
        
        # Insurance product recommendation
        if healthcare_ratio > 0.1 or age > 60 or family_size > 3:
            recommendations['needs_insurance_product'] = {
                'needs_product': True,
                'probability': min(0.9, healthcare_ratio * 5) if healthcare_ratio > 0 else 0.5,
                'reason': 'Healthcare costs or family vulnerability detected',
                'urgency': 'High',
                'action': 'Review health and life insurance coverage'
            }
        else:
            recommendations['needs_insurance_product'] = {
                'needs_product': False,
                'probability': 0.2,
                'reason': 'Adequate protection currently'
            }
        
        # Loan product recommendation
        expenditure_ratio = expenditure / income if income > 0 else 1
        if expenditure_ratio > 0.9 and 25 <= age <= 65 and income > 0:
            recommendations['needs_loan_product'] = {
                'needs_product': True,
                'probability': min(0.8, expenditure_ratio),
                'reason': 'High spending ratio indicates cash flow issues',
                'urgency': 'Medium',
                'action': 'Consider debt consolidation or personal loan'
            }
        else:
            recommendations['needs_loan_product'] = {
                'needs_product': False,
                'probability': 0.1,
                'reason': 'Cash flow appears manageable'
            }
        
        # Display recommendations
        print(f"\nPersonalized Recommendations:")
        
        urgent_recommendations = []
        medium_recommendations = []
        low_recommendations = []
        
        for product, rec in recommendations.items():
            if rec['needs_product']:
                product_name = product.replace('_', ' ').title()
                
                recommendation_text = f"\n{product_name}:"
                recommendation_text += f"\n   Reason: {rec['reason']}"
                recommendation_text += f"\n   Probability: {rec['probability']:.1%}"
                recommendation_text += f"\n   Urgency: {rec['urgency']}"
                recommendation_text += f"\n   Action: {rec['action']}"
                
                if rec['urgency'] == 'High':
                    urgent_recommendations.append(recommendation_text)
                elif rec['urgency'] == 'Medium':
                    medium_recommendations.append(recommendation_text)
                else:
                    low_recommendations.append(recommendation_text)
        
        if urgent_recommendations:
            print(f"\nURGENT RECOMMENDATIONS:")
            for rec in urgent_recommendations:
                print(rec)
        
        if medium_recommendations:
            print(f"\nMEDIUM PRIORITY RECOMMENDATIONS:")
            for rec in medium_recommendations:
                print(rec)
        
        if low_recommendations:
            print(f"\nLOW PRIORITY RECOMMENDATIONS:")
            for rec in low_recommendations:
                print(rec)
        
        if not any(rec['needs_product'] for rec in recommendations.values()):
            print(f"\nNo immediate financial product needs identified.")
            
            income = user_data['total_income']
            family_size = user_data['family_size']
            age = user_data['age_ref']
            expenditure = user_data['total_expenditure']
            
            # Calculate savings rate
            savings_rate = (income - expenditure) / income if income > 0 else -1
            
            # Classification logic (simplified version of our system)
            if income == 0 or income < 5000:
                cluster_id = 1
                cluster_name = "Zero Income Households"
                confidence = 0.9
            elif income > 150000 and savings_rate > 0.15:
                cluster_id = 0
                cluster_name = "High Income Savers"
                confidence = 0.85
            else:
                cluster_id = 2
                cluster_name = "Middle Income Families"
                confidence = 0.8
            
            # Create user profile
            user_profile = {
                'user_id': user_data['NEWID'],
                'input_data': user_data,
                'consensus_cluster_id': cluster_id,
                'consensus_cluster_name': cluster_name,
                'classification_confidence': confidence,
                'ce_classification': {
                    'cluster_id': cluster_id,
                    'cluster_name': cluster_name,
                    'total_income': income,
                    'savings_rate': savings_rate
                },
                'ui_classification': {
                    'cluster_id': cluster_id,
                    'cluster_name': cluster_name,
                    'total_income': income
                }
            }
            
            # Display classification
            print(f"\nYour Financial Profile:")
            print(f"   Income: ${income:,.2f}")
            print(f"   Family Size: {family_size}")
            print(f"   Age: {age}")
            print(f"   Employment: {user_data['employment_status']}")
            print(f"   Estimated Annual Expenses: ${expenditure:,.2f}")
            print(f"   Savings Rate: {savings_rate:.1%}")
            
            print(f"\nClassification Results:")
            print(f"   Your Group: {cluster_name}")
            print(f"   Group ID: {cluster_id}")
            print(f"   Confidence: {confidence:.1%}")
            
            return user_profile
    
    def generate_product_recommendations(self, user_profile):
        """Generate product recommendations based on user profile"""
        print(f"\n{'='*50}")
        print(" PRODUCT RECOMMENDATIONS")
        print(f"{'='*50}")
        
        user_data = user_profile['input_data']
        income = user_data['total_income']
        age = user_data['age_ref']
        family_size = user_data['family_size']
        expenditure = user_data['total_expenditure']
        savings_rate = (income - expenditure) / income if income > 0 else -1
        healthcare_ratio = user_data.get('healthcare_expenditure_ratio', 0)
        
        recommendations = {}
        
        # Savings product recommendation
        if savings_rate < 0.1 and income > 0:
            recommendations['needs_savings_product'] = {
                'needs_product': True,
                'probability': max(0.1, 1 - savings_rate),
                'reason': 'Low savings rate detected',
                'urgency': 'High',
                'action': 'Build emergency fund with high-yield savings account'
            }
        else:
            recommendations['needs_savings_product'] = {
                'needs_product': False,
                'probability': 0.2,
                'reason': 'Savings rate is adequate'
            }
        
        # Investment product recommendation
        if income > 80000 and savings_rate > 0.15:
            recommendations['needs_investment_product'] = {
                'needs_product': True,
                'probability': min(0.95, income / 100000),
                'reason': 'High income with good savings capacity',
                'urgency': 'Medium',
                'action': 'Start with diversified ETF portfolio'
            }
        else:
            recommendations['needs_investment_product'] = {
                'needs_product': False,
                'probability': 0.3,
                'reason': 'Income or savings rate not sufficient for investing'
            }
        
        # Insurance product recommendation
        if healthcare_ratio > 0.1 or age > 60 or family_size > 3:
            recommendations['needs_insurance_product'] = {
                'needs_product': True,
                'probability': min(0.9, healthcare_ratio * 5) if healthcare_ratio > 0 else 0.5,
                'reason': 'Healthcare costs or family vulnerability detected',
                'urgency': 'High',
                'action': 'Review health and life insurance coverage'
            }
        else:
            recommendations['needs_insurance_product'] = {
                'needs_product': False,
                'probability': 0.2,
                'reason': 'Adequate protection currently'
            }
        
        # Loan product recommendation
        expenditure_ratio = expenditure / income if income > 0 else 1
        if expenditure_ratio > 0.9 and 25 <= age <= 65 and income > 0:
            recommendations['needs_loan_product'] = {
                'needs_product': True,
                'probability': min(0.8, expenditure_ratio),
                'reason': 'High spending ratio indicates cash flow issues',
                'urgency': 'Medium',
                'action': 'Consider debt consolidation or personal loan'
            }
        else:
            recommendations['needs_loan_product'] = {
                'needs_product': False,
                'probability': 0.1,
                'reason': 'Cash flow appears manageable'
            }
        
        # Display recommendations
        print(f"\nPersonalized Recommendations:")
        
        urgent_recommendations = []
        medium_recommendations = []
        low_recommendations = []
        
        for product, rec in recommendations.items():
            if rec['needs_product']:
                product_name = product.replace('_', ' ').title()
                
                recommendation_text = f"\n{product_name}:"
                recommendation_text += f"\n   Reason: {rec['reason']}"
                recommendation_text += f"\n   Probability: {rec['probability']:.1%}"
                recommendation_text += f"\n   Urgency: {rec['urgency']}"
                recommendation_text += f"\n   Action: {rec['action']}"
                
                if rec['urgency'] == 'High':
                    urgent_recommendations.append(recommendation_text)
                elif rec['urgency'] == 'Medium':
                    medium_recommendations.append(recommendation_text)
                else:
                    low_recommendations.append(recommendation_text)
        
        if urgent_recommendations:
            print(f"\nURGENT RECOMMENDATIONS:")
            for rec in urgent_recommendations:
                print(rec)
        
        if medium_recommendations:
            print(f"\nMEDIUM PRIORITY RECOMMENDATIONS:")
            for rec in medium_recommendations:
                print(rec)
        
        if low_recommendations:
            print(f"\nLOW PRIORITY RECOMMENDATIONS:")
            for rec in low_recommendations:
                print(rec)
        
        if not any(rec['needs_product'] for rec in recommendations.values()):
            print(f"\nNo immediate financial product needs identified.")
            print(f"Your current financial profile appears well-balanced.")
        
        return recommendations
    
    def generate_explanations(self, user_profile, recommendations):
            """Generate simple explanations without LLM"""
            print(f"\n{'='*50}")
            print(" DETAILED EXPLANATIONS")
            print(f"{'='*60}")
            
            # Simple cluster explanation
            cluster_name = user_profile.get('consensus_cluster_name', 'Unknown')
            cluster_id = user_profile.get('consensus_cluster_id', 2)
            
            print(f"\n{cluster_name} Cluster Explanation:")
            print("-" * (len(cluster_name) + 17))
            
            # Basic explanation based on cluster
            if cluster_name == "High Income Savers":
                print(f"\nWhy you're in this group:")
                print(f"   • High income (> $100K) with strong savings capacity")
                print(f"   • Consistent investment behavior with low debt")
                print(f"   • Long-term investment perspective")
                print(f"   • Moderate to high risk tolerance")
                
            elif cluster_name == "Middle Income Families":
                print(f"\nWhy you're in this group:")
                print(f"   • Middle income ($5K-$100K) with steady cash flow")
                print(f"   • Balanced saving and spending patterns")
                print(f"   • Medium to long-term financial goals")
                print(f"   • Moderate risk tolerance")
                
            elif cluster_name == "Zero Income Households":
                print(f"\nWhy you're in this group:")
                print(f"   • No income or very low income requiring assistance")
                print(f"   • Limited savings with potential financial stress")
                print(f"   • Short-term stability needs")
                print(f"   • Very low risk tolerance")
            
            # Product explanations
            print(f"\nProduct Recommendations Explained:")
            for product, rec in recommendations.items():
                if rec['needs_product']:
                    product_name = product.replace('_', ' ').title()
                    print(f"\n{product_name}:")
                    print(f"   Reason: {rec['reason']}")
                    print(f"   Probability: {rec['probability']:.1%}")
                    print(f"   Urgency: {rec['urgency']}")
                    print(f"   Action: {rec['action']}")
            
            # Next steps
            print(f"\nRecommended Next Steps:")
            if cluster_name == "High Income Savers":
                print(f"   1. Consider premium investment services")
                print(f"   2. Explore tax optimization strategies")
                print(f"   3. Review estate planning needs")
            elif cluster_name == "Middle Income Families":
                print(f"   1. Build emergency fund (3-6 months)")
                print(f"   2. Start or increase retirement contributions")
                print(f"   3. Consider education savings plans")
            else:
                print(f"   1. Seek financial counseling services")
                print(f"   2. Explore assistance programs")
                print(f"   3. Focus on basic banking needs")
    
    def run_demo(self):
        """Run complete interactive demo"""
        try:
            # Get user input
            user_data = self.get_user_input()
            if user_data is None:
                return
            
            # Classify user
            user_profile = self.classify_user(user_data)
            
            # Generate recommendations
            recommendations = self.generate_product_recommendations(user_profile)
            
            # Generate explanations
            self.generate_explanations(user_profile, recommendations)
            
            # Final summary
            print(f"\n{'='*50}")
            print(" DEMO COMPLETE - YOUR FINANCIAL RECOMMENDATIONS")
            print(f"{'='*50}")
            
            print(f"\nSummary:")
            print(f"   Your Group: {user_profile['consensus_cluster_name']}")
            print(f"   Products Recommended: {sum(1 for r in recommendations.values() if r['needs_product'])}")
            print(f"   Urgent Actions: {sum(1 for r in recommendations.values() if r['needs_product'] and r['urgency'] == 'High')}")
            
            print(f"\nThank you for using the Financial Recommendation System!")
            
        except Exception as e:
            print(f"\nError during demo: {e}")
            print("Please try again or contact support.")

def main():
    """Main function to run the interactive demo"""
    demo = InteractiveDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
