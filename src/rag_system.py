"""
Financial RAG System - Retrieval-Augmented Generation for Financial Advice
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from datetime import datetime
import sys

# Disable ChromaDB telemetry to avoid capture() errors
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'

# Also disable posthog telemetry
os.environ['POSTHOG_DISABLE'] = 'True'
os.environ['POSTHOG_HOST'] = ''

# Setup logging with ChromaDB telemetry suppression
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialRAGSystem:
    """RAG system for financial knowledge retrieval and generation"""
    
    def __init__(self, data_path: str = None, api_key: str = None):
        self.data_path = Path(data_path) if data_path else Path(__file__).parent.parent
        self.knowledge_base_path = self.data_path / "data" / "rag_knowledge_base"
        self.vector_store_path = self.data_path / "data" / "vector_store"
        
        # Initialize paths
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB - use in-memory client for Streamlit Cloud compatibility
        self.chroma_client = chromadb.Client()
        self.collection = None
        
        # Gemini API configuration - use environment variable first, then parameter
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        
        self.documents = []
        self.is_initialized = False
    
    def create_financial_knowledge_base(self) -> List[Dict]:
        """Create a comprehensive financial knowledge base"""
        logger.info("Creating financial knowledge base...")
        
        knowledge_base = [
            {
                "id": "emergency_fund_basics",
                "title": "Emergency Fund Basics",
                "category": "savings",
                "content": """
                An emergency fund is a crucial component of financial security. It should cover 3-6 months of living expenses
                and be kept in a liquid, easily accessible account like a high-yield savings account. This fund protects you
                from unexpected expenses, job loss, or medical emergencies without requiring you to take on debt.
                
                Key points:
                - Start with $1,000 as a beginner goal
                - Build up to 3-6 months of expenses
                - Keep in high-yield savings or money market account
                - Only use for true emergencies
                - Replenish after withdrawals
                """
            },
            {
                "id": "retirement_planning_fundamentals",
                "title": "Retirement Planning Fundamentals",
                "category": "retirement",
                "content": """
                Retirement planning should start as early as possible to take advantage of compound interest.
                The general rule is to save 10-15% of your income for retirement, but more is better if possible.
                
                Retirement account options:
                - 401(k): Employer-sponsored, often with matching
                - IRA: Individual Retirement Account (Traditional or Roth)
                - Roth IRA: Post-tax contributions, tax-free withdrawals
                - Traditional IRA: Pre-tax contributions, taxed withdrawals
                
                Target retirement savings by age:
                - Age 30: 1x annual salary
                - Age 40: 3x annual salary
                - Age 50: 6x annual salary
                - Age 67: 10x annual salary
                """
            },
            {
                "id": "budgeting_strategies",
                "title": "Effective Budgeting Strategies",
                "category": "budgeting",
                "content": """
                Budgeting is the foundation of financial success. Popular methods include:
                
                50/30/20 Rule:
                - 50% for needs (housing, food, utilities, transportation)
                - 30% for wants (entertainment, dining out, hobbies)
                - 20% for savings and debt repayment
                
                Zero-Based Budget:
                - Every dollar has a purpose
                - Income minus expenses equals zero
                - Requires detailed tracking
                
                Envelope System:
                - Physical or digital envelopes for categories
                - When envelope is empty, spending stops
                - Great for visual learners and overspenders
                
                Budget tracking tools:
                - Apps like Mint, YNAB, or Personal Capital
                - Spreadsheets for custom tracking
                - Bank's built-in budgeting features
                """
            },
            {
                "id": "investment_basics",
                "title": "Investment Fundamentals",
                "category": "investing",
                "content": """
                Investing is essential for building long-term wealth. Key principles:
                
                Risk and Return:
                - Higher potential returns usually mean higher risk
                - Diversification helps manage risk
                - Time horizon affects appropriate risk level
                
                Investment Types:
                - Stocks: Ownership in companies, high risk/return
                - Bonds: Loans to governments/companies, lower risk/return
                - ETFs: Baskets of securities, instant diversification
                - Mutual Funds: Professionally managed portfolios
                - Real Estate: Property investment, can provide rental income
                
                Investment Accounts:
                - Taxable brokerage account
                - Retirement accounts (401k, IRA)
                - Education savings (529 plan)
                - Health savings (HSA)
                
                Starting Tips:
                - Begin with low-cost index funds or ETFs
                - Invest consistently (dollar-cost averaging)
                - Reinvest dividends
                - Review and rebalance quarterly
                """
            },
            {
                "id": "debt_management",
                "title": "Debt Management Strategies",
                "category": "debt",
                "content": """
                Effective debt management is crucial for financial health. Strategies include:
                
                Debt Avalanche Method:
                - Pay minimums on all debts
                - Attack highest interest rate debt first
                - Most mathematically efficient
                - Saves most money on interest
                
                Debt Snowball Method:
                - Pay minimums on all debts
                - Attack smallest balance first
                - Provides psychological wins
                - Builds momentum
                
                Good Debt vs Bad Debt:
                Good Debt:
                - Mortgage (builds equity)
                - Student loans (invests in human capital)
                - Business loans (potential for income growth)
                
                Bad Debt:
                - Credit card debt (high interest)
                - Personal loans (consumption)
                - Payday loans (predatory rates)
                
                Debt-to-Income Ratio:
                - Below 36% is considered healthy
                - Below 28% is excellent
                - Above 43% may indicate financial stress
                """
            },
            {
                "id": "tax_planning_basics",
                "title": "Tax Planning Fundamentals",
                "category": "taxes",
                "content": """
                Tax planning can significantly impact your net worth. Key strategies:
                
                Tax-Advantaged Accounts:
                - 401(k): Pre-tax contributions, reduces taxable income
                - Traditional IRA: Tax-deductible contributions
                - Roth IRA: Tax-free growth and withdrawals
                - HSA: Triple tax advantage (contributions, growth, withdrawals)
                - 529 Plan: Tax-free education savings
                
                Tax Deductions and Credits:
                - Standard deduction vs itemized deduction
                - Mortgage interest deduction
                - Student loan interest deduction
                - Child tax credit
                - Earned Income Tax Credit
                
                Tax-Loss Harvesting:
                - Sell investments at a loss
                - Offset capital gains
                - Deduct up to $3,000 in ordinary income
                - Buy similar investment after 30 days
                
                Retirement Withdrawal Strategy:
                - Traditional accounts taxed as ordinary income
                - Roth accounts tax-free
                - Consider tax bracket in retirement
                - Required Minimum Distributions (RMDs) at age 72
                """
            },
            {
                "id": "insurance_fundamentals",
                "title": "Insurance Fundamentals",
                "category": "insurance",
                "content": """
                Insurance protects against financial catastrophes. Essential types:
                
                Health Insurance:
                - Covers medical expenses
                - Prevents medical bankruptcy
                - Consider high-deductible plans with HSA
                - Check network coverage
                
                Life Insurance:
                - Term life: Pure protection, affordable
                - Whole life: Investment component, expensive
                - Rule of thumb: 10x annual income
                - More important if you have dependents
                
                Disability Insurance:
                - Protects income if you can't work
                - More likely than premature death
                - Short-term and long-term coverage
                - Often available through employers
                
                Auto Insurance:
                - Liability coverage required by law
                - Comprehensive and collision optional
                - Consider umbrella policy for extra protection
                - Shop around for best rates
                
                Homeowners/Renters Insurance:
                - Protects dwelling and personal property
                - Liability coverage for injuries on property
                - Renters insurance covers personal property only
                - Document possessions for claims
                """
            },
            {
                "id": "home_buying_guide",
                "title": "Home Buying Guide",
                "category": "housing",
                "content": """
                Home buying is a major financial decision. Key considerations:
                
                Financial Preparation:
                - Credit score of 740+ for best rates
                - 20% down payment avoids PMI
                - 3-5% for closing costs
                - Debt-to-income ratio below 43%
                
                The Home Buying Process:
                1. Get pre-approved for mortgage
                2. Find a real estate agent
                3. Search for homes
                4. Make an offer
                5. Get home inspection
                6. Secure financing
                7. Close on the home
                
                Mortgage Types:
                - Fixed-rate: Stable payments, predictable
                - Adjustable-rate: Lower initial rate, risky
                - FHA: Lower down payment, mortgage insurance
                - VA: No down payment, for veterans
                
                Ongoing Costs:
                - Property taxes (1-2% of home value annually)
                - Homeowners insurance
                - Maintenance (1-4% of home value annually)
                - HOA fees (if applicable)
                - Utilities
                """
            },
            {
                "id": "college_savings_strategies",
                "title": "College Savings Strategies",
                "category": "education",
                "content": """
                College costs continue to rise, making early planning essential.
                
                529 Education Savings Plans:
                - Tax-free growth and withdrawals
                - State-sponsored but portable nationwide
                - No income limits for contributions
                - High contribution limits
                
                Coverdell Education Savings Account:
                - $2,000 annual contribution limit
                - Can fund K-12 expenses
                - Income limits apply
                - More investment options
                
                Other Strategies:
                - UGMA/UTMA custodial accounts
                - Roth IRA (for working teens)
                - Permanent life insurance
                - Scholarships and grants
                
                College Funding Timeline:
                - Birth: Start 529 plan
                - Ages 0-10: Aggressive growth investments
                - Ages 11-15: Moderate risk investments
                - Ages 16-18: Conservative investments
                - College: Use 529 funds tax-free
                
                FAFSA and Financial Aid:
                - Complete annually for financial aid
                - Expected Family Contribution (EFC)
                - Need-based vs merit-based aid
                - Compare total cost of attendance
                """
            },
            {
                "id": "financial_independence_retire_early",
                "title": "Financial Independence & Retire Early (FIRE)",
                "category": "retirement",
                "content": """
                FIRE movement focuses on aggressive saving and investing for early retirement.
                
                FIRE Principles:
                - Save 50%+ of income
                - Invest in low-cost index funds
                - Minimize lifestyle inflation
                - Focus on financial independence rather than retirement
                
                FIRE Variations:
                Lean FIRE:
                - Extreme frugality
                - Retire on minimal expenses
                - $500k-$1M needed
                
                Standard FIRE:
                - Traditional retirement lifestyle
                - $1M-$2M needed
                - 4% safe withdrawal rate
                
                Fat FIRE:
                - Luxurious retirement lifestyle
                - $2M+ needed
                - Higher spending in retirement
                
                Barista FIRE:
                - Part-time work in retirement
                - Less savings needed
                - Health insurance coverage
                
                Math Behind FIRE:
                - Annual expenses × 25 = retirement number
                - Based on 4% safe withdrawal rate
                - Adjust for inflation and market returns
                - Consider healthcare costs
                """
            },
            {
                "id": "etf_basics",
                "title": "ETF Basics",
                "category": "investing",
                "content": """
                Exchange-Traded Funds (ETFs) are investment funds traded on stock exchanges.
                
                ETF Advantages:
                - Instant diversification
                - Lower expenses than mutual funds
                - Trade throughout the day like stocks
                - Tax-efficient structure
                - Transparent holdings
                
                ETF Types:
                - Index ETFs: Track market indices
                - Sector ETFs: Focus on specific industries
                - International ETFs: Foreign market exposure
                - Bond ETFs: Fixed income investments
                - Commodity ETFs: Gold, oil, other commodities
                
                Popular ETF Examples:
                - SPY: S&P 500 Index
                - QQQ: Nasdaq 100 Index
                - VTI: Total Stock Market
                - BND: Total Bond Market
                - VOO: S&P 500 (Vanguard)
                
                ETF Considerations:
                - Expense ratios impact returns
                - Tracking error vs index
                - Bid-ask spreads affect trading costs
                - Premium/discount to NAV
                - Tax efficiency varies by type
                """
            },
            {
                "id": "mutual_fund_basics",
                "title": "Mutual Fund Basics",
                "category": "investing",
                "content": """
                Mutual funds pool money from many investors to purchase securities.
                
                Mutual Fund Advantages:
                - Professional management
                - Diversification
                - Accessibility with small amounts
                - Systematic investment plans
                - Regulatory oversight
                
                Mutual Fund Types:
                - Equity funds: Stock investments
                - Bond funds: Fixed income
                - Balanced funds: Mix of stocks and bonds
                - Money market funds: Short-term debt
                - Index funds: Track market indices
                
                Load vs No-Load Funds:
                - Load funds: Sales commission (front-end or back-end)
                - No-load funds: No sales commission
                - Class A, B, C shares have different fee structures
                - No-load funds generally better for most investors
                
                Mutual Fund Considerations:
                - Expense ratios eat into returns
                - Minimum investment requirements
                - Trading only once daily at NAV
                - Capital gains distributions
                - Tax efficiency varies by type
                
                Choosing Mutual Funds:
                - Check expense ratios
                - Review fund manager track record
                - Understand investment strategy
                - Consider tax implications
                - Compare to ETF alternatives
                """
            },
            {
                "id": "etf_vs_mutual_funds",
                "title": "ETFs vs Mutual Funds",
                "category": "investing",
                "content": """
                ETFs and mutual funds both offer diversification, but have key differences.
                
                Trading Differences:
                ETFs:
                - Trade throughout the day like stocks
                - Real-time pricing
                - Can use limit orders
                - Short selling and options available
                
                Mutual Funds:
                - Trade once daily at closing NAV
                - End-of-day pricing
                - No limit orders
                - Cannot short sell
                
                Cost Differences:
                ETFs:
                - Generally lower expense ratios
                - No minimum investment
                - Trading commissions (though many brokers offer free trades)
                - Bid-ask spreads
                
                Mutual Funds:
                - Higher expense ratios (especially actively managed)
                - Minimum investment requirements
                - No trading commissions
                - Potential 12b-1 fees
                
                Tax Efficiency:
                ETFs:
                - More tax-efficient structure
                - Fewer capital gains distributions
                - In-kind redemptions minimize taxable events
                
                Mutual Funds:
                - Less tax-efficient
                - Annual capital gains distributions
                - Forced sales when investors redeem shares
                
                When to Choose ETFs:
                - Taxable accounts
                - Frequent trading
                - Specific sector exposure
                - Lower costs preferred
                
                When to Choose Mutual Funds:
                - Retirement accounts
                - Dollar-cost averaging
                - Active management desired
                - No trading during day preferred
                """
            }
        ]
        
        # Save knowledge base to file
        knowledge_file = self.knowledge_base_path / "financial_knowledge.json"
        with open(knowledge_file, 'w') as f:
            json.dump(knowledge_base, f, indent=2)
        
        self.documents = knowledge_base
        self.is_initialized = True
        
        logger.info(f"Created knowledge base with {len(knowledge_base)} documents")
        return knowledge_base
    
    def initialize_vector_store(self):
        """Initialize ChromaDB vector store with documents"""
        logger.info("Initializing vector store...")
        
        # Create or get collection using new API
        self.collection = self.chroma_client.get_or_create_collection(
            name="financial_knowledge",
            metadata={"description": "Financial knowledge base for RAG system"}
        )
                
        # Prepare documents for embedding
        if not self.documents:
            self.create_financial_knowledge_base()
        
        documents = []
        metadatas = []
        ids = []
        
        for doc in self.documents:
            documents.append(doc["content"])
            metadatas.append({
                "title": doc["title"],
                "category": doc["category"],
                "id": doc["id"]
            })
            ids.append(doc["id"])
        
        # Create embeddings
        logger.info("Creating embeddings...")
        embeddings = self.embedding_model.encode(documents)
        
        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )
        
        logger.info(f"Vector store initialized with {len(documents)} documents")
    
    def query_knowledge_base(self, query: str, n_results: int = 3) -> List[Dict]:
        """Query the knowledge base for relevant documents"""
        if not self.collection:
            self.initialize_vector_store()
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        # Format results
        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return documents
    
    def generate_contextual_response(self, query: str, user_profile: Dict = None) -> str:
        """
        Generates a contextual response using the retrieved documents and an LLM.
        If no LLM API key is available, it provides a context-based response.
        """
        # Retrieve relevant documents
        relevant_docs = self.query_knowledge_base(query)
        
        context = "\n".join([doc["document"] for doc in relevant_docs])
        
        # Prepare the prompt for the LLM
        system_prompt = f"""You are a helpful financial advisor assistant. Use the provided context to answer the user's question.
        Provide clear, practical advice and always include a disclaimer that this is not professional financial advice.
        
        Context:
        {context}
        
        User Profile: {user_profile if user_profile else "Not provided"}
        
        Question: {query}
        
        Provide a comprehensive answer based on the context. Include specific, actionable advice."""
        
        try:
            # Generate response using Gemini
            if self.api_key:
                response = self.model.generate_content(
                    f"""You are a helpful financial advisor assistant. Use the provided context to answer the user's question.
                    Provide clear, practical advice and always include a disclaimer that this is not professional financial advice.
                    
                    Context:
                    {context}
                    
                    User Profile: {user_profile if user_profile else "Not provided"}
                    
                    Question: {query}
                    
                    Provide a comprehensive answer based on the context. Include specific, actionable advice."""
                )
                
                answer = response.text
                
                # Add disclaimer
                disclaimer = "\n\n*Note: This information is for educational purposes only and should not be considered as professional financial advice. Please consult with a qualified financial advisor for personalized guidance.*"
                
                return answer + disclaimer
            else:
                # No API key available, use fallback
                logger.warning("No Gemini API key available, using fallback response")
                raise Exception("No API key")
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Fallback to context-based response without AI generation
            return f"""Based on the financial knowledge I found:

{context}

*Note: This information is for educational purposes only and should not be considered as professional financial advice. Please consult with a qualified financial advisor for personalized guidance.*"""
