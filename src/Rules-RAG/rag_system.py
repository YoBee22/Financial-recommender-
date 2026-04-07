"""
RAG System for Financial Recommendation System
Implements retrieval-augmented generation using ChromaDB and sentence transformers
"""

# SQLite fix for Streamlit Cloud (MUST be before chromadb import)
import os
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from google import genai
from datetime import datetime
import sys

# Disable ChromaDB telemetry to avoid capture() errors
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiEmbeddingFunction:
    """Custom ChromaDB embedding function using the new google-genai SDK."""
    
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
    
    def __call__(self, input):
        result = self.client.models.embed_content(
            model='text-embedding-004',
            contents=input
        )
        return [e.values for e in result.embeddings]


class FinancialRAGSystem:
    """RAG system for financial knowledge retrieval and generation"""
    
    def __init__(self, data_path: str = None, api_key: str = None):
        self.data_path = Path(data_path) if data_path else Path(__file__).parent.parent
        self.knowledge_base_path = self.data_path / "data" / "rag_knowledge_base"
        self.vector_store_path = self.data_path / "data" / "vector_store"
        
        # Initialize paths
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Gemini API configuration - use environment variable first, then parameter
        self.api_key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        
        # Initialize Google embedding function using new google-genai SDK
        self.embedding_function = GeminiEmbeddingFunction(api_key=self.api_key)
        
        # Initialize ChromaDB - EphemeralClient avoids SQLite tenant issues on Streamlit Cloud
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = None
        
        # Initialize Gemini client for text generation
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
        
        self.documents = []
        self.is_initialized = False
        
    def create_financial_knowledge_base(self):
        """Create comprehensive financial knowledge base"""
        logger.info("Creating financial knowledge base...")
        
        # Financial knowledge documents
        knowledge_documents = [
            {
                "id": "emergency_fund_basics",
                "content": """
                Emergency Fund Basics:
                An emergency fund is a dedicated savings account for unexpected expenses or financial emergencies.
                Recommended amount: 3-6 months of essential living expenses.
                Essential expenses include: rent/mortgage, utilities, food, transportation, insurance, minimum debt payments.
                Start with a $1,000 starter emergency fund if you're beginning.
                Keep emergency funds in a high-yield savings account for liquidity and some growth.
                Do not invest emergency funds in stocks or other volatile investments.
                Emergency fund purposes: job loss, medical emergencies, car repairs, home repairs, unexpected travel.
                """,
                "metadata": {
                    "category": "emergency_fund",
                    "priority": "high",
                    "topic": "basics"
                }
            },
            {
                "id": "retirement_planning_fundamentals",
                "content": """
                Retirement Planning Fundamentals:
                Start saving for retirement as early as possible to benefit from compound interest.
                General rule: Save 10-15% of your income for retirement.
                Retirement account types: 401(k), Traditional IRA, Roth IRA, 403(b), TSP.
                401(k): Employer-sponsored, pre-tax contributions, employer match is free money.
                Roth IRA: Post-tax contributions, tax-free withdrawals in retirement, income limits apply.
                Traditional IRA: Pre-tax contributions, taxed on withdrawal in retirement.
                Required Minimum Distributions (RMDs) start at age 73 for most retirement accounts.
                Roth IRAs have no RMDs.
                Consider target-date funds for hands-off retirement investing.
                """,
                "metadata": {
                    "category": "retirement",
                    "priority": "high",
                    "topic": "fundamentals"
                }
            },
            {
                "id": "budgeting_strategies",
                "content": """
                Budgeting Strategies:
                50/30/20 Rule: 50% for needs, 30% for wants, 20% for savings and debt repayment.
                Zero-based budgeting: Every dollar has a purpose - income minus expenses equals zero.
                Envelope system: Use cash envelopes for discretionary spending categories.
                Pay yourself first: Automatically transfer savings before spending on anything else.
                Track spending for 30 days to understand your spending patterns.
                Review and adjust your budget monthly.
                Use budgeting apps or spreadsheets to track expenses automatically.
                Include irregular expenses in your budget (car maintenance, gifts, subscriptions).
                """,
                "metadata": {
                    "category": "budgeting",
                    "priority": "high",
                    "topic": "strategies"
                }
            },
            {
                "id": "investment_basics",
                "content": """
                Investment Basics:
                Diversification: Don't put all your eggs in one basket.
                Asset allocation: Mix of stocks, bonds, and cash based on your age and risk tolerance.
                Rule of thumb: 110 minus your age = percentage in stocks.
                Stocks: Ownership in companies, higher risk, higher potential returns.
                Bonds: Loans to governments or companies, lower risk, steady income.
                ETFs: Exchange-traded funds, diversified, trade like stocks, lower fees.
                Mutual funds: Professionally managed portfolios, higher fees than ETFs.
                Index funds: Track market indices, low fees, good for beginners.
                Dollar-cost averaging: Invest fixed amount regularly regardless of market conditions.
                """,
                "metadata": {
                    "category": "investing",
                    "priority": "high",
                    "topic": "basics"
                }
            },
            {
                "id": "debt_management",
                "content": """
                Debt Management Strategies:
                Avalanche method: Pay minimums on all debts, attack highest-interest debt first.
                Snowball method: Pay minimums on all debts, attack smallest balance first.
                Good debt: Mortgage, student loans (potentially), business loans.
                Bad debt: Credit cards, payday loans, high-interest personal loans.
                Debt-to-income ratio: Keep below 36% for healthy financial profile.
                Credit card utilization: Keep below 30% of your credit limit.
                Consider debt consolidation for high-interest credit card debt.
                Avoid minimum payments on credit cards - pay much more than minimum.
                Emergency fund can prevent going into debt for unexpected expenses.
                """,
                "metadata": {
                    "category": "debt",
                    "priority": "high",
                    "topic": "management"
                }
            },
            {
                "id": "tax_planning_basics",
                "content": """
                Tax Planning Basics:
                Tax-advantaged accounts reduce your taxable income and help you save more.
                401(k) and Traditional IRA contributions are pre-tax (reduce current taxable income).
                Roth contributions are post-tax (tax-free growth and withdrawals).
                Health Savings Account (HSA): Triple tax advantage if you have high-deductible health plan.
                529 Plan: Tax-advantaged savings for education expenses.
                Standard deduction vs. itemized: Choose whichever gives you the larger tax benefit.
                Tax-loss harvesting: Sell investments at a loss to offset capital gains taxes.
                Consider tax implications when selling investments in taxable accounts.
                Keep good records of all financial transactions for tax purposes.
                """,
                "metadata": {
                    "category": "taxes",
                    "priority": "medium",
                    "topic": "planning"
                }
            },
            {
                "id": "insurance_fundamentals",
                "content": """
                Insurance Fundamentals:
                Health insurance: Protects against medical expenses, consider high-deductible plans with HSA.
                Life insurance: Term life for income replacement, whole life for estate planning.
                Disability insurance: Replaces income if you can't work due to illness or injury.
                Homeowners/renters insurance: Protects your property and belongings.
                Auto insurance: Required by law, covers liability and vehicle damage.
                Umbrella insurance: Extra liability coverage beyond standard policies.
                Insurance deductible: Higher deductible = lower premium, but more out-of-pocket costs.
                Review insurance coverage annually to ensure adequate protection.
                """,
                "metadata": {
                    "category": "insurance",
                    "priority": "medium",
                    "topic": "fundamentals"
                }
            },
            {
                "id": "home_buying_guide",
                "content": """
                Home Buying Guide:
                Down payment: Aim for 20% to avoid PMI (Private Mortgage Insurance).
                Mortgage pre-approval: Get before house hunting to know your budget.
                Closing costs: Typically 2-5% of home purchase price.
                28/36 rule: Housing costs ≤28% of gross income, total debt ≤36% of gross income.
                Fixed-rate mortgage: Predictable payments, good for long-term stability.
                Adjustable-rate mortgage: Lower initial rate, payments can increase over time.
                Home inspection: Always get one to identify potential issues.
        Consider ongoing costs: property taxes, insurance, maintenance, HOA fees.
                Build emergency fund before buying a home.
                """,
                "metadata": {
                    "category": "real_estate",
                    "priority": "medium",
                    "topic": "buying"
                }
            },
            {
                "id": "college_savings_strategies",
                "content": """
                College Savings Strategies:
                529 plans: Tax-advantaged savings for education, state-sponsored.
                Coverdell ESA: Education savings with contribution limits.
                UGMA/UTMA accounts: Custodial accounts for minors, no tax advantages.
                Start early: Compound growth works best over long time periods.
                Consider in-state public schools for cost savings.
                FAFSA: Free Application for Federal Student Aid, complete annually.
                Scholarships and grants: Free money that doesn't need repayment.
                Work-study programs: Part-time work to help pay for college.
                Student loans: Borrow as little as possible, understand repayment terms.
                """,
                "metadata": {
                    "category": "education",
                    "priority": "medium",
                    "topic": "savings"
                }
            },
            {
                "id": "financial_independence_retire_early",
                "content": """
                Financial Independence Retire Early (FIRE):
                FIRE movement: Save aggressively to achieve financial independence before traditional retirement age.
                4% rule: Withdraw 4% of retirement savings annually, adjust for inflation.
                FIRE types: Lean FIRE (minimal lifestyle), Fat FIRE (luxury lifestyle), Barista FIRE (part-time work).
                Savings rate: Critical factor - higher savings rate = faster FIRE achievement.
                Investment strategy: Typically aggressive stock market investing for growth.
                Geographic arbitrage: Live in low-cost areas to accelerate savings.
                Side hustles: Additional income streams to increase savings rate.
                Healthcare planning: Major consideration for early retirees.
                Tax optimization: Important for early retirees with limited income.
                """,
                "metadata": {
                    "category": "retirement",
                    "priority": "low",
                    "topic": "fire"
                }
            }
        ]
        
        # Add ETF and mutual fund specific knowledge
        etf_knowledge = [
            {
                "id": "etf_basics",
                "content": """
                ETF (Exchange-Traded Fund) Basics:
                ETFs are investment funds traded on stock exchanges like individual stocks.
                Advantages: Diversification, lower fees than mutual funds, intraday trading, tax efficiency.
                Types of ETFs: Index ETFs, sector ETFs, international ETFs, bond ETFs, commodity ETFs.
                Index ETFs track market indices like S&P 500, NASDAQ, or specific sectors.
                Expense ratios: Annual fees, typically 0.03% to 0.75% for ETFs.
                Trading: Buy and sell throughout the day at market prices.
                Tax efficiency: ETFs typically generate fewer capital gains distributions than mutual funds.
                Creation/redemption process: Authorized participants create/redeem ETF shares.
                """,
                "metadata": {
                    "category": "etf",
                    "priority": "high",
                    "topic": "basics"
                }
            },
            {
                "id": "mutual_fund_basics",
                "content": """
                Mutual Fund Basics:
                Mutual funds pool money from many investors to purchase securities.
                Professional management: Fund managers make investment decisions.
                NAV (Net Asset Value): Calculated once daily after market close.
                Minimum investments: Often $1,000 to $3,000, some as low as $100.
                Types: Equity funds, bond funds, balanced funds, money market funds.
                Load fees: Some funds charge sales commissions (front-end or back-end loads).
                No-load funds: No sales commissions, increasingly common.
                Expense ratios: Annual fees, typically 0.5% to 1.5% for actively managed funds.
                Index funds: Passively managed, track market indices, lower fees.
                """,
                "metadata": {
                    "category": "mutual_funds",
                    "priority": "high",
                    "topic": "basics"
                }
            },
            {
                "id": "etf_vs_mutual_funds",
                "content": """
                ETFs vs Mutual Funds Comparison:
                Trading: ETFs trade intraday like stocks, mutual funds price once daily at NAV.
                Minimum investment: ETFs - share price, mutual funds - often $1,000+ minimum.
                Expense ratios: ETFs generally lower (0.03-0.75%), mutual funds higher (0.5-1.5%).
                Tax efficiency: ETFs typically more tax efficient due to creation/redemption process.
                Trading costs: ETFs may incur brokerage commissions, mutual funds often commission-free.
                Automatic investing: Mutual funds easier for automatic investments, ETFs require market orders.
                Fractional shares: Some mutual funds allow fractional purchases, ETFs typically whole shares.
                Premium/discount: ETFs may trade at premium/discount to NAV, mutual funds always at NAV.
                """,
                "metadata": {
                    "category": "comparison",
                    "priority": "high",
                    "topic": "etf_vs_mutual_funds"
                }
            }
        ]
        
        # Combine all knowledge documents
        all_documents = knowledge_documents + etf_knowledge
        
        # Save knowledge base to file
        knowledge_base_file = self.knowledge_base_path / "financial_knowledge.json"
        with open(knowledge_base_file, 'w') as f:
            json.dump(all_documents, f, indent=2)
        
        self.documents = all_documents
        logger.info(f"Created knowledge base with {len(all_documents)} documents")
        
        return all_documents
    
    def initialize_vector_store(self):
        """Initialize ChromaDB vector store with documents"""
        logger.info("Initializing vector store...")
        
        # Create or get collection with Google embedding function
        self.collection = self.chroma_client.get_or_create_collection(
            name="financial_knowledge",
            metadata={"description": "Financial knowledge base for RAG system"},
            embedding_function=self.embedding_function
        )
              
        # Prepare documents for embedding
        if not self.documents:
            self.create_financial_knowledge_base()
        
        documents = []
        metadatas = []
        ids = []
        
        for doc in self.documents:
            documents.append(doc["content"])
            metadatas.append(doc["metadata"])
            ids.append(doc["id"])
        
        # Add to ChromaDB (embedding function handles encoding automatically)
        logger.info("Adding documents to vector store...")
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        self.is_initialized = True
        logger.info(f"Vector store initialized with {len(documents)} documents")
    
    def query_knowledge_base(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Query the knowledge base for relevant information"""
        if not self.is_initialized:
            self.initialize_vector_store()
        
        # Search ChromaDB (embedding function handles query encoding)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else 0
            })
        
        return formatted_results
    
    def generate_contextual_response(self, query: str, user_profile: Dict = None) -> str:
        """Generate contextual response using RAG"""
        # Retrieve relevant documents
        relevant_docs = self.query_knowledge_base(query, n_results=3)
        
        if not relevant_docs:
            return "I don't have specific information about that topic. Please consult with a financial advisor for personalized advice."
        
        # Build context
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        
        # Build system prompt
        system_prompt = f"""
        You are a helpful financial advisor assistant. Use the provided context to answer the user's question.
        Provide clear, practical advice and always include a disclaimer that this is not professional financial advice.
        
        Context:
        {context}
        
        User Profile: {user_profile if user_profile else "Not provided"}
        
        Question: {query}
        
        Provide a comprehensive answer based on the context. Include specific, actionable advice.
        """
        
        try:
            # Generate response using Gemini
            if self.client:
                response = self.client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=f"""You are a helpful financial advisor assistant. Use the provided context to answer the user's question.
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
    
    def get_relevant_topics(self, query: str) -> List[str]:
        """Get relevant topics for a query"""
        relevant_docs = self.query_knowledge_base(query, n_results=5)
        topics = list(set([doc["metadata"].get("category", "general") for doc in relevant_docs]))
        return topics
    
    def expand_knowledge_base(self, new_documents: List[Dict[str, Any]]):
        """Add new documents to the knowledge base"""
        if not self.is_initialized:
            self.initialize_vector_store()
        
        documents = []
        metadatas = []
        ids = []
        
        for doc in new_documents:
            documents.append(doc["content"])
            metadatas.append(doc["metadata"])
            ids.append(doc["id"])
        
        # Add to ChromaDB (embedding function handles encoding automatically)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(new_documents)} new documents to knowledge base")
    
    def save_knowledge_base(self):
        """Save knowledge base to file"""
        knowledge_base_file = self.knowledge_base_path / "financial_knowledge.json"
        with open(knowledge_base_file, 'w') as f:
            json.dump(self.documents, f, indent=2)
        logger.info(f"Knowledge base saved to {knowledge_base_file}")
    
    def load_knowledge_base(self):
        """Load knowledge base from file"""
        knowledge_base_file = self.knowledge_base_path / "financial_knowledge.json"
        if knowledge_base_file.exists():
            with open(knowledge_base_file, 'r') as f:
                self.documents = json.load(f)
            logger.info(f"Loaded {len(self.documents)} documents from knowledge base")
            return True
        return False

def main():
    """Test the RAG system"""
    # Initialize RAG system
    rag = FinancialRAGSystem()
    
    # Create knowledge base
    rag.create_financial_knowledge_base()
    
    # Initialize vector store
    rag.initialize_vector_store()
    
    # Test queries
    test_queries = [
        "How much should I save for emergency fund?",
        "What's the difference between ETF and mutual fund?",
        "How do I start investing for retirement?",
        "What's the best budgeting strategy?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag.generate_contextual_response(query)
        print(f"Response: {result[:200]}...")
        print("-" * 80)

if __name__ == "__main__":
    main()