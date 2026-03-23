"""
Lightweight Streamlit Chatbot Interface
Optimized for lower memory usage with fixed syntax
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

# Lightweight imports - only load what we need
from cluster_mapping import ClusterMapper

# Page configuration
st.set_page_config(
    page_title="Financial Advisor Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for better UI
st.markdown("""
<style>
    
    /* Remove default margin and padding from body and html */
    html, body {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
    }
    
    /* Main container styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        padding: 1rem;
        margin: 0;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Message container */
    .chat-container {
        max-height: 300px;
        overflow-y: auto;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 15px;
        margin: 0.2rem 0;
    }
    
    /* Enhanced message styling */
    .message {
        padding: 0.6rem 1rem;
        border-radius: 20px;
        margin: 0.3rem 0;
        max-width: 75%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .message:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    /* User message styling */
    .user-message {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        margin-left: auto;
        text-align: right;
        border-bottom-right-radius: 5px;
    }
    
    .user-message strong {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Bot message styling */
    .bot-message {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: #2c3e50;
        margin-right: auto;
        border: 2px solid #e9ecef;
        border-bottom-left-radius: 5px;
    }
    
    .bot-message strong {
        color: #56ab2f;
        font-weight: 600;
    }
    .high-priority {
        border-left: 4px solid #e74c3c;
        background: linear-gradient(135deg, #ffebee 0%, #ffffff 100%);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(231, 76, 60, 0.1);
    }
    
    .medium-priority {
        border-left: 4px solid #f39c12;
        background: linear-gradient(135deg, #fff3e0 0%, #ffffff 100%);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(243, 156, 18, 0.1);
    }
    
    .low-priority {
        border-left: 4px solid #27ae60;
        background: linear-gradient(135deg, #e8f5e8 0%, #ffffff 100%);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(39, 174, 96, 0.1);
    }
    
    /* Input area styling */
    .input-container {
        background: white;
        padding: 0;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(86, 171, 47, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(86, 171, 47, 0.4);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #56ab2f;
        box-shadow: 0 0 0 3px rgba(86, 171, 47, 0.1);
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background: #f8f9fa;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #56ab2f;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
        }
        30% {
            transform: translateY(-10px);
        }
    }
    
    /* Help text styling */
    .help-text {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #56ab2f;
        margin-top: 1rem;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #56ab2f;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #4a9628;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        color: #495057;
        text-align: center;
        padding: 1rem;
        margin: 0;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        width: 100%;
        font-size: 0.9rem;
        box-shadow: 0 -5px 20px rgba(0,0,0,0.1);
        z-index: 999;
    }
    
    .footer p {
        margin: 0;
        opacity: 0.9;
    }
    
    /* Main content area with fixed header and footer */
    .main-content {
        padding-top: 5px;
        padding-bottom: 80px;
        min-height: 100vh;
        box-sizing: border-box;
        margin-top: 0;
    }
    
    /* Remove Streamlit default spacing */
    .main-content > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .main-content .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .main-content .stContainer {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Hide Streamlit's default elements */
    .stApp > header {
        display: none !important;
    }
    
    .stApp > footer {
        display: none !important;
    }
    
    /* Ensure full width for main content */
    .stApp > div {
        padding: 0 !important;
    }
    
    .stMainBlock {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

class LiteFinancialChatbot:
    """Lightweight chatbot interface for financial recommendations"""
    
    def __init__(self):
        self.mapper = ClusterMapper()
        
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            self._add_bot_message("Hello! I'm your financial advisor bot FinWise. I can help you understand your financial situation and provide personalized recommendations.")
            self._add_bot_message("What is your annual household income? (Please enter a number like 50000, 75000, 120000)")
        
        if 'user_data' not in st.session_state:
            st.session_state.user_data = {}
        
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 'income'
        
        if 'classification_done' not in st.session_state:
            st.session_state.classification_done = False
    
    def _add_bot_message(self, content):
        """Add a bot message to the chat"""
        st.session_state.messages.append({
            'role': 'bot',
            'content': content,
            'timestamp': datetime.now()
        })
    
    def _add_user_message(self, content):
        """Add a user message to the chat"""
        st.session_state.messages.append({
            'role': 'user',
            'content': content,
            'timestamp': datetime.now()
        })

    def _display_messages(self):
        """Display all chat messages with enhanced styling"""
        if not st.session_state.messages:
            return
        
        # Create chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.messages):
            timestamp = message['timestamp'].strftime("%I:%M %p") if hasattr(message['timestamp'], 'strftime') else ""
            
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="message user-message">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong style="color: white;">You</strong>
                        <span class="timestamp" style="color: rgba(255,255,255,0.8);">{timestamp}</span>
                    </div>
                    <div style="color: white; line-height: 1.5;">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Escape HTML content in bot messages to prevent interference
                import html
                safe_content = html.escape(str(message['content']))
                
                st.markdown(f"""
                <div class="message bot-message">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong style="display: flex; align-items: center;">
                            Financial Advisor (FinWise)
                        </strong>
                        <span class="timestamp">{timestamp}</span>
                    </div>
                    <div style="color: #2c3e50; line-height: 1.5; white-space: pre-wrap;">{safe_content}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    def _parse_number(self, text):
        """Parse a number from user input"""
        # Remove common formatting characters
        cleaned = text.replace('$', '').replace(',', '').replace(' ', '').strip()
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def _process_user_input(self, user_input):
        """Process user input and update conversation"""
        try:
            # Parse input based on current step
            if st.session_state.current_step == 'income':
                income = self._parse_number(user_input)
                if income is not None and income >= 0:
                    st.session_state.user_data['income'] = income
                    st.session_state.current_step = 'family'
                    self._add_bot_message(f"Got it! Your annual income is ${income:,.0f}. Now, let me know about your family size.")
                else:
                    self._add_bot_message("I couldn't understand that number. Please enter your annual income as a number (e.g., 75000).")
            
            elif st.session_state.current_step == 'family':
                family_size = self._parse_number(user_input)
                if family_size is not None and 1 <= family_size <= 10:
                    st.session_state.user_data['family_size'] = int(family_size)
                    st.session_state.current_step = 'expenses'
                    self._add_bot_message(f"Thanks! You have {int(family_size)} people in your household. Finally, what are your annual expenses?")
                else:
                    self._add_bot_message("Please enter a number between 1 and 10 for family size.")
            
            elif st.session_state.current_step == 'expenses':
                expenses = self._parse_number(user_input)
                if expenses is not None and expenses >= 0:
                    st.session_state.user_data['expenses'] = expenses
                    st.session_state.current_step = 'complete'
                    self._provide_analysis()
                else:
                    self._add_bot_message("I couldn't understand that number. Please enter your annual expenses as a number (e.g., 60000).")
        
        except Exception as e:
            self._add_bot_message("I had trouble understanding that. Could you please try again with a simple number?")
    
    def _provide_analysis(self):
        """Provide financial analysis and recommendations"""
        income = st.session_state.user_data['income']
        family_size = st.session_state.user_data['family_size']
        expenses = st.session_state.user_data['expenses']
        
        # Calculate savings rate
        savings_rate = (income - expenses) / income if income > 0 else -1
        
        # Classify income bracket
        if income == 0 or income < 5000:
            bracket = "Zero Income Households"
            bracket_id = 1
            description = "No income or very low income requiring assistance"
        elif income > 150000 and savings_rate > 0.15:
            bracket = "High Income Savers"
            bracket_id = 0
            description = "High income with strong savings capacity"
        else:
            bracket = "Middle Income Families"
            bracket_id = 2
            description = "Middle income with steady cash flow"
        
        # Store classification
        st.session_state.user_data.update({
            'income_bracket': bracket,
            'savings_rate': savings_rate,
            'classification_done': True
        })
        
        # Send analysis message
        analysis_message = f"""
**Your Financial Analysis:**

**Income:** ${income:,.0f}
**Family Size:** {family_size} people
**Annual Expenses:** ${expenses:,.0f}
**Savings Rate:** {savings_rate:.1%}

**Your Income Bracket:** {bracket}
*{description}*

Let me provide you with personalized recommendations based on your situation...
        """
        self._add_bot_message(analysis_message)
        
        # Generate and display recommendations (simplified)
        recommendations = self._generate_simple_recommendations()
        
        for rec in recommendations:
            priority_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
            emoji = priority_emoji.get(rec['priority'], '📋')
            
            rec_message = f"""
{emoji} **{rec['type']} ({rec['priority']} Priority)**

**Action:** {rec['action']}
**Reason:** {rec['reason']}
**Details:** {rec['details']}
            """
            self._add_bot_message(rec_message)
        
        # Send follow-up message
        follow_up = f"""
**Summary:**
• Total recommendations: {len(recommendations)}
• High priority actions: {len([r for r in recommendations if r['priority'] == 'High'])}
• Your savings status: {'Needs Improvement' if savings_rate < 0.1 else 'Good'}

**Next Steps:**
1. Focus on high priority recommendations first
2. Create a budget to track income and expenses
3. Set financial goals based on your income bracket
4. Review your situation every 3-6 months

Would you like to ask me anything specific about your financial situation or recommendations?
        """
        self._add_bot_message(follow_up)
        
        st.session_state.classification_done = True
    
    def _generate_simple_recommendations(self):
        """Generate simplified recommendations without heavy ETF/MF data"""
        bracket = st.session_state.user_data.get('income_bracket', 'Middle Income Families')
        savings_rate = st.session_state.user_data.get('savings_rate', 0)
        income = st.session_state.user_data.get('income', 0)
        
        recommendations = []
        
        # Base recommendations by bracket
        if bracket == "High Income Savers":
            recommendations = [
                {
                    'type': 'Investment',
                    'priority': 'High',
                    'action': 'Diversified investment portfolio',
                    'reason': 'High income with good savings capacity',
                    'details': 'Consider low-cost index funds, ETFs, and retirement accounts'
                },
                {
                    'type': 'Tax Planning',
                    'priority': 'Medium', 
                    'action': 'Tax optimization strategies',
                    'reason': 'High income bracket benefits from tax planning',
                    'details': 'Consider tax-advantaged accounts and deductions'
                }
            ]
        elif bracket == "Middle Income Families":
            recommendations = [
                {
                    'type': 'Emergency Fund',
                    'priority': 'High',
                    'action': 'Build emergency fund (3-6 months)',
                    'reason': 'Financial security for family',
                    'details': 'High-yield savings account for easy access'
                },
                {
                    'type': 'Retirement',
                    'priority': 'High',
                    'action': 'Start or increase retirement contributions',
                    'reason': 'Long-term financial security',
                    'details': '401(k), IRA, or other retirement accounts'
                }
            ]
        else:  # Zero Income Households
            recommendations = [
                {
                    'type': 'Basic Banking',
                    'priority': 'High',
                    'action': 'Establish basic banking services',
                    'reason': 'Financial foundation',
                    'details': 'Checking and savings accounts'
                },
                {
                    'type': 'Assistance',
                    'priority': 'High',
                    'action': 'Explore assistance programs',
                    'reason': 'Income support needs',
                    'details': 'Government programs, community resources'
                }
            ]
        
        # Add simple investment recommendation
        if income > 25000:
            recommendations.append({
                'type': 'Investment',
                'priority': 'High',
                'action': 'Start with index fund investing',
                'reason': 'Long-term wealth building',
                'details': 'Low-cost S&P 500 index fund with automatic investments'
            })
        
        # Add savings recommendation if needed
        if savings_rate < 0.1 and income > 0:
            savings_rec = {
                'type': 'Savings',
                'priority': 'High',
                'action': 'Improve savings rate',
                'reason': f'Low savings rate ({savings_rate:.1%})',
                'details': 'Reduce expenses or increase income by at least 10%'
            }
            recommendations.insert(0, savings_rec)
        
        return recommendations
    
    def _handle_follow_up_questions(self, user_input):
        """Handle follow-up questions after analysis"""
        user_input_lower = user_input.lower()
        
        if 'emergency' in user_input_lower or 'fund' in user_input_lower:
            expenses = st.session_state.user_data.get('expenses', 0)
            self._add_bot_message(f"An emergency fund should cover 3-6 months of expenses. For your situation, that would be ${expenses * 0.25:,.0f} to ${expenses * 0.5:,.0f}. Start with a smaller goal of $1,000 and build up gradually.")
        elif 'retirement' in user_input_lower or '401' in user_input_lower or 'ira' in user_input_lower:
            income = st.session_state.user_data.get('income', 0)
            self._add_bot_message(f"Start with 10-15% of your income for retirement. For your income of ${income:,.0f}, that's ${income * 0.1:,.0f} to ${income * 0.15:,.0f} per year. Take advantage of any employer matching first!")
        elif 'invest' in user_input_lower or 'etf' in user_input_lower or 'fund' in user_input_lower:
            self._add_bot_message("Based on your risk profile, consider starting with low-cost index funds. They provide diversification and have lower fees. You can start with as little as $500-$1,000.")
        elif 'budget' in user_input_lower or 'save' in user_input_lower:
            income = st.session_state.user_data.get('income', 0)
            self._add_bot_message(f"Try the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings. With your income, that means ${income * 0.2 / 12:,.0f} for savings monthly. Track your spending for a month to see where your money goes.")
        elif 'restart' in user_input_lower or 'again' in user_input_lower:
            self._restart_chat()
        else:
            self._add_bot_message("That's a great question! Based on your financial profile, I'd recommend focusing on high priority items I mentioned earlier. Is there something specific about your financial situation you'd like to discuss?")
    
    def _restart_chat(self):
        """Restart chat conversation"""
        st.session_state.messages = []
        st.session_state.user_data = {}
        st.session_state.current_step = 'income'
        st.session_state.classification_done = False
        self._add_bot_message("Chat cleared! Starting fresh conversation.")
        self._add_bot_message("Hello! I'm your financial advisor bot. I can help you understand your financial situation and provide personalized recommendations.")
        self._add_bot_message("What is your annual household income? (Please enter a number like 50000, 75000, 120000)")
    
    def run(self):
        """Run the chatbot interface with enhanced UI"""
        # Full-width header at top
        st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 3rem;">FinWise</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.5rem;">Your personal AI-powered financial assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tips section (only show before classification) - right after header
        if not st.session_state.classification_done:
            st.markdown("""
            <div class="help-text" style="margin: 5rem 0 1rem 0; padding: 0.5rem;">
                <strong>Tips:</strong><br>
                • Enter numbers without commas or dollar signs<br>
                • Be honest about your financial situation for better recommendations<br>
                • You can restart anytime by clicking the Restart button<br>
                • All your information is kept private and secure
            </div>
            """, unsafe_allow_html=True)
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            self._display_messages()
        
        # Input area with enhanced styling
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # User input
        if st.session_state.classification_done:
            with st.form(key="followup_form", clear_on_submit=True):
                user_input = st.text_input("Ask me anything about your finances:", key="followup_input", 
                                         placeholder="e.g., How much should I save for retirement?")
                submit_button = st.form_submit_button("Send", type="primary")
                if submit_button and user_input:
                    self._add_user_message(user_input)
                    self._handle_follow_up_questions(user_input)
                    st.rerun()
        else:
            with st.form(key="main_form", clear_on_submit=True):
                user_input = st.text_input("Your answer:", key="main_input",
                                         placeholder="Enter your response here...")
                col1, col2 = st.columns([1, 1])
                with col1:
                    submit_button = st.form_submit_button("Send", type="primary")
                with col2:
                    restart_button = st.form_submit_button("Restart")
                
                if submit_button and user_input:
                    self._add_user_message(user_input)
                    self._process_user_input(user_input)
                    st.rerun()
                elif restart_button:
                    self._restart_chat()
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer at bottom
        st.markdown("""
        <div class="footer">
            <p>© 2026 Financial Advisor Chatbot (Fin the Wise) - All Rights Reserved | Developed by Yogita and Suchita</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the chatbot"""
    chatbot = LiteFinancialChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()
