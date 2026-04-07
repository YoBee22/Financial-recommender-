# ─── SQLite fix for Streamlit Cloud (MUST be before any chromadb import) ───
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Local dev — system SQLite is fine

"""
streamlit_app.py — All-in-one Financial Advisor
Runs ML pipeline + rule engine + RAG chatbot in a single Streamlit app.

Local:  streamlit run streamlit_app.py
Deploy: Push to GitHub → Streamlit Community Cloud (free)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# ─────────────────────────────────────
# SETUP
# ─────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────
# PAGE CONFIG (must be the FIRST st. call)
# ─────────────────────────────────────

st.set_page_config(
    page_title="💰 Financial Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bridge Streamlit Cloud secrets → env vars
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

from app.ml_pipeline import (
    load_artifacts, predict_user_profile,
    match_funds, get_persona_descriptions,
)
from app.rule_engine import apply_rules


# ─────────────────────────────────────
# CSS
# ─────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { font-size: 1.05rem; color: #666; margin-bottom: 1.8rem; }
    .profile-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 14px; padding: 1.4rem 1.8rem; color: white;
        margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    .profile-card h3 { color: white; margin: 0 0 0.5rem 0; font-size: 1.25rem; }
    .profile-card p  { color: #e8e8ff; margin: 0.2rem 0; font-size: 0.95rem; }
    .profile-card strong { color: white; }
    .warning-box {
        background: #fff8e1; border-left: 4px solid #ff9800;
        padding: 0.7rem 1rem; border-radius: 0 8px 8px 0; margin: 0.35rem 0;
    }
    .rec-box {
        background: #e8f5e9; border-left: 4px solid #4caf50;
        padding: 0.7rem 1rem; border-radius: 0 8px 8px 0; margin: 0.35rem 0;
    }
    .chat-user {
        background: #e3f2fd; border-radius: 14px 14px 4px 14px;
        padding: 0.65rem 1rem; margin: 0.35rem 0;
    }
    .chat-bot {
        background: #f3e5f5; border-radius: 14px 14px 14px 4px;
        padding: 0.65rem 1rem; margin: 0.35rem 0;
    }
    .source-chip {
        display: inline-block; background: #e0e0e0; border-radius: 12px;
        padding: 0.2rem 0.6rem; margin: 0.15rem; font-size: 0.78rem; color: #555;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────
# LOAD ML MODELS (cached — runs once)
# ─────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML models …")
def init_models():
    return load_artifacts()

try:
    artifacts = init_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)


# ─────────────────────────────────────
# INIT RAG CHATBOT (cached — runs once)
# ─────────────────────────────────────

@st.cache_resource(show_spinner="Setting up AI chatbot …")
def init_chatbot():
    """Initialise RAG. Returns dict with qa_chain or None."""
    try:
        from app.rag_pipeline import build_documents, init_rag
        a = load_artifacts()
        docs = build_documents(a['fund_feat'], rec_map=a['rec_map'])
        rag = init_rag(docs)
        return rag
    except ImportError:
        return None
    except Exception as e:
        print(f"[RAG] Init error: {e}")
        return None


# ─────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────

if 'result' not in st.session_state:
    st.session_state.result = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# ─────────────────────────────────────
# SIDEBAR — USER INPUT FORM
# ─────────────────────────────────────

with st.sidebar:
    st.markdown("## 📝 Your Information")

    if not models_loaded:
        st.error(
            f"**Model files not found.**\n\n`{load_error}`\n\n"
            "Make sure your `data/processed/` folder contains all PKL and CSV "
            "files from Notebooks 1 & 2."
        )
        st.stop()

    # ── Demographics ──
    with st.expander("👤 Demographics", expanded=True):
        age = st.slider("Age", 18, 80, 30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        marital = st.selectbox("Marital Status", [
            "Never married", "Married-civilian spouse present",
            "Divorced", "Widowed", "Separated",
        ])
        education = st.slider("Education Level (0–50)", 0, 50, 40,
                               help="40 ≈ Bachelor's · 46 ≈ Master's")

    # ── Employment ──
    with st.expander("💼 Employment & Income", expanded=True):
        occupation = st.selectbox("Occupation", [
            "Professional specialty", "Executive admin and managerial",
            "Sales", "Adm support including clerical",
            "Precision production craft & repair",
            "Machine operators assmblrs & inspctrs",
            "Transportation and material moving",
            "Technicians and related support",
            "Farming forestry and fishing",
            "Protective services", "Other service", "Armed Forces",
        ])
        industry = st.selectbox("Industry", [
            "Finance insurance and real estate",
            "Manufacturing-durable goods", "Manufacturing-nondurable goods",
            "Retail trade", "Education", "Medical except hospital",
            "Hospital services", "Business and repair services",
            "Construction", "Transportation", "Communications",
            "Public administration", "Agriculture", "Other",
        ])
        worker_class = st.selectbox("Worker Class", [
            "Private", "Self-employed-not incorporated",
            "Self-employed-incorporated", "Local government",
            "State government", "Federal government",
        ])
        annual_income = st.number_input("Annual Income ($)", 0, 500_000, 55_000, step=5_000)
        hourly_wage = st.number_input("Hourly Wage ($)", 0.0, 200.0, 0.0, step=5.0)
        weeks_worked = st.slider("Weeks Worked / Year", 0, 52, 50)

    # ── Investments ──
    with st.expander("📊 Investments & Gains"):
        cap_gain = st.number_input("Capital Gains ($)", 0, 100_000, 0, step=1_000)
        cap_loss = st.number_input("Capital Losses ($)", 0, 100_000, 0, step=1_000)
        dividends = st.number_input("Dividends ($)", 0, 100_000, 0, step=500)

    # ── Spending ──
    with st.expander("💳 Spending Profile", expanded=True):
        savings_rate = st.slider("Savings Rate (%)", 0, 60, 15) / 100
        needs_ratio = st.slider("Needs Spending (%)", 20, 80, 50) / 100
        wants_ratio = st.slider("Wants Spending (%)", 5, 50, 30) / 100
        budget_health = st.slider("Budget Health Score", 0.0, 1.0, 0.50, step=0.05)
        income_pctile = st.slider("Income Percentile", 0.0, 1.0, 0.50, step=0.05)

    # ── Accounts ──
    with st.expander("🏦 Accounts & Savings"):
        filing_status = st.selectbox("Filing Status", [
            "single", "married_filing_jointly",
            "married_filing_separately", "head_of_household",
        ])
        has_401k = st.checkbox("Has 401(k)")
        has_match = st.checkbox("Employer Match Available") if has_401k else False
        has_roth = st.checkbox("Has Roth IRA")
        has_trad = st.checkbox("Has Traditional IRA")
        has_hdhp = st.checkbox("Has HDHP (High-Deductible Health Plan)")
        has_hsa = st.checkbox("Has HSA") if has_hdhp else False
        emergency_months = st.slider("Emergency Fund (months)", 0, 12, 2)
        total_debt = st.number_input("Total Debt ($)", 0, 500_000, 10_000, step=5_000)

    st.markdown("---")
    get_recs = st.button(
        "🚀 Get My Recommendations",
        use_container_width=True, type="primary",
    )


# ─────────────────────────────────────
# HELPER: build payload dict
# ─────────────────────────────────────

def get_user_input():
    return {
        "AAGE": age, "AHGA": education, "ACLSWKR": worker_class,
        "AMARITL": marital, "ASEX": sex, "AHRSPAY": hourly_wage,
        "WKSWORK": weeks_worked, "CAPGAIN": cap_gain,
        "GAPLOSS": cap_loss, "DIVVAL": dividends,
        "FILESTAT": filing_status.replace("_", " ").title(),
        "AMJOCC": occupation, "AMJIND": industry,
        "savings_rate": savings_rate, "needs_ratio": needs_ratio,
        "wants_ratio": wants_ratio, "budget_health_score": budget_health,
        "income_percentile": income_pctile,
    }


# ─────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────

st.markdown(
    '<div class="main-header">💰 Personalised Financial Advisor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">'
    'ML-powered fund recommendations · IRS-aware account guidance · AI chatbot'
    '</div>',
    unsafe_allow_html=True,
)

tab_recs, tab_chat, tab_about = st.tabs([
    "📊 Recommendations", "💬 AI Chatbot", "ℹ️ About"
])


# ═══════════════════════════════════════
# TAB 1: RECOMMENDATIONS
# ═══════════════════════════════════════

with tab_recs:

    if get_recs:
        user_input = get_user_input()

        with st.spinner("Analysing your profile and matching funds …"):
            # 1. Predict
            profile = predict_user_profile(user_input)

            # 2. Match funds
            funds_df = match_funds(
                profile['risk_tolerance'],
                profile['spending_behavior'],
                top_n=10,
            )

            # 3. Apply rules
            rule_result = apply_rules(
                user_info={
                    'age': age, 'annual_income': annual_income,
                    'filing_status': filing_status,
                    'has_401k': has_401k, 'has_employer_match': has_match,
                    'has_roth_ira': has_roth, 'has_traditional_ira': has_trad,
                    'has_hsa': has_hsa, 'has_hdhp': has_hdhp,
                    'emergency_fund_months': emergency_months,
                    'total_debt': total_debt,
                },
                top_funds=funds_df,
            )

            # 4. Persona
            personas = get_persona_descriptions()
            persona_desc = personas.get(
                (profile['risk_tolerance'], profile['spending_behavior']),
                "Custom profile",
            )

            st.session_state.result = {
                'profile': profile,
                'persona': persona_desc,
                'funds': funds_df,
                'account_recs': rule_result['account_recommendations'],
                'warnings': rule_result['warnings'],
            }

    # ── Display results ──
    result = st.session_state.result

    if result is None:
        st.info("👈 Fill in the sidebar and click **Get My Recommendations** to start.")
    else:
        p = result['profile']

        # Profile card
        st.markdown(f"""
        <div class="profile-card">
            <h3>🎯 Your Investor Profile</h3>
            <p><strong>Risk Tolerance:</strong> {p['risk_tolerance'].title()}</p>
            <p><strong>Spending Behaviour:</strong> {p['spending_behavior'].replace('_',' ').title()}</p>
            <p><strong>Persona:</strong> {result['persona']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Profile", p['risk_tolerance'].title())
        c2.metric("Spending Type", p['spending_behavior'].replace('_', ' ').title())
        c3.metric("Funds Matched", len(result['funds']))

        # Warnings
        if result['warnings']:
            st.markdown("### ⚠️ Important Warnings")
            for w in result['warnings']:
                st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)

        # Account recommendations
        if result['account_recs']:
            st.markdown("### 🏦 Account & Savings Recommendations")
            for r in result['account_recs']:
                st.markdown(f'<div class="rec-box">{r}</div>', unsafe_allow_html=True)

        # Fund table
        st.markdown("### 📈 Top Fund Recommendations")
        funds_df = result['funds']

        if not funds_df.empty:
            display = funds_df.copy()

            # Convert similarity to Match %
            if 'similarity_score' in display.columns:
                display.insert(0, 'Match %', (display['similarity_score'] * 100).round(1))
                display = display.drop(columns=['similarity_score'])

            # Friendly column names
            rename = {
                'fund_name': 'Fund Name', 'fund_symbol': 'Symbol',
                'fund_long_name': 'Fund Name', 'symbol': 'Symbol',
                'name': 'Fund Name', 'investment_type': 'Type',
                'fund_risk_tier': 'Risk Tier', 'expense_ratio': 'Expense Ratio',
                'avg_return': 'Avg Return %', 'composite_score': 'Quality Score',
                'alloc_stocks': 'Stocks %', 'alloc_bonds': 'Bonds %',
            }
            display = display.rename(columns={
                k: v for k, v in rename.items() if k in display.columns
            })

            # Preferred column order
            priority = [
                'Match %', 'Fund Name', 'Symbol', 'Type', 'Risk Tier',
                'Expense Ratio', 'Avg Return %', 'Quality Score',
                'Stocks %', 'Bonds %',
            ]
            cols = list(display.columns)
            ordered = [c for c in priority if c in cols]
            ordered += [c for c in cols if c not in ordered]
            display = display[ordered]

            st.dataframe(display, use_container_width=True, hide_index=True, height=420)

            st.download_button(
                "📥 Download as CSV",
                display.to_csv(index=False),
                "my_fund_recommendations.csv",
                "text/csv",
            )
        else:
            st.warning("No funds matched. Try adjusting your profile.")


# ═══════════════════════════════════════
# TAB 2: AI CHATBOT
# ═══════════════════════════════════════

with tab_chat:
    st.markdown("### 💬 Ask Your Financial Advisor")

    # Check RAG availability
    rag_available = False
    if models_loaded:
        chatbot = init_chatbot()
        rag_available = chatbot is not None and chatbot.get('qa_chain') is not None

    if not rag_available:
        st.info(
            "**To enable the AI chatbot:**\n\n"
            "**1.** Install dependencies:\n"
            "```\n"
            "pip install langchain langchain-google-genai langchain-community "
            "chromadb sentence-transformers\n"
            "```\n\n"
            "**2.** Get a free Google Gemini API key at "
            "[ai.google.dev](https://ai.google.dev)\n\n"
            "**3.** Set it:\n"
            "```\n"
            "export GOOGLE_API_KEY='AIza-your-key'\n"
            "```\n"
            "Or on Streamlit Cloud: **Settings → Secrets** → add "
            "`GOOGLE_API_KEY = \"AIza...\"`\n\n"
            "**4.** Restart the app.\n\n"
            "---\n"
            "*Fund recommendations and account guidance work without the chatbot.*"
        )
    else:
        st.markdown(
            "Ask about your recommendations, retirement accounts, "
            "investment strategies, or personal finance."
        )

        # ── Chat history ──
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(
                    f'<div class="chat-user">🧑 <strong>You:</strong> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bot">🤖 <strong>Advisor:</strong> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                # Show source chips
                if msg.get('sources'):
                    chips = ''.join(
                        f'<span class="source-chip">{s["type"]}</span>'
                        for s in msg['sources']
                    )
                    st.markdown(f"Sources: {chips}", unsafe_allow_html=True)

        # ── Quick question buttons ──
        st.markdown("**Quick questions:**")
        q1, q2, q3 = st.columns(3)
        quick_q = None
        with q1:
            if st.button("Why these funds?", use_container_width=True):
                quick_q = "Why were these specific funds recommended for my profile?"
        with q2:
            if st.button("Roth vs Traditional?", use_container_width=True):
                quick_q = "Should I use a Roth IRA or Traditional IRA given my situation?"
        with q3:
            if st.button("How to start investing?", use_container_width=True):
                quick_q = "I'm new to investing. What should I do first?"

        # More quick questions
        q4, q5, q6 = st.columns(3)
        with q4:
            if st.button("What is an expense ratio?", use_container_width=True):
                quick_q = "What is an expense ratio and why does it matter?"
        with q5:
            if st.button("Emergency fund advice", use_container_width=True):
                quick_q = "How much should I save in an emergency fund before investing?"
        with q6:
            if st.button("401k vs Roth IRA?", use_container_width=True):
                quick_q = "Should I prioritize my 401k or Roth IRA contributions?"

        # ── Text input ──
        user_question = st.chat_input("Type your question …")
        question = quick_q or user_question

        if question:
            # Build user context from current recommendations
            user_context = ""
            if st.session_state.result:
                r = st.session_state.result
                user_context = (
                    f"Profile: {r['profile']['risk_tolerance']} risk tolerance, "
                    f"{r['profile']['spending_behavior']} spending behaviour. "
                    f"Persona: {r['persona']}. "
                    f"Age: {age}, Annual income: ${annual_income:,}, "
                    f"Filing status: {filing_status}."
                )

            st.session_state.chat_history.append({
                'role': 'user', 'content': question
            })

            with st.spinner("Thinking …"):
                from app.rag_pipeline import ask
                resp = ask(question, user_context)

            st.session_state.chat_history.append({
                'role': 'bot',
                'content': resp.get('answer', 'Sorry, I could not generate an answer.'),
                'sources': resp.get('sources', []),
            })
            st.rerun()

        # ── Clear chat ──
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()


# ═══════════════════════════════════════
# TAB 3: ABOUT
# ═══════════════════════════════════════

with tab_about:
    st.markdown("""
    ### How It Works

    This system uses a **hybrid approach** combining machine learning,
    rule-based logic, and retrieval-augmented generation:

    ---

    #### 1. 🤖 ML Segmentation
    - **UCI Census-Income** clustered with K-Means (**K=10**) →
      captures *who you are* (income, age, education, employment)
    - **BLS Consumer Expenditure Survey** clustered with K-Means (**K=2**) →
      captures *how you spend* (savings rate, needs vs wants)
    - Supervised classifiers (**XGBoost, Random Forest, Logistic Regression**)
      predict your profile from raw inputs
    - Composite profile = (risk_tolerance × spending_behavior) = **6 personas**

    #### 2. 📊 Fund Matching
    - **Content-based filtering** using **cosine similarity** between your
      profile's preference vector and fund feature vectors
    - Fund features: risk score, expense ratio, average return, return
      consistency, sector diversification, asset allocation
    - Data source: Kaggle US ETF & Mutual Fund dataset
    - Returns a ranked list of top 10 best-matching funds

    #### 3. 📋 Rule-Based Engine
    - **Roth IRA** income limits and contribution caps
    - **401(k)** employer match prioritisation
    - **HSA** eligibility (requires HDHP)
    - **Traditional IRA** deduction limits
    - Emergency fund and debt-to-income safety checks
    - Tax-efficient fund placement advice

    #### 4. 💬 RAG Chatbot
    - **ChromaDB** vector store with embedded fund data + financial education
    - **sentence-transformers** (all-MiniLM-L6-v2) for local embeddings
    - **Google Gemini 2.5 Flash** (free) for natural language generation
    - **LangChain** RetrievalQA chain for orchestration
    - Answers grounded in your profile and retrieved context

    ---

    ### 6 Investor Personas

    | Risk | Spending | Persona |
    |------|----------|---------|
    | Aggressive | High Saver | 🚀 Max Growth Investor |
    | Aggressive | Low Saver | 📈 Growth Seeker |
    | Moderate | High Saver | ⚖️ Balanced Builder |
    | Moderate | Low Saver | 🛡️ Cautious Grower |
    | Conservative | High Saver | 🏦 Steady Preserver |
    | Conservative | Low Saver | 🔒 Safety First |

    ---

    ### Tech Stack

    | Component | Technology |
    |-----------|-----------|
    | Frontend | Streamlit |
    | ML Models | scikit-learn, XGBoost |
    | Clustering | K-Means (UCI K=10, CE K=2) |
    | Fund Matching | Cosine Similarity |
    | Rule Engine | Python (IRS rules) |
    | Embeddings | sentence-transformers |
    | Vector Store | ChromaDB |
    | LLM | Google Gemini 2.5 Flash (free) |
    | Orchestration | LangChain |
    | Deployment | Streamlit Community Cloud (free) |

    ---

    **Capstone Project — Northeastern University**
    """)
