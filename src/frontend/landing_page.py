import streamlit as st

def main():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }

        .stApp, [data-testid="stAppViewContainer"] {
            background: #0a0f0a !important;
        }

        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }

        /* ── HERO ── */
        .hero {
            min-height: 100vh;
            background: #0a0f0a;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            padding: 4rem 2rem;
        }

        /* Radial glow */
        .hero::before {
            content: '';
            position: absolute;
            top: -20%;
            left: 50%;
            transform: translateX(-50%);
            width: 900px;
            height: 700px;
            background: radial-gradient(ellipse, rgba(138,195,90,0.12) 0%, transparent 70%);
            pointer-events: none;
        }

        /* Grid texture */
        .hero::after {
            content: '';
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(138,195,90,0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(138,195,90,0.04) 1px, transparent 1px);
            background-size: 60px 60px;
            pointer-events: none;
        }

        .hero-inner {
            position: relative;
            z-index: 2;
            text-align: center;
            max-width: 820px;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: rgba(138,195,90,0.1);
            border: 1px solid rgba(138,195,90,0.25);
            border-radius: 100px;
            padding: 6px 18px;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.78rem;
            font-weight: 500;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8ac35a;
            margin-bottom: 2.4rem;
            animation: fadeDown 0.7s ease both;
        }

        .eyebrow-dot {
            width: 6px; height: 6px;
            background: #8ac35a;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.7); }
        }

        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: clamp(3rem, 7vw, 5.5rem);
            font-weight: 700;
            line-height: 1.08;
            color: #f0ede6;
            letter-spacing: -0.02em;
            margin-bottom: 1.6rem;
            animation: fadeUp 0.7s 0.1s ease both;
        }

        .hero-title em {
            font-style: italic;
            color: #8ac35a;
        }

        .hero-sub {
            font-family: 'DM Sans', sans-serif;
            font-size: 1.15rem;
            font-weight: 300;
            color: rgba(240,237,230,0.55);
            line-height: 1.7;
            max-width: 560px;
            margin: 0 auto 3rem;
            animation: fadeUp 0.7s 0.2s ease both;
        }

        .cta-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1.2rem;
            flex-wrap: wrap;
            animation: fadeUp 0.7s 0.3s ease both;
        }

        /* ── STATS STRIP ── */
        .stats-strip {
            background: rgba(138,195,90,0.06);
            border-top: 1px solid rgba(138,195,90,0.12);
            border-bottom: 1px solid rgba(138,195,90,0.12);
            padding: 2.5rem 4rem;
            display: flex;
            justify-content: center;
            gap: 5rem;
            flex-wrap: wrap;
            position: relative;
            z-index: 2;
            animation: fadeUp 0.7s 0.4s ease both;
        }

        .stat-item {
            text-align: center;
        }

        .stat-num {
            font-family: 'Playfair Display', serif;
            font-size: 2.4rem;
            font-weight: 700;
            color: #8ac35a;
            display: block;
            line-height: 1;
        }

        .stat-label {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.82rem;
            color: rgba(240,237,230,0.4);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-top: 0.4rem;
        }

        /* ── FEATURE CARDS ── */
        .features-section {
            background: #0a0f0a;
            padding: 6rem 3rem;
            position: relative;
        }

        .section-label {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.75rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: #8ac35a;
            text-align: center;
            margin-bottom: 1rem;
        }

        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 700;
            color: #f0ede6;
            text-align: center;
            margin-bottom: 4rem;
        }

        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.5rem;
            max-width: 1100px;
            margin: 0 auto;
        }

        .feat-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(138,195,90,0.12);
            border-radius: 20px;
            padding: 2.2rem 2rem;
            transition: transform 0.3s, border-color 0.3s, box-shadow 0.3s;
            position: relative;
            overflow: hidden;
        }

        .feat-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #8ac35a, transparent);
            opacity: 0;
            transition: opacity 0.3s;
        }

        .feat-card:hover {
            transform: translateY(-6px);
            border-color: rgba(138,195,90,0.3);
            box-shadow: 0 24px 48px rgba(0,0,0,0.4);
        }

        .feat-card:hover::before { opacity: 1; }

        .feat-icon {
            font-size: 2rem;
            margin-bottom: 1.2rem;
        }

        .feat-name {
            font-family: 'Playfair Display', serif;
            font-size: 1.2rem;
            font-weight: 600;
            color: #f0ede6;
            margin-bottom: 0.7rem;
        }

        .feat-desc {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.9rem;
            color: rgba(240,237,230,0.45);
            line-height: 1.65;
        }

        /* ── QUOTE SECTION ── */
        .quote-section {
            background: linear-gradient(135deg, rgba(138,195,90,0.08) 0%, rgba(138,195,90,0.02) 100%);
            border-top: 1px solid rgba(138,195,90,0.12);
            border-bottom: 1px solid rgba(138,195,90,0.12);
            padding: 5rem 3rem;
            text-align: center;
        }

        .quote-mark {
            font-family: 'Playfair Display', serif;
            font-size: 5rem;
            line-height: 0.5;
            color: rgba(138,195,90,0.3);
            margin-bottom: 1rem;
        }

        .quote-text {
            font-family: 'Playfair Display', serif;
            font-size: clamp(1.4rem, 3vw, 2rem);
            font-style: italic;
            color: #f0ede6;
            max-width: 720px;
            margin: 0 auto 1.5rem;
            line-height: 1.5;
        }

        .quote-attr {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.85rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8ac35a;
        }

        /* ── CTA BOTTOM ── */
        .cta-bottom {
            background: #0a0f0a;
            padding: 6rem 3rem;
            text-align: center;
        }

        .cta-bottom-title {
            font-family: 'Playfair Display', serif;
            font-size: clamp(2rem, 4vw, 3rem);
            color: #f0ede6;
            margin-bottom: 1rem;
        }

        .cta-bottom-sub {
            font-family: 'DM Sans', sans-serif;
            color: rgba(240,237,230,0.45);
            font-size: 1rem;
            margin-bottom: 2.5rem;
        }

        /* ── FOOTER ── */
        .site-footer {
            background: rgba(138,195,90,0.04);
            border-top: 1px solid rgba(138,195,90,0.1);
            padding: 1.8rem 3rem;
            text-align: center;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.82rem;
            color: rgba(240,237,230,0.3);
        }

        /* ── ANIMATIONS ── */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(24px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeDown {
            from { opacity: 0; transform: translateY(-16px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ── BUTTON overrides ── */
        .stButton > button {
            background: #8ac35a !important;
            color: #0a0f0a !important;
            border: none !important;
            padding: 0.9rem 2.4rem !important;
            border-radius: 100px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.04em !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 20px rgba(138,195,90,0.25) !important;
        }

        .stButton > button:hover {
            background: #a0d870 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 30px rgba(138,195,90,0.35) !important;
        }

        div[data-testid="stButton"] {
            display: flex;
            justify-content: center;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── HERO ──
    st.markdown("""
    <div class="hero">
        <div class="hero-inner">
            <div class="eyebrow">
                <span class="eyebrow-dot"></span>
                AI-Powered Financial Intelligence
            </div>
            <h1 class="hero-title">
                Your money,<br><em>finally working</em><br>for you.
            </h1>
            <p class="hero-sub">
                FinWise analyzes your income, spending patterns, and family profile
                to deliver personalized ETF &amp; mutual fund recommendations — in minutes.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA Button (Streamlit native, centered)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Begin Your Journey →", key="hero_cta"):
            st.session_state.force_page = "chatbot"
            st.query_params.page = "chatbot"
            st.rerun()

    # ── FEATURES ──
    st.markdown("""
    <div class="features-section">
        <p class="section-label">What We Offer</p>
        <h2 class="section-title">Built on real data. Designed for real life.</h2>
        <div class="cards-grid">
            <div class="feat-card">
                <div class="feat-icon"></div>
                <div class="feat-name">Smart Profiling</div>
                <div class="feat-desc">K-Means, GMM, HDBSCAN, and hierarchical clustering map your financial DNA into a precise peer group.</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"></div>
                <div class="feat-name">ETF &amp; Fund Matching</div>
                <div class="feat-desc">Cosine similarity and XGBoost rank funds from real market data against your unique spending and savings profile.</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"></div>
                <div class="feat-name">IRS-Grounded Rules</div>
                <div class="feat-desc">Roth &amp; Traditional IRA eligibility checks built on current IRS thresholds not guesswork.</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"></div>
                <div class="feat-name">Conversational Advisor</div>
                <div class="feat-desc">FinWise walks you through income, expenses, and family size in a natural chat no forms, no jargon.</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"></div>
                <div class="feat-name">Private by Design</div>
                <div class="feat-desc">Your data lives only in your session. Nothing is stored or shared beyond the conversation.</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"></div>
                <div class="feat-name">Dashboard for Tracking</div>
                <div class="feat-desc">Visualize your wealth trajectory, track savings goals, and monitor financial progress with interactive dashboards.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── QUOTE ──
    st.markdown("""
    <div class="quote-section">
        <div class="quote-mark">"</div>
        <p class="quote-text">Do not save what is left after spending, but spend what is left after saving.</p>
        <p class="quote-attr">— Warren Buffett</p>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA BOTTOM ──
    st.markdown("""
    <div class="cta-bottom">
        <h2 class="cta-bottom-title">Ready to take control?</h2>
        <p class="cta-bottom-sub">A 2-minute conversation with FinWise is all it takes.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Start with FinWise →", key="bottom_cta"):
            st.session_state.force_page = "chatbot"
            st.query_params.page = "chatbot"
            st.rerun()

    # ── FOOTER ──
    st.markdown("""
    <div class="site-footer">
        © 2026 FinWise — Financial Recommendation System &nbsp;·&nbsp; Developed by Yogita &amp; Suchita
    </div>
    """, unsafe_allow_html=True)