"""
FinWise Dashboard — Projected savings & wealth visualizations
Reads from st.session_state populated by the chatbot flow.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime


# ── helpers ───────────────────────────────────────────────────────────────────

def _compound(principal, monthly_contrib, annual_rate, years):
    """Return year-by-year balance list."""
    balance = principal
    r = annual_rate / 12
    balances = []
    for y in range(1, years + 1):
        for _ in range(12):
            balance = balance * (1 + r) + monthly_contrib
        balances.append(balance)
    return balances


def _get_user_data():
    """Pull data from session_state or return safe defaults."""
    ud = st.session_state.get('user_data', {})
    income   = ud.get('income', 0)
    expenses = ud.get('expenses', 0)
    family   = ud.get('family_size', 1)
    bracket  = ud.get('income_bracket', 'Middle Income Families')
    sr       = ud.get('savings_rate', 0)
    done     = ud.get('classification_done', False)
    return income, expenses, family, bracket, sr, done


def _plotly_theme():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='rgba(240,237,230,0.7)', size=12),
        xaxis=dict(
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)'),
        ),
        yaxis=dict(
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)'),
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(240,237,230,0.6)'),
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )


GREEN   = '#8ac35a'
GREEN2  = '#5a9c8a'
YELLOW  = '#d4b84a'
RED     = '#c05a5a'
MUTED   = 'rgba(138,195,90,0.18)'


# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_dash_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

        * { box-sizing: border-box; }

        html, body,
        [data-testid="stAppViewContainer"],
        .stApp { background: #0a0f0a !important; font-family: 'DM Sans', sans-serif; }

        .block-container { padding: 0 !important; max-width: 100% !important; }

        /* topbar */
        .dash-topbar {
            position: fixed; top: 0; left: 0; right: 0; height: 60px;
            background: rgba(10,15,10,0.96); backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(138,195,90,0.15);
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 2rem; z-index: 1000;
        }
        .dash-brand {
            font-family: 'Playfair Display', serif; font-size: 1.2rem;
            font-weight: 600; color: #f0ede6;
            display: flex; align-items: center; gap: 10px;
        }
        .live-dot {
            width: 8px; height: 8px; background: #8ac35a;
            border-radius: 50%; animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)}
        }
        .nav-pills {
            display: flex; gap: 0.5rem;
        }
        .nav-pill-link {
            padding: 0.3rem 1rem; border-radius: 100px; font-size: 0.82rem;
            cursor: pointer; transition: all .2s; text-decoration: none;
            font-family: 'DM Sans', sans-serif; font-weight: 500;
        }
        .nav-pill-link.active {
            background: rgba(138,195,90,0.18); color: #8ac35a;
            border: 1px solid rgba(138,195,90,0.3);
        }
        .nav-pill-link:not(.active) {
            color: rgba(240,237,230,0.35); border: 1px solid transparent;
        }
        .nav-pill-link:not(.active):hover {
            color: rgba(240,237,230,0.6);
            background: rgba(138,195,90,0.08);
        }

        /* page wrapper */
        .dash-page {
            padding: 80px 2.5rem 3rem;
            max-width: 1280px; margin: 0 auto;
        }

        /* hero row */
        .dash-hero {
            margin-bottom: 2rem;
            animation: fadeUp .5s ease both;
        }
        .dash-eyebrow {
            font-size: 0.78rem; letter-spacing: .14em; text-transform: uppercase;
            color: rgba(138,195,90,.6); margin-bottom: .5rem;
        }

        .dash-title {
            font-family: 'Playfair Display', serif;
            font-size: clamp(2rem, 4.5vw, 3.5rem);
            font-weight: 700; color: #f0ede6; line-height: 1.1;
            margin-bottom: .5rem;
        }
        .dash-title em { font-style: italic; color: #8ac35a; }
        .dash-sub {
            font-size: 0.92rem; color: rgba(240,237,230,.4); max-width: 520px;
        }

        /* KPI row */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem; margin-bottom: 2rem;
        }
        .kpi-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(138,195,90,0.12);
            border-radius: 16px; padding: 1.4rem 1.2rem;
            position: relative; overflow: hidden;
            animation: fadeUp .5s ease both;
            transition: transform .2s, border-color .2s;
        }
        .kpi-card:hover {
            transform: translateY(-3px);
            border-color: rgba(138,195,90,.28);
        }
        .kpi-card::after {
            content: '';
            position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, var(--kpi-color, #8ac35a), transparent);
        }
        .kpi-label {
            font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
            color: rgba(240,237,230,.35); margin-bottom: .6rem;
        }
        .kpi-value {
            font-family: 'Playfair Display', serif;
            font-size: 1.9rem; font-weight: 700;
            color: var(--kpi-color, #8ac35a); line-height: 1;
        }
        .kpi-delta {
            font-size: .78rem; color: rgba(240,237,230,.35);
            margin-top: .4rem;
        }

        /* section headings */
        .section-head {
            display: flex; align-items: baseline; gap: 1rem;
            margin: 2.2rem 0 1rem;
        }
        .section-head h3 {
            font-family: 'Playfair Display', serif;
            font-size: 1.5rem; color: #f0ede6; font-weight: 600;
        }
        .section-badge {
            font-size: .7rem; letter-spacing: .1em; text-transform: uppercase;
            padding: 2px 10px; border-radius: 100px;
            background: rgba(138,195,90,.1); color: #8ac35a;
            border: 1px solid rgba(138,195,90,.2);
        }

        /* chart cards */
        .chart-card {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(138,195,90,0.1);
            border-radius: 20px; padding: 1.5rem;
            margin-bottom: 1.5rem;
            animation: fadeUp .5s ease both;
        }
        .chart-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.2rem; color: rgba(240,237,230,.8);
            margin-bottom: .3rem;
        }
        .chart-desc {
            font-size: .8rem; color: rgba(240,237,230,.3);
            margin-bottom: 1rem;
        }

        /* milestone table */
        .milestone-row {
            display: flex; align-items: center; gap: 1rem;
            padding: .75rem 0;
            border-bottom: 1px solid rgba(138,195,90,.07);
        }
        .milestone-row:last-child { border-bottom: none; }
        .ms-year {
            font-family: 'Playfair Display', serif;
            font-size: 1.1rem; color: #8ac35a; font-weight: 700;
            width: 50px; flex-shrink: 0;
        }
        .ms-bar-wrap {
            flex: 1; height: 6px;
            background: rgba(138,195,90,0.1); border-radius: 10px; overflow: hidden;
        }
        .ms-bar { height: 100%; border-radius: 10px;
            background: linear-gradient(90deg, #8ac35a, #5a9c8a); }
        .ms-amount {
            font-size: .88rem; color: rgba(240,237,230,.6);
            width: 100px; text-align: right; flex-shrink: 0;
        }

        /* scenario toggle */
        .scenario-label {
            font-size: .78rem; color: rgba(240,237,230,.4);
            letter-spacing: .06em;
        }

        /* no-data banner */
        .no-data-banner {
            text-align: center;
            padding: 4rem 2rem;
            background: rgba(138,195,90,0.04);
            border: 1px dashed rgba(138,195,90,0.2);
            border-radius: 20px; margin: 3rem 0;
        }
        .no-data-icon { font-size: 3rem; margin-bottom: 1rem; }
        .no-data-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.5rem; color: #f0ede6; margin-bottom: .6rem;
        }
        .no-data-sub { font-size: .9rem; color: rgba(240,237,230,.4); }

        @keyframes fadeUp {
            from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)}
        }

        /* Streamlit buttons */
        .stButton > button {
            background: transparent !important;
            color: rgba(240,237,230,.55) !important;
            border: 1px solid rgba(138,195,90,.2) !important;
            padding: .5rem 1.4rem !important;
            border-radius: 100px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: .82rem !important; transition: all .2s !important;
        }
        .stButton > button:hover {
            border-color: rgba(138,195,90,.5) !important;
            color: #f0ede6 !important;
            background: rgba(138,195,90,.06) !important;
        }
        .stButton.primary-btn > button {
            background: #8ac35a !important;
            color: #0a0f0a !important;
            border-color: #8ac35a !important;
        }
        .stButton.primary-btn > button:hover {
            background: #a0d870 !important;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label {
            color: rgba(240,237,230,.45) !important;
            font-size: .82rem !important;
        }

        /* selectbox */
        div[data-testid="stSelectbox"] > div > div {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(138,195,90,.2) !important;
            border-radius: 10px !important; color: #f0ede6 !important;
        }

        /* slider */
        div[data-testid="stSlider"] .stSlider > div {
            color: #8ac35a !important;
        }

        /* metric overrides */
        [data-testid="stMetric"] { display: none !important; }

        /* info box */
        .info-box {
            background: rgba(138,195,90,0.06);
            border-left: 3px solid #8ac35a;
            border-radius: 0 10px 10px 0;
            padding: .8rem 1.2rem;
            font-size: .85rem; color: rgba(240,237,230,.55);
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)


# ── charts ────────────────────────────────────────────────────────────────────

def chart_projected_savings(income, expenses, years=30):
    """Three scenario compound growth chart."""
    current_savings = max(income - expenses, 0)
    monthly_current = current_savings / 12

    # Scenarios: current, recommended (10%), aggressive (20%)
    monthly_rec  = income * 0.10 / 12
    monthly_aggr = income * 0.20 / 12

    yrs = list(range(1, years + 1))
    bal_current = _compound(0, monthly_current, 0.06, years)
    bal_rec     = _compound(0, monthly_rec,     0.07, years)
    bal_aggr    = _compound(0, monthly_aggr,    0.08, years)

    fig = go.Figure()

    # Fill between current and aggressive
    fig.add_trace(go.Scatter(
        x=yrs + yrs[::-1],
        y=bal_aggr + bal_current[::-1],
        fill='toself',
        fillcolor='rgba(138,195,90,0.06)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False, hoverinfo='skip',
    ))

    fig.add_trace(go.Scatter(
        x=yrs, y=bal_current,
        name='Current pace',
        line=dict(color=RED, width=2, dash='dot'),
        mode='lines',
    ))
    fig.add_trace(go.Scatter(
        x=yrs, y=bal_rec,
        name='Recommended (10%)',
        line=dict(color=GREEN, width=2.5),
        mode='lines',
    ))
    fig.add_trace(go.Scatter(
        x=yrs, y=bal_aggr,
        name='Aggressive (20%)',
        line=dict(color=GREEN2, width=2, dash='dash'),
        mode='lines',
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='rgba(240,237,230,0.7)', size=12),
        height=360,
        yaxis=dict(
            tickprefix='$', 
            tickformat=',.0f',
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        xaxis=dict(
            title='Years', 
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(240,237,230,0.6)'),
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='left', 
            x=0
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def chart_savings_breakdown(income, expenses):
    """Donut — where the money goes."""
    savings = max(income - expenses, 0)  # Ensure never negative
    deficit = max(expenses - income, 0)

    if income <= 0:
        return None

    labels = ['Expenses', 'Current Savings']
    values = [min(expenses, income), savings]
    colors = [RED, GREEN]

    if deficit > 0:
        labels = ['Expenses (Deficit)', f'Income Shortfall']
        values = [income, deficit]
        colors = [RED, YELLOW]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color='#0a0f0a', width=3)),
        textfont=dict(color='rgba(240,237,230,0.7)', size=12),
        hovertemplate='%{label}: $%{value:,.0f}<extra></extra>',
    ))
    fig.add_annotation(
        text=f"${income:,.0f}<br><span style='font-size:10px'>Annual</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color='#f0ede6', family='Playfair Display, serif'),
        align='center',
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='rgba(240,237,230,0.7)', size=12),
        height=300,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(240,237,230,0.6)'),
            orientation='h', 
            yanchor='bottom', 
            y=-0.15
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def chart_monthly_budget(income, expenses):
    """50/30/20 vs actual horizontal bar."""
    monthly = income / 12
    act_exp = min(expenses / 12, monthly)
    act_sav = max((income - expenses) / 12, 0)  # Ensure never negative

    cats = ['Needs', 'Wants', 'Savings']
    recommended = [monthly * 0.50, monthly * 0.30, monthly * 0.20]
    # Ensure actual values are never negative
    actual = [min(act_exp * 0.65, monthly), min(act_exp * 0.35, monthly), act_sav]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Recommended (50/30/20)',
        x=recommended, y=cats,
        orientation='h',
        marker_color=[GREEN, GREEN2, YELLOW],
        marker_line_color='rgba(0,0,0,0)',
        opacity=0.5,
    ))
    fig.add_trace(go.Bar(
        name='Your Actual',
        x=actual, y=cats,
        orientation='h',
        marker_color=[RED, YELLOW, GREEN],
        marker_line_color='rgba(0,0,0,0)',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='rgba(240,237,230,0.7)', size=12),
        height=260,
        barmode='group',
        xaxis=dict(
            tickprefix='$', 
            tickformat=',.0f',
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        yaxis=dict(
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(240,237,230,0.6)'),
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='left', 
            x=0
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def chart_retirement_runway(income, savings_rate, years=35):
    """Stacked bar — contributions vs growth each 5-year block."""
    monthly = income * savings_rate / 12
    monthly_rec = income * 0.10 / 12

    periods = list(range(5, years+1, 5))
    contrib_actual, growth_actual = [], []
    contrib_rec, growth_rec = [], []

    years_list = periods
    for y in years_list:
        bal_a = _compound(0, monthly, 0.07, y)[-1]
        total_contrib_a = monthly * 12 * y
        contrib_actual.append(min(total_contrib_a, bal_a))
        growth_actual.append(max(bal_a - total_contrib_a, 0))

        bal_r = _compound(0, monthly_rec, 0.07, y)[-1]
        total_contrib_r = monthly_rec * 12 * y
        contrib_rec.append(min(total_contrib_r, bal_r))
        growth_rec.append(max(bal_r - total_contrib_r, 0))

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Your Contributions', x=[f'Yr {p}' for p in periods],
                         y=contrib_actual, marker_color=GREEN2))
    fig.add_trace(go.Bar(name='Investment Growth', x=[f'Yr {p}' for p in periods],
                         y=growth_actual, marker_color=GREEN))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='rgba(240,237,230,0.7)', size=12),
        height=300, 
        barmode='stack',
        yaxis=dict(
            tickprefix='$', 
            tickformat=',.0f',
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        xaxis=dict(
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(240,237,230,0.6)'),
            orientation='h', 
            yanchor='bottom', 
            y=1.02
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def chart_emergency_fund(expenses, current_savings_monthly, years=5):
    """Line chart — how fast you reach 3-month & 6-month targets."""
    target_3 = expenses * 0.25
    target_6 = expenses * 0.50

    months = list(range(0, years*12+1))
    # Ensure monthly savings is never negative and calculate realistic monthly contribution
    monthly_contribution = max(current_savings_monthly * 1.5, expenses * 0.01, 50)  # Minimum $50 or 1% of expenses
    balance = [min(monthly_contribution * m, target_6 * 1.05) for m in months]

    fig = go.Figure()
    fig.add_hline(y=target_3, line=dict(color=YELLOW, dash='dot', width=1.5),
                  annotation_text='3-month target', annotation_font_color=YELLOW,
                  annotation_position='top right')
    fig.add_hline(y=target_6, line=dict(color=GREEN, dash='dot', width=1.5),
                  annotation_text='6-month target', annotation_font_color=GREEN,
                  annotation_position='top right')
    fig.add_trace(go.Scatter(
        x=[m/12 for m in months], y=balance,
        fill='tozeroy', fillcolor='rgba(138,195,90,0.08)',
        line=dict(color=GREEN, width=2),
        name='Projected fund', mode='lines',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='rgba(240,237,230,0.7)', size=12),
        height=280,
        xaxis=dict(
            title='Years', 
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        yaxis=dict(
            tickprefix='$', 
            tickformat=',.0f',
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def chart_savings_comparison(income, expenses, years=30):
    """Compare current savings vs recommended savings over time."""
    current_monthly = max(income - expenses, 0) / 12
    recommended_monthly = income * 0.10 / 12
    current_rate = max(income - expenses, 0) / income if income > 0 else 0
    
    # If user is already saving more than recommended, use their current rate as the "recommended"
    if current_rate >= 0.10:
        recommended_monthly = current_monthly
        recommended_rate = current_rate
    else:
        recommended_rate = 0.10
    
    yrs = list(range(1, years + 1))
    current_balances = _compound(0, current_monthly, 0.06, years)
    recommended_balances = _compound(0, recommended_monthly, 0.07, years)
    
    # Calculate additional savings from following recommendations
    additional_savings = [max(rec - cur, 0) for rec, cur in zip(recommended_balances, current_balances)]
    
    fig = go.Figure()
    
    # Add area chart showing the gap
    fig.add_trace(go.Scatter(
        x=yrs + yrs[::-1],
        y=recommended_balances + current_balances[::-1],
        fill='toself',
        fillcolor='rgba(138,195,90,0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Additional Savings',
        hoverinfo='skip',
    ))
    
    fig.add_trace(go.Scatter(
        x=yrs, y=current_balances,
        name='Your Current Path',
        line=dict(color=RED, width=2.5),
        mode='lines+markers',
        marker=dict(size=4)
    ))
    
    # Dynamic label for recommended path
    if current_rate >= 0.10:
        rec_label = f'Your Excellent Path ({current_rate:.0%} savings)'
    else:
        rec_label = 'Recommended Path (10% savings)'
    
    fig.add_trace(go.Scatter(
        x=yrs, y=recommended_balances,
        name=rec_label,
        line=dict(color=GREEN, width=3),
        mode='lines+markers',
        marker=dict(size=4)
    ))
    
    # Add annotations for key milestones
    for i, year in enumerate([5, 10, 20, min(30, years)]):
        if year <= years:
            idx = year - 1
            additional_amount = additional_savings[idx]
            if additional_amount > 0:
                fig.add_annotation(
                    x=year, y=recommended_balances[idx],
                    text=f"+${additional_amount:,.0f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=GREEN,
                    font=dict(size=10, color=GREEN),
                    ax=40,
                    ay=-40
                )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='rgba(240,237,230,0.7)', size=12),
        height=400,
        yaxis=dict(
            tickprefix='$', 
            tickformat=',.0f',
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        xaxis=dict(
            title='Years', 
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(240,237,230,0.6)'),
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='left', 
            x=0
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def chart_savings_impact_breakdown(income, expenses, years=30):
    """Show the impact of following savings recommendations on wealth building."""
    current_monthly = max(income - expenses, 0) / 12
    recommended_monthly = max(income * 0.10 / 12, current_monthly)  # Ensure recommended is at least current
    
    # Calculate projections
    current_final = _compound(0, current_monthly, 0.06, years)[-1]
    recommended_final = _compound(0, recommended_monthly, 0.07, years)[-1]
    additional_wealth = recommended_final - current_final
    
    # Calculate contribution vs growth breakdown
    current_contrib = current_monthly * 12 * years
    recommended_contrib = recommended_monthly * 12 * years
    additional_contrib = recommended_contrib - current_contrib
    
    additional_growth = additional_wealth - additional_contrib
    
    categories = ['Current Path', 'Following Recommendations']
    contributions = [current_contrib, recommended_contrib]
    growth = [current_final - current_contrib, additional_growth]
    total = [current_final, recommended_final]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Your Contributions',
        x=categories,
        y=contributions,
        marker_color=GREEN2,
        text=[f'${c:,.0f}' for c in contributions],
        textposition='inside',
        textfont=dict(color='rgba(240,237,230,0.8)', size=11)
    ))
    
    fig.add_trace(go.Bar(
        name='Investment Growth',
        x=categories,
        y=growth,
        marker_color=GREEN,
        text=[f'${g:,.0f}' for g in growth],
        textposition='inside',
        textfont=dict(color='rgba(240,237,230,0.8)', size=11)
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='rgba(240,237,230,0.7)', size=12),
        height=350,
        barmode='stack',
        yaxis=dict(
            tickprefix='$', 
            tickformat=',.0f',
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(240,237,230,0.6)'),
            orientation='h', 
            yanchor='bottom', 
            y=1.02
        ),
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    
    return fig


def get_savings_recommendations(income, expenses):
    """Display actionable savings recommendations with impact."""
    current_savings_rate = max(income - expenses, 0) / income if income > 0 else 0
    recommended_rate = 0.10
    
    monthly_income = income / 12
    current_monthly_savings = max(income - expenses, 0) / 12
    recommended_monthly_savings = income * recommended_rate / 12
    monthly_gap = max(recommended_monthly_savings - current_monthly_savings, 0)
    
    # Create recommendation categories
    recommendations = []
    
    # Dynamic recommendations based on current savings rate
    if current_savings_rate < recommended_rate:
        # User needs to increase savings
        recommendations.append({
            'category': 'Increase Savings Rate',
            'current': f'{current_savings_rate:.1%}',
            'target': f'{recommended_rate:.1%}',
            'monthly_impact': f'+${monthly_gap:,.0f}',
            'annual_impact': f'+${monthly_gap*12:,.0f}',
            'priority': 'High',
            'actions': [
                'Automate transfers to savings',
                'Review and cancel unused subscriptions',
                'Negotiate bills (insurance, phone, internet)',
                'Use cashback apps for everyday purchases'
            ]
        })
    else:
        # Good saver - focus on consistency
        recommendations.append({
            'category': 'Maintain Strong Savings Habit',
            'current': f'{current_savings_rate:.1%}',
            'target': 'Continue Current Path',
            'monthly_impact': f'${current_monthly_savings:,.0f}',
            'annual_impact': f'${current_monthly_savings*12:,.0f}',
            'priority': 'Low',
            'actions': [
                'Automate investment contributions',
                'Rebalance portfolio quarterly',
                'Track investment performance',
                'Consider increasing contributions with raises'
            ]
        })
    
    # Emergency fund recommendation - adjust logic for high savers
    emergency_target_3mo = expenses * 0.25
    emergency_target_6mo = expenses * 0.5
    # For high savers, assume they may already have substantial emergency fund
    current_emergency = min(current_monthly_savings * 12, income * 0.10)  # More realistic assumption
    
    if current_emergency < emergency_target_3mo:
        recommendations.append({
            'category': 'Build Emergency Fund',
            'current': f'${current_emergency:,.0f}',
            'target': f'${emergency_target_3mo:,.0f}',
            'monthly_impact': f'${min(current_monthly_savings * 0.2, 500):,.0f}',  # Conservative portion
            'annual_impact': f'${min(current_monthly_savings * 0.2, 500)*12:,.0f}',
            'priority': 'High' if current_savings_rate < 0.20 else 'Medium',
            'actions': [
                'Open high-yield savings account',
                'Start with $1,000 mini-fund',
                'Automate monthly contributions',
                'Use windfalls (tax refunds, bonuses) to boost'
            ]
        })
    
    # Investment optimization - make it relevant for all income levels
    if income > 30000:
        if current_savings_rate >= 0.20:
            investment_category = 'Investment Portfolio Optimization'
            current_status = 'Growing investments'
            target_status = 'Balanced diversification'
            actions = [
                'Maximize employer 401(k) match',
                'Open Roth IRA for tax-free growth',
                'Consider HSA if eligible',
                'Add index fund diversification'
            ]
        else:
            investment_category = 'Start Investment Journey'
            current_status = 'Beginning investor'
            target_status = 'Consistent investing'
            actions = [
                'Open employer 401(k) with match',
                'Start with low-cost index funds',
                'Set up automatic investing',
                'Learn basic investment principles'
            ]
        
        recommendations.append({
            'category': investment_category,
            'current': current_status,
            'target': target_status,
            'monthly_impact': f'${recommended_monthly_savings:,.0f}',
            'annual_impact': f'${recommended_monthly_savings*12:,.0f}',
            'priority': 'High' if current_savings_rate < 0.20 else 'Medium',
            'actions': actions
        })
    
    return recommendations


def chart_net_worth_waterfall(income, expenses, years_horizon):
    """Waterfall showing cumulative wealth components."""
    monthly_sav = max(income - expenses, 0) / 12
    rec_monthly = income * 0.10 / 12

    extra_monthly = max(rec_monthly - monthly_sav, 0)
    investment_growth = _compound(0, rec_monthly, 0.07, years_horizon)[-1]
    total_contrib = rec_monthly * 12 * years_horizon
    growth_only = investment_growth - total_contrib
    tax_drag = investment_growth * 0.15  # rough estimate

    labels = ['Gross Contributions', 'Market Growth', 'Tax Drag (est.)', 'Net Wealth']
    values = [total_contrib, growth_only, -tax_drag, investment_growth - tax_drag]
    measures = ['relative', 'relative', 'relative', 'total']
    colors = [GREEN2, GREEN, RED, '#f0ede6']

    fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=measures,
        x=labels, y=values,
        connector=dict(line=dict(color='rgba(138,195,90,0.2)', width=1)),
        increasing=dict(marker=dict(color=GREEN)),
        decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color='#f0ede6')),
        texttemplate='$%{y:,.0f}',
        textposition='outside',
        textfont=dict(color='rgba(240,237,230,0.6)', size=11),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='rgba(240,237,230,0.7)', size=12),
        height=320,
        yaxis=dict(
            tickprefix='$', 
            tickformat=',.0f',
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        xaxis=dict(
            gridcolor='rgba(138,195,90,0.08)',
            zerolinecolor='rgba(138,195,90,0.15)',
            tickfont=dict(color='rgba(240,237,230,0.5)')
        ),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    inject_dash_css()

    income, expenses, family, bracket, sr, done = _get_user_data()

    # ── TOPBAR ──
    st.markdown("""
    <div class="dash-topbar">
        <div class="dash-brand">
            <span class="live-dot"></span>
            FinWise
        </div>
        <div class="nav-pills">
            <a href="?page=landing" class="nav-pill-link" target="_blank">Home</a>
            <a href="?page=chatbot" class="nav-pill-link">Advisor</a>
            <a href="?page=dashboard" class="nav-pill-link active">Dashboard</a>
        </div>
        <div style="font-size:.78rem; color:rgba(240,237,230,.3); letter-spacing:.06em;">
            Financial Projection Center
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dash-page">', unsafe_allow_html=True)

    # ── NAV BUTTONS ──
    c1, c2, c3, _ = st.columns([1, 1, 1, 5])
    with c1:
        if st.button("↩ Advisor", key="advisor_btn"):
            st.query_params.page = "chatbot"
            st.rerun()

    # ── NO DATA STATE ──
    if not done or income == 0:
        st.markdown("""
        <div class="no-data-banner">
            <div class="no-data-icon">📊</div>
            <div class="no-data-title">No financial profile yet</div>
            <div class="no-data-sub">
                Complete a conversation with FinWise first — your projections will appear here automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Demo mode with sample data
        st.markdown("""
        <div class="info-box">
            💡 <strong>Demo mode:</strong> Showing sample projections for a $75,000 income household.
            Complete the advisor flow to see <em>your</em> numbers.
        </div>
        """, unsafe_allow_html=True)

        income   = 75_000
        expenses = 55_000
        family   = 3
        bracket  = 'Middle Income Families'
        sr       = (income - expenses) / income

    savings      = max(income - expenses, 0)
    monthly_sav  = savings / 12
    rec_monthly  = income * 0.10 / 12
    rec_annual   = income * 0.10

    # ── HERO ──
    current_year = datetime.now().year
    st.markdown(f"""
    <div class="dash-hero">
        <div class="dash-eyebrow">Financial Projection Dashboard · {current_year}</div>
        <h1 class="dash-title">Your <em>wealth trajectory,</em><br>made visible.</h1>
        <p class="dash-sub">
            Based on your {bracket.lower()} profile with ${income:,.0f} annual income.
            Projections assume market returns of 6–8% annually.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── CONTROLS ──
    with st.expander("⚙️  Adjust projection parameters", expanded=False):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            horizon = st.selectbox("Time horizon", [10, 15, 20, 25, 30, 35, 40], index=0)
        with cc2:
            assumed_return = st.selectbox("Assumed annual return", ["6%", "7%", "8%", "9%"], index=1)
            ret_rate = int(assumed_return.strip('%')) / 100
        with cc3:
            savings_pct = st.selectbox("Target savings rate", ["10%", "15%", "20%"], index=0)
            target_rate = int(savings_pct.strip('%')) / 100

    # ── KPI CARDS ──
    monthly_income  = income / 12
    current_savings_rate = max(income - expenses, 0) / income if income > 0 else 0
    
    # If user already saves more than target, use their current rate
    if current_savings_rate >= target_rate:
        effective_target_rate = current_savings_rate
        gap_monthly = 0  # No gap needed- they're already doing great!
    else:
        effective_target_rate = target_rate
        gap_monthly = (income * effective_target_rate / 12) - monthly_sav
    
    target_monthly = income * effective_target_rate / 12
    projected_30yr  = _compound(0, target_monthly, ret_rate, horizon)[-1]
    ef_months_away  = max(expenses * 0.25 - savings, 0) / max(monthly_sav, 1)

    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)

    # Single KPI Box: All Financial Metrics
    st.markdown(f"""
    <div class="kpi-box" style="background: rgba(138,195,90,0.05); border: 1px solid rgba(138,195,90,0.2); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem;">
            <div>
                <div style="font-size: 0.8rem; color: rgba(240,237,230,0.4); margin-bottom: 0.5rem;">MONTHLY INCOME</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 1.5rem; color: #8ac35a; margin-bottom: 0.3rem;">
                    ${monthly_income:,.0f}
                </div>
                <div style="font-size: 0.85rem; color: rgba(240,237,230,0.5);">
                    gross, pre-tax
                </div>
            </div>
            <div>
                <div style="font-size: 0.8rem; color: rgba(240,237,230,0.4); margin-bottom: 0.5rem;">CURRENT SAVINGS</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 1.5rem; color: #5a9c8a; margin-bottom: 0.3rem;">
                    ${monthly_sav:,.0f}/mo
                </div>
                <div style="font-size: 0.85rem; color: rgba(240,237,230,0.5);">
                    Rate: {sr:.1%}
                </div>
            </div>
            <div>
                <div style="font-size: 0.8rem; color: rgba(240,237,230,0.4); margin-bottom: 0.5rem;">TARGET MONTHLY Δ</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 1.5rem; color: #d4b84a; margin-bottom: 0.3rem;">
                    {"✓ On Track" if gap_monthly == 0 else f"${gap_monthly:,.0f}"}
                </div>
                <div style="font-size: 0.85rem; color: rgba(240,237,230,0.5);">
                    {"Already exceeding target!" if gap_monthly == 0 else f"to reach {savings_pct} rate"}
                </div>
            </div>
            <div>
                <div style="font-size: 0.8rem; color: rgba(240,237,230,0.4); margin-bottom: 0.5rem;">{horizon}-YR WEALTH</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 1.5rem; color: #8ac35a; margin-bottom: 0.3rem;">
                    {"${projected_30yr/1e6:.2f}M" if projected_30yr>1e6 else f"${projected_30yr:,.0f}"}
                </div>
                <div style="font-size: 0.85rem; color: rgba(240,237,230,0.5);">
                    at {effective_target_rate:.0%} rate
                </div>
            </div>
            <div>
                <div style="font-size: 0.8rem; color: rgba(240,237,230,0.4); margin-bottom: 0.5rem;">FAMILY SIZE</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 1.5rem; color: #5a9c8a; margin-bottom: 0.3rem;">
                    {family} {'person' if family==1 else 'people'}
                </div>
                <div style="font-size: 0.85rem; color: rgba(240,237,230,0.5);">
                    {bracket.replace(' Families','').replace(' Households','')}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── ENCOURAGING MESSAGE ──
    if sr >= 0.20:  # High savings rate
        st.markdown(f"""
        <div style="background: rgba(138,195,90,0.1); border-left: 4px solid #8ac35a; padding: 1rem 1.5rem; margin: 1.5rem 0; border-radius: 8px;">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; color: #8ac35a; margin-bottom: 0.5rem;">
                You're already on an excellent financial path!
            </div>
            <div style="color: rgba(240,237,230,0.8); line-height: 1.5;">
                ${projected_30yr:,.0f} With your impressive {sr:.0%} savings rate, you're on track to build substantial wealth. 
                Continuing your current strategy at 7% annual return. You're already exceeding typical recommendations!
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── CHART ROW 1: Savings Comparison & Impact Analysis ──
    st.markdown("""
    <div class="section-head">
        <h3>Savings Comparison: Current vs Recommended</h3>
        <span class="section-badge">Impact Analysis</span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Your Savings Trajectory</div>
            <div class="chart-desc">
                See the difference between your current savings path and following FinWise recommendations.
                The shaded area shows your potential additional wealth.
            </div>
        """, unsafe_allow_html=True)
        fig_comparison = chart_savings_comparison(income, expenses, years=horizon)
        st.plotly_chart(fig_comparison, use_container_width=True, config={'displayModeBar': False})
        
        # Show wealth milestones up to 20 years
        milestone_years = [1, 3, 5, 10, 15, 20, 25, 30]
        milestone_years = [y for y in milestone_years if y <= horizon]
        balances = _compound(0, target_monthly, ret_rate, max(milestone_years))
        max_bal = balances[-1] if balances else 1
        
        for y in milestone_years:
            bal = _compound(0, target_monthly, ret_rate, y)[-1]
            pct = bal / max_bal * 100
            fmt = f"${bal/1e6:.2f}M" if bal >= 1e6 else f"${bal:,.0f}"
            st.markdown(f"""
            <div class="milestone-row">
                <div class="ms-year">Yr {y}</div>
                <div class="ms-bar-wrap">
                    <div class="ms-bar" style="width:{pct:.1f}%"></div>
                </div>
                <div class="ms-amount">
                    {fmt}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Wealth Building Impact</div>
            <div class="chart-desc">How your contributions and investment growth compare over time.</div>
        """, unsafe_allow_html=True)
        fig_impact = chart_savings_impact_breakdown(income, expenses, years=horizon)
        st.plotly_chart(fig_impact, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SAVINGS RECOMMENDATIONS SECTION ──
    st.markdown("""
    <div class="section-head">
        <h3>Personalized Savings Action Plan</h3>
        <span class="section-badge">Actionable Steps</span>
    </div>
    """, unsafe_allow_html=True)

    recommendations = get_savings_recommendations(income, expenses)
    
    for i, rec in enumerate(recommendations):
        priority_color = {'High': '#c05a5a', 'Medium': '#d4b84a', 'Low': '#8ac35a'}.get(rec['priority'], '#8ac35a')
        
        st.markdown(f"""
        <div class="chart-card" style="border-left: 4px solid {priority_color};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                <div>
                    <div class="chart-title" style="margin-bottom: 0.3rem;">{rec['category']}</div>
                    <div class="chart-desc" style="margin-bottom: 0;">
                        Priority: <span style="color: {priority_color}; font-weight: 500;">{rec['priority']}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.85rem; color: rgba(240,237,230,0.5);">Current → Target</div>
                    <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; color: #f0ede6; margin-bottom: 0.2rem;">
                        {rec['current']} → {rec['target']}
                    </div>
                    <div style="font-size: 0.9rem; color: #8ac35a;">
                        {rec['monthly_impact']}/mo
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        for action in rec['actions']:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.3rem; margin-left: 1rem;">
                <span style="color: #8ac35a; font-size: 0.8rem;">•</span>
                <span style="font-size: 0.85rem; color: rgba(240,237,230,0.6);">{action}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)

    # ── INTERACTIVE SAVINGS GOAL TRACKER ──
    st.markdown("""
    <div class="section-head">
        <h3>Interactive Savings Goal Tracker</h3>
        <span class="section-badge">Plan Your Future</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    
    # Goal selection controls
    goal_col1, goal_col2, goal_col3 = st.columns([2, 2, 2])
    
    with goal_col1:
        goal_type = st.selectbox(
            "Savings Goal",
            ["Emergency Fund", "House Down Payment", "Retirement", "Education Fund", "Custom Goal"],
            key="goal_type"
        )
    
    with goal_col2:
        if goal_type == "Emergency Fund":
            target_amount = expenses * 0.25  # 3 months
            goal_amount = st.number_input(
                "Target Amount ($)",
                value=int(target_amount),
                min_value=1000,
                step=1000,
                key="emergency_target"
            )
        elif goal_type == "House Down Payment":
            goal_amount = st.number_input(
                "Target Amount ($)",
                value=50000,
                min_value=10000,
                step=5000,
                key="house_target"
            )
        elif goal_type == "Retirement":
            goal_amount = st.number_input(
                "Target Amount ($)",
                value=1000000,
                min_value=100000,
                step=50000,
                key="retirement_target"
            )
        elif goal_type == "Education Fund":
            goal_amount = st.number_input(
                "Target Amount ($)",
                value=100000,
                min_value=10000,
                step=5000,
                key="education_target"
            )
        else:
            goal_amount = st.number_input(
                "Target Amount ($)",
                value=25000,
                min_value=1000,
                step=1000,
                key="custom_target"
            )
    
    with goal_col3:
        time_horizon = st.selectbox(
            "Time Horizon (Years)",
            [1, 2, 3, 5, 10, 15, 20, 25, 30],
            index=4
        )
    
    # Calculate goal scenarios
    current_monthly = max(income - expenses, 0) / 12
    recommended_monthly = income * 0.10 / 12
    
    current_time_to_goal = goal_amount / max(current_monthly, 1) / 12
    recommended_time_to_goal = goal_amount / max(recommended_monthly, 1) / 12
    # Display goal analysis
    analysis_col1, analysis_col2, analysis_col3 = st.columns([1, 1, 1])
    
    with analysis_col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.02); border-radius: 12px;">
            <div style="font-size: 0.8rem; color: rgba(240,237,230,0.4); margin-bottom: 0.5rem;">Current Pace</div>
            <div style="font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #f0ede6; margin-bottom: 0.3rem;">
                {current_time_to_goal:.1f} years
            </div>
            <div style="font-size: 0.85rem; color: rgba(240,237,230,0.5);">
                ${current_monthly:,.0f}/mo
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with analysis_col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: rgba(138,195,90,0.08); border-radius: 12px; border: 1px solid rgba(138,195,90,0.2);">
            <div style="font-size: 0.8rem; color: rgba(240,237,230,0.4); margin-bottom: 0.5rem;">With Recommendations</div>
            <div style="font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #8ac35a; margin-bottom: 0.3rem;">
                {recommended_time_to_goal:.1f} years
            </div>
            <div style="font-size: 0.85rem; color: rgba(240,237,230,0.5);">
                ${recommended_monthly:,.0f}/mo
            </div>
        </div>
        """, unsafe_allow_html=True)

    with analysis_col3:
        monthly_needed = goal_amount / (time_horizon * 12)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: rgba(212,184,74,0.08); border-radius: 12px; border: 1px solid rgba(212,184,74,0.2);">
            <div style="font-size: 0.8rem; color: rgba(240,237,230,0.4); margin-bottom: 0.5rem;">Monthly Needed</div>
            <div style="font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #d4b84a; margin-bottom: 0.3rem;">
                ${monthly_needed:,.0f}
            </div>
            <div style="font-size: 0.85rem; color: rgba(240,237,230,0.5);">
                to reach {goal_type} goal
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── BUDGET ANALYSIS SECTION ──
    st.markdown("""
    <div class="section-head">
        <h3>Budget Analysis &amp; Emergency Fund</h3>
        <span class="section-badge">Monthly Health</span>
    </div>
    """, unsafe_allow_html=True)

    l2, r2 = st.columns(2)

    with l2:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">50/30/20 Budget vs. Actual</div>
            <div class="chart-desc">
                The recommended rule splits income into needs, wants, and savings. How close are you?
            </div>
        """, unsafe_allow_html=True)
        fig3 = chart_monthly_budget(income, expenses)
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Current vs Recommended Savings</div>
            <div class="chart-desc">
                Compare your current savings trajectory with the recommended 10% savings rate over time.
            </div>
        """, unsafe_allow_html=True)
        fig4 = chart_savings_comparison(income, expenses)
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── CHART ROW 3: Retirement stacked + Waterfall ──
    st.markdown("""
    <div class="section-head">
        <h3>Retirement &amp; Net Worth Breakdown</h3>
        <span class="section-badge">Long-Term View</span>
    </div>
    """, unsafe_allow_html=True)

    l3, r3 = st.columns(2)

    with l3:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Retirement Accumulation</div>
            <div class="chart-desc">
                Stacked: your raw contributions (dark) vs. compound market growth (light) at each 5-year mark.
            </div>
        """, unsafe_allow_html=True)
        fig5 = chart_retirement_runway(income, sr, years=horizon)
        st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r3:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Net Wealth Waterfall</div>
            <div class="chart-desc">
                Gross contributions → market growth → estimated tax drag → final net wealth projection.
            </div>
        """, unsafe_allow_html=True)
        fig6 = chart_net_worth_waterfall(income, expenses, horizon)
        st.plotly_chart(fig6, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown("""
    <div style="
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(240,237,230,0.1);
    ">
        2026 FinWise — Financial Recommendation System · Developed by Yogita &amp; Suchita
        · Projections are illustrative and not financial advice.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close dash-page


if __name__ == "__main__":
    main()