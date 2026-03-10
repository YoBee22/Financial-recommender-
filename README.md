# Personalized-Financial-Recommendation-System
Suchita Sharma, Yogita Bisht
An **AI-powered financial assistant** that provides personalized financial guidance using **Machine Learning, Retrieval-Augmented Generation (RAG), and rule-based systems**.  
The system analyzes user income, spending behavior, demographics, and risk tolerance to deliver actionable recommendations for budgeting, savings, investing, and retirement planning.

## Features
- ETF & mutual fund suggestions based on risk profile   
- Smart savings recommendations (HYSA, CDs)  
- Retirement planning guidance (IRA, 401k)  
- Identify spending optimization opportunities

## Approach
A **hybrid architecture** combining:
- Machine Learning for personalization  
- Rule-based systems for financial constraints  
- RAG for contextual explanations

## Goal
Make accessible, personalized financial guidance available to everyone and improve long-term financial decision-making through explainable AI.

## current progress
We completed EDA and feature engineering on all three datasets — Census-Income (199K users), US Funds (26K ETFs/mutual funds), and Consumer Expenditure Survey (23K households). K-Means clustering on Census data identified 10 user personas (silhouette 0.483) mapped to fund risk tiers and financial product recommendations. Supervised classifiers (Logistic Regression, Random Forest, XGBoost) were built to predict risk tolerance from raw demographics for real-time user classification. 

Next steps: CE Survey spending persona clustering, content-based filtering for fund matching via cosine similarity, and integration with the rule engine, RAG pipeline, and FastAPI/Streamlit application.
