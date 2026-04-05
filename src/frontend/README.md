# Streamlit Components

This folder contains all Streamlit applications and pages for the FinWise Financial Recommendation System.

## Files Structure

```
streamlit/
├── __init__.py          # Package initialization
├── app.py              # Main application with navigation
├── landing_page.py     # Landing page component
├── streamlit_chatbot.py # Chatbot interface with RAG
├── dashboard.py        # Financial analytics dashboard
└── README.md           # This file
```

## Components

### `app.py`
- Main entry point for the application
- Handles navigation between pages
- Manages session state and URL parameters
- Routes to different components based on page parameter

### `landing_page.py`
- Welcome and introduction page
- Navigation buttons to chatbot and dashboard
- Overview of FinWise features

### `streamlit_chatbot.py`
- Interactive financial advisor chatbot
- RAG-enhanced responses using financial knowledge base
- User profile collection and classification
- Integration with ETF/Mutual fund recommendations

### `dashboard.py`
- Financial analytics and visualizations
- User profile summary cards
- Savings projections and comparisons
- Interactive charts using Plotly

## Running the Application

```bash
# From the src directory
streamlit run streamlit/app.py

# Or from the project root
cd src && streamlit run streamlit/app.py
```

## Navigation

- **Landing Page**: `/` (default)
- **Chatbot**: `?page=chatbot`
- **Dashboard**: `?page=dashboard`

## Dependencies

All components share the same dependencies as the main project:
- Streamlit
- Plotly
- Pandas
- NumPy
- Custom modules from parent directory
