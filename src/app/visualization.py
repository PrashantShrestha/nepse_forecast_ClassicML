import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def display_signal(signal, confidence):
    """Display trading signal with color coding"""
    color = "#4CAF50" if signal == "Buy" else "#F44336" if signal == "Sell" else "#2196F3"
    confidence_color = "#4CAF50" if confidence > 0.7 else "#FF9800" if confidence > 0.5 else "#F44336"
    
    st.markdown(f"""
    <div style="
        border: 2px solid {color};
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        background-color: rgba{(*bytes.fromhex(color[1:]), 0.1)};
    ">
        <h2 style="color: {color};">{signal}</h2>
        <h3>Confidence: <span style="color: {confidence_color};">{confidence:.0%}</span></h3>
    </div>
    """, unsafe_allow_html=True)

def plot_feature_importance(model, features):
    """Visualize feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance.head(10), x='Feature', y='Importance', 
                     title='Top 10 Important Features')
        st.plotly_chart(fig)
    else:
        st.warning("Feature importance not available for this model")

def plot_price_history(symbol, price_data):
    """Plot historical prices with indicators"""
    if price_data.empty or symbol not in price_data['Symbol'].unique():
        st.warning(f"No price data available for {symbol}")
        return
    
    symbol_data = price_data[price_data['Symbol'] == symbol].sort_values('Date')
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=symbol_data['Date'], 
        y=symbol_data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#1f77b4')
    ))
    
    # Moving average
    if '5d_ma' in symbol_data.columns:
        fig.add_trace(go.Scatter(
            x=symbol_data['Date'], 
            y=symbol_data['5d_ma'],
            mode='lines',
            name='5-day MA',
            line=dict(dash='dot', color='#ff7f0e')
        ))
    
    # Set title and axis labels
    fig.update_layout(
        title=f"{symbol} Price History",
        xaxis_title='Date',
        yaxis_title='Price (NPR)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig)
    
    # RSI subplot
    if '14d_rsi' in symbol_data.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=symbol_data['Date'], 
            y=symbol_data['14d_rsi'],
            mode='lines',
            name='RSI',
            line=dict(color='#9467bd')
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(
            title="RSI Indicator",
            xaxis_title='Date',
            yaxis_title='RSI',
            height=300
        )
        st.plotly_chart(fig_rsi)

def plot_performance_history(history):
    """Plot model performance over time"""
    if history.empty:
        return
        
    fig = px.line(
        history, 
        x='date', 
        y=['accuracy', 'precision', 'recall', 'f1'],
        title='Model Performance Over Time',
        labels={'value': 'Score', 'date': 'Date'},
        height=400
    )
    
    fig.update_layout(
        legend_title='Metrics',
        hovermode="x unified"
    )
    
    st.plotly_chart(fig)