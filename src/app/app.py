import streamlit as st
import pandas as pd
import os
import numpy as np
from datetime import datetime
from src.modeling.predictor import Predictor
from src.utils.config_loader import load_config
from src.utils.data_manager import load_features
from src.app.visualization import (
    display_signal, 
    plot_feature_importance, 
    plot_price_history,
    plot_performance_history
)

def main():
    # Load configuration
    config = load_config()

    # Page configuration
    st.set_page_config(
        page_title="NEPSE Trading Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None

    # Sidebar controls
    st.sidebar.header("Configuration")
    symbol = st.sidebar.selectbox(
        "Select Stock", 
        options=['HIDCLP', 'NICA', 'SCB', 'NIFRA', 'SHL', 'CBBL', 'GBIME', 'NMB'],
        index=0
    )

    horizon_map = {
        'Next Day': 'next_day',
        '3-Day': '3day',
        'Weekly': 'weekly'
    }
    horizon_label = st.sidebar.radio(
        "Prediction Horizon",
        options=list(horizon_map.keys()),
        index=0
    )
    horizon = horizon_map[horizon_label]

    broker_mode = st.sidebar.selectbox(
        "Broker Feature Mode",
        options=['Relative', 'Absolute'],
        index=0
    ).lower()

    # Update config based on user selection
    config['training']['horizon'] = horizon
    config['training']['broker_mode'] = broker_mode

    # Initialize predictor
    try:
        if st.session_state.predictor is None or \
           st.session_state.predictor.horizon != horizon:
            st.session_state.predictor = Predictor(horizon=horizon)
    except Exception as e:
        st.error(f"Error initializing predictor: {str(e)}")
        st.stop()

    # Main content area
    st.title("NEPSE Trading Signal Assistant")
    st.markdown("""
        **Real-time trading signal predictions** based on technical indicators and broker behavior analysis.
    """)

    # Get prediction
    prediction = st.session_state.predictor.predict(symbol)

    # Display results
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(f"Signal for {symbol}")
        display_signal(prediction['signal'], prediction['confidence'])
        
        if 'probabilities' in prediction:
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                'Signal': list(prediction['probabilities'].keys()),
                'Probability': list(prediction['probabilities'].values())
            })
            st.bar_chart(prob_df.set_index('Signal'))

    with col2:
        # Show feature importance
        if 'features' in prediction and st.session_state.predictor.model:
            features_df = pd.DataFrame([prediction['features']])
            plot_feature_importance(st.session_state.predictor.model, features_df)
        
        # Show price history
        price_data = load_features('technical')
        plot_price_history(symbol, price_data)

    # Additional sections
    st.subheader("Model Information")
    if st.session_state.predictor:
        st.write(f"**Model Version:** {prediction.get('model_version', 'Unknown')}")
        st.write(f"**Horizon:** {horizon_label}")
        st.write(f"**Broker Mode:** {broker_mode.title()}")
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Performance history
    try:
        eval_csv = os.path.join(config['logs']['log_dir'], "evaluation_history.csv")
        if os.path.exists(eval_csv):
            eval_history = pd.read_csv(eval_csv)
            st.subheader("Model Performance History")
            plot_performance_history(eval_history)
            
            # Latest metrics
            if not eval_history.empty:
                latest = eval_history.iloc[-1]
                cols = st.columns(4)
                cols[0].metric("Accuracy", f"{latest['accuracy']:.2%}")
                cols[1].metric("Precision", f"{latest['precision']:.2%}")
                cols[2].metric("Recall", f"{latest['recall']:.2%}")
                cols[3].metric("F1 Score", f"{latest['f1']:.2%}")
    except Exception as e:
        st.warning(f"Performance data not available: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("")

if __name__ == "__main__":
    main()