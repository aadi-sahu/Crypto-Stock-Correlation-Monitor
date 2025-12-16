"""
Crypto-Stock Price Analysis
Educational financial analytics tool
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from utils import (
    DataFetcher,
    DataPreprocessor,
    CorrelationAnalyzer,
    format_currency,
    format_percentage,
    get_latest_price,
    get_price_change
)

# --- Configuration ---

# Page config
st.set_page_config(
    page_title="Crypto-Stock Price Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom styling (Removed .disclaimer styling)
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.data = None

# Asset Lists (Expanded)
ALL_CRYPTOS = ["Bitcoin", "Ethereum", "Solana", "Ripple", "Dogecoin"]
ALL_STOCKS = ["S&P 500", "NASDAQ 100", "NIFTY 50", "FTSE 100", "DAX"]

# --- Header ---

# Removed "Dashboard" from the title
st.title("📈 Crypto-Stock Price Analysis") 
st.markdown("### Educational Financial Analytics Tool")

# Removed the entire Disclaimer section
# st.markdown("""...""", unsafe_allow_html=True)

# --- Sidebar ---

st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")

# Period selection
periods = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}
selected_period = st.sidebar.selectbox(
    "Time Period",
    list(periods.keys()),
    index=3,
    key='period_select'
)

# Asset selection (Updated with more assets)
st.sidebar.markdown("### 📊 Assets")
cryptos = st.sidebar.multiselect(
    "Cryptocurrencies",
    ALL_CRYPTOS,
    default=["Bitcoin", "Ethereum", "Solana"],
    key='crypto_select'
)
stocks = st.sidebar.multiselect(
    "Stock Indices",
    ALL_STOCKS,
    default=["S&P 500", "NIFTY 50"],
    key='stock_select'
)

# Load button
if st.sidebar.button("🔄 Load Data", type="primary", key='load_btn'):
    with st.spinner("Fetching data..."):
        fetcher = DataFetcher()
        period = periods[selected_period]
        
        # Pass the full list of assets to the fetcher
        all_selected_assets = cryptos + stocks
        all_data = fetcher.fetch_all_assets(period, all_selected_assets) 
        
        filtered_data = {}
        for asset in all_selected_assets:
            if asset in all_data and not all_data[asset].empty:
                filtered_data[asset] = all_data[asset]
        
        if filtered_data:
            st.session_state.data = filtered_data
            st.session_state.data_loaded = True
            st.sidebar.success(f"✅ Data loaded for {len(filtered_data)} assets!")
        else:
            st.session_state.data_loaded = False
            st.session_state.data = None
            st.sidebar.error("❌ Failed to load data. Check asset names or data availability.")

# --- Main Content ---

if st.session_state.data_loaded:
    data = st.session_state.data
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "💹 Price Analysis", "🔗 Correlation"])
    
    # TAB 1: Overview
    with tab1:
        st.header("📊 Market Overview")
        
        # Metrics
        cols = st.columns(len(data))
        for idx, (name, df) in enumerate(data.items()):
            with cols[idx % len(cols)]: # Use modulo to handle more assets than columns
                price = get_latest_price(df)
                change_1d = get_price_change(df, 1)
                change_7d = get_price_change(df, 7)
                
                st.metric(
                    label=name,
                    value=format_currency(price),
                    delta=format_percentage(change_1d) if change_1d else "N/A"
                )
                if change_7d:
                    st.caption(f"7d: {format_percentage(change_7d)}")
        
        st.markdown("---")
        
        # Normalized chart
        st.subheader("📈 Normalized Price Comparison")
        fig = go.Figure()
        for name, df in data.items():
            if not df.empty and df['Close'].iloc[0] != 0:
                normalized = (df['Close'] / df['Close'].iloc[0]) * 100
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized,
                    mode='lines',
                    name=name,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Normalized Price (Base 100)",
            height=500,
            template="plotly_dark",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats table
        st.subheader("📋 Statistics")
        stats = []
        for name, df in data.items():
            try:
                current = float(df['Close'].iloc[-1])
                highest = float(df['Close'].max())
                lowest = float(df['Close'].min())
            except:
                current = highest = lowest = float('nan')
            
            avg_vol = "N/A"
            if 'Volume' in df.columns:
                try:
                    vol = df['Volume'].mean()
                    if pd.notna(vol) and vol > 0:
                        avg_vol = f"{vol:,.0f}"
                except:
                    pass
            
            stats.append({
                'Asset': name,
                'Current': format_currency(current),
                'High': format_currency(highest),
                'Low': format_currency(lowest),
                'Avg Volume': avg_vol
            })
        
        st.dataframe(pd.DataFrame(stats).fillna("N/A"), use_container_width=True)
    
    # TAB 2: Price Analysis
    with tab2:
        st.header("💹 Price Analysis")
        
        asset = st.selectbox("Select Asset", list(data.keys()), key='asset_select')
        df = data[asset]
        df_proc = DataPreprocessor.add_all_features(df.copy())
        
        # Price + MA chart
        st.subheader(f"{asset} - Price & Moving Averages")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )
        
        fig.add_trace(
            go.Scatter(x=df_proc.index, y=df_proc['Close'],
                       name='Price', line=dict(color='#00ff00', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_proc.index, y=df_proc['MA_7'],
                       name='MA 7', line=dict(color='#ffff00', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_proc.index, y=df_proc['MA_30'],
                       name='MA 30', line=dict(color='#ff00ff', width=1)),
            row=1, col=1
        )
        
        if 'Volume' in df_proc.columns and not df_proc['Volume'].isnull().all():
            fig.add_trace(
                go.Bar(x=df_proc.index, y=df_proc['Volume'],
                       name='Volume', marker_color='#1f77b4'),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            template="plotly_dark",
            showlegend=True
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI
        if 'RSI' in df_proc.columns and not df_proc['RSI'].isnull().all():
            st.subheader("📊 RSI Indicator")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(
                go.Scatter(x=df_proc.index, y=df_proc['RSI'],
                           name='RSI', line=dict(color='#00ffff', width=2))
            )
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(
                height=300,
                template="plotly_dark",
                yaxis_title="RSI"
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Returns
        if 'Returns' in df_proc.columns and not df_proc['Returns'].isnull().all():
            st.subheader("📊 Returns Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter out NaN/inf before plotting
                returns_data = df_proc['Returns'].replace([float('inf'), float('-inf')], float('nan')).dropna()
                if not returns_data.empty:
                    fig_hist = px.histogram(
                        returns_data,
                        x='Returns',
                        nbins=50,
                        title="Daily Returns"
                    )
                    fig_hist.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.warning("Not enough returns data to display histogram.")
            
            with col2:
                st.markdown("**Statistics:**")
                st.dataframe(df_proc['Returns'].describe())
    
    # TAB 3: Correlation
    with tab3:
        st.header("🔗 Correlation Analysis")
        
        analyzer = CorrelationAnalyzer()
        corr_matrix = analyzer.calculate_correlation_matrix(data)
        
        # Heatmap
        if not corr_matrix.empty:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                fmt='.2f',
                ax=ax
            )
            plt.title("Asset Correlation Matrix")
            st.pyplot(fig)
            
            # Insights
            st.subheader("🔍 Insights")
            for i, a1 in enumerate(corr_matrix.columns):
                for j, a2 in enumerate(corr_matrix.columns):
                    if i < j:
                        val = corr_matrix.loc[a1, a2]
                        strength = analyzer.get_correlation_strength(val)
                        
                        c1, c2, c3 = st.columns([2, 1, 2])
                        with c1:
                            st.write(f"**{a1}** ↔ **{a2}**")
                        with c2:
                            st.metric("Corr", f"{val:.3f}")
                        with c3:
                            st.write(f"**{strength}**")
            
            # Rolling correlation
            if len(data) >= 2:
                st.subheader("📈 Rolling Correlation")
                
                assets = list(data.keys())
                # Ensure a2 default index is valid
                default_a2_index = 0
                if len(assets) > 1 and assets[0] == assets[1]:
                    default_a2_index = 1
                
                a1 = st.selectbox("First Asset", assets, key='corr_a1')
                
                # Filter out a1 for a2 selection
                available_a2 = [a for a in assets if a != a1]
                if available_a2:
                    a2 = st.selectbox("Second Asset", available_a2, key='corr_a2')
                    window = st.slider("Window (days)", 10, 90, 30, key='corr_window')
                    
                    roll_corr = analyzer.calculate_rolling_correlation(
                        data[a1], data[a2], window
                    )
                    
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=roll_corr.index, y=roll_corr,
                                   mode='lines',
                                   line=dict(color='#ff00ff', width=2))
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="white")
                    fig.update_layout(
                        template="plotly_dark",
                        height=400,
                        title=f"{a1} vs {a2} - {window}d Rolling Correlation",
                        yaxis_title="Correlation"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select at least two assets to view Rolling Correlation.")
        else:
            st.info("Not enough data loaded to calculate correlation.")

else:
    st.info("👈 Select assets and click 'Load Data' to begin")
    
    st.markdown("## 🌟 Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        ### 📊 Overview
        - Latest prices
        - Price changes (1D/7D)
        - Normalized comparison
        """)
    with c2:
        st.markdown("""
        ### 💹 Analysis
        - Price charts
        - Moving averages (MA)
        - Relative Strength Index (RSI)
        """)
    with c3:
        st.markdown("""
        ### 🔗 Correlation
        - Heatmaps
        - Rolling correlations
        - Correlation strength
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>📈 Crypto-Stock Price Analysis</p>
    <p>Data: Yahoo Finance | Educational Use Only</p>
</div>
""", unsafe_allow_html=True)