import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Zach's Auction Draft Analyzer",
    page_icon="🏈",
    layout="wide"
)

# ----------------------------------------------------
# 1. DATA LOADING & CACHING
# ----------------------------------------------------
@st.cache_data
def load_and_clean_data():
    # Update this path to where your file sits relative to app.py
    # For sharing, it's best to keep the CSV in the same folder as this script!
    try:
        df = pd.read_csv('zach-final-data-table - data_table.csv')
    except FileNotFoundError:
        st.error("Could not find 'zach-final-data-table - data_table.csv'. Please make sure it's in the same directory as app.py.")
        return pd.DataFrame()

    # Drop unwanted columns if present
    if 'Player_Team' in df.columns:
        df = df.drop(columns=['Player_Team'])
        
    # Clean position names and strip whitespace
    df['Position'] = df['Position'].astype(str).str.strip()
    
    # Filter out K and D/ST as per notebook logic
    df = df[~df['Position'].isin(['K', 'D/ST'])]
    
    # Ensure numeric columns are handled gracefully
    df['Pick Number'] = pd.to_numeric(df['Pick Number'], errors='coerce')
    df = df.dropna(subset=['Pick Number'])
    df['Year'] = df['Year'].astype(int)
    
    return df

df_clean = load_and_clean_data()

if df_clean.empty:
    st.stop()

# ----------------------------------------------------
# 2. SIDEBAR NAVIGATION & FILTERS
# ----------------------------------------------------
st.sidebar.title("🏈 Auction Analysis")
st.sidebar.markdown("Use this dashboard to find inefficiencies and manager tendencies in Zach's Superflex league.")

# Global Filter Options
all_years = sorted(df_clean['Year'].unique())
all_positions = sorted(df_clean['Position'].unique())

view_option = st.sidebar.radio(
    "Select Analysis View",
    ["Manager Spending Habits", "Draft Position Lulls", "Player Market Value"]
)

# ----------------------------------------------------
# VIEW 1: MANAGER SPENDING HABITS
# ----------------------------------------------------
if view_option == "Manager Spending Habits":
    st.header("💰 How do Managers Value Positions?")
    st.markdown("This view displays the historical percent of cap a manager spent on a position versus the consensus value of the players they selected.")
    
    # Filter by position dynamically
    selected_position = st.selectbox("Select Position to Analyze", all_positions)
    
    # Process group-by logic from notebook
    summary_df = df_clean.groupby(['Manager', 'Year', 'Position']).agg(
        Total_Cap_Percent=('Cap_Percent', 'sum'),
        Total_Consensus_AAV_Cap_Percent=('Consensus_AAV_Cap_Percent', 'sum')
    ).reset_index()
    
    subset_df = summary_df[summary_df['Position'] == selected_position]
    
    # Reshape data cleanly for visualization
    melted_df = subset_df.melt(
        id_vars=['Manager', 'Year', 'Position'],
        value_vars=['Total_Cap_Percent', 'Total_Consensus_AAV_Cap_Percent'],
        var_name='Cap_Metric',
        value_name='Cap_Value'
    )
    melted_df['Cap_Metric'] = melted_df['Cap_Metric'].map({
        'Total_Cap_Percent': 'Actual League Spent %',
        'Total_Consensus_AAV_Cap_Percent': 'Consensus AAV Value %'
    })
    
    # Render with Matplotlib/Seaborn as per notebook instructions
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=melted_df,
        x='Year',
        y='Cap_Value',
        hue='Manager',
        style='Cap_Metric',
        markers=True,
        dashes=[(1, 0), (2, 2)],
        palette='deep',
        ax=ax
    )
    ax.set_title(f'Cap % vs. Consensus AAV Cap % for {selected_position} Over Time', fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel('Cap Percentage (%)')
    plt.xticks(all_years)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Manager & Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    st.pyplot(fig)
    
    with st.expander("View Raw Spreadsheet Data"):
        st.dataframe(subset_df, use_container_width=True)

# ----------------------------------------------------
# VIEW 2: DRAFT POSITION LULLS
# ----------------------------------------------------
elif view_option == "Draft Position Lulls":
    st.header("⏳ Draft Runs and Market Lulls")
    st.markdown("Identify exactly where in the draft flow specific positions are heavily targeted or completely ignored.")
    
    selected_year = st.selectbox("Select Draft Year", all_years)
    subset_year_df = df_clean[df_clean['Year'] == selected_year]
    
    # Kernel Density Estimate Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.kdeplot(
        data=subset_year_df,
        x='Pick Number',
        hue='Position',
        fill=True,
        common_norm=False,
        palette='viridis',
        linewidth=2,
        ax=ax
    )
    ax.set_title(f'Distribution of Pick Numbers by Position for {selected_year}', fontsize=14)
    ax.set_xlabel('Pick Number (Higher = Picked Later)')
    ax.set_ylabel('Draft Frequency Density')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)
    st.caption("Peaks show a high volume of that position going off the board simultaneously (a run). Valleys indicate drafting lulls.")

# ----------------------------------------------------
# VIEW 3: PLAYER MARKET VALUE
# ----------------------------------------------------
elif view_option == "Player Market Value":
    st.header("⚖️ League Pricing vs. Consensus Valuation")
    st.markdown("Compare actual costs against market expectations to spot which managers aggressively overpay or find bargains.")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox("Select Year", all_years, key="player_year")
    with col2:
        selected_position = st.selectbox("Select Position", all_positions, key="player_pos")
        
    # Melt data format for side-by-side player comparisons
    melted_players = df_clean.melt(
        id_vars=['Manager', 'Year', 'Position', 'Player', 'Team', 'Pick Number', 'Amount'],
        value_vars=['Cap_Percent', 'Consensus_AAV_Cap_Percent'],
        var_name='Cap_Metric_Type',
        value_name='Cap_Value_Percent'
    )
    melted_players['Cap_Metric_Type'] = melted_players['Cap_Metric_Type'].map({
        'Cap_Percent': 'League Paid Cap %',
        'Consensus_AAV_Cap_Percent': 'Consensus AAV Cap %'
    })
    
    # Apply user filters
    final_player_df = melted_players[
        (melted_players['Year'] == selected_year) & 
        (melted_players['Position'] == selected_position)
    ]
    
    # Fixed Axis Buffer Calculation
    max_val = melted_players['Cap_Value_Percent'].max() * 1.1
    
    # Plotly Interactivity 
    fig = px.bar(
        final_player_df,
        y='Player',
        x='Cap_Value_Percent',
        color='Cap_Metric_Type',
        orientation='h',
        barmode='group',
        hover_data=['Manager', 'Team', 'Pick Number', 'Amount'],
        labels={
            'Cap_Value_Percent': 'Cap Percentage (%)',
            'Player': 'Player Name',
            'Cap_Metric_Type': 'Valuation'
        },
        height=max(400, len(final_player_df) * 15), # Dynamically sizes based on player count
        range_x=[0, max_val]
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        legend_title="Valuation Type",
        margin=dict(l=150, r=20, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
