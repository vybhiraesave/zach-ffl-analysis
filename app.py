import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from google import genai

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
    try:
        df = pd.read_csv('zach-final-data-table - data_table.csv')
    except FileNotFoundError:
        st.error("Could not find 'zach-final-data-table - data_table.csv'. Please make sure it's in the same directory as app.py.")
        return pd.DataFrame()

    if 'Player_Team' in df.columns:
        df = df.drop(columns=['Player_Team'])
        
    df['Position'] = df['Position'].astype(str).str.strip()
    df = df[~df['Position'].isin(['K', 'D/ST'])]
    
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

all_years = sorted(df_clean['Year'].unique())
all_positions = sorted(df_clean['Position'].unique())
all_managers = sorted(df_clean['Manager'].unique())

view_option = st.sidebar.radio(
    "Select Analysis View",
    ["Manager Spending Habits", "Draft Position Lulls", "Player Market Value"]
)

# ----------------------------------------------------
# VIEW 1: MANAGER SPENDING HABITS
# ----------------------------------------------------
if view_option == "Manager Spending Habits":
    st.header("💰 How do Managers Value Positions?")
    st.markdown("Track historical budget allocation versus market consensus to break down positional investment habits.")
    
    # NEW FILTERS: Filter down to a specific manager and position to clean up the noise!
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_manager = st.selectbox("Select Manager", all_managers)
    with col_f2:
        selected_position = st.selectbox("Select Position", all_positions)
    
    # Process group-by logic from notebook
    summary_df = df_clean.groupby(['Manager', 'Year', 'Position']).agg(
        Total_Cap_Percent=('Cap_Percent', 'sum'),
        Total_Consensus_AAV_Cap_Percent=('Consensus_AAV_Cap_Percent', 'sum')
    ).reset_index()
    
    # Filter by both user criteria
    subset_df = summary_df[
        (summary_df['Position'] == selected_position) & 
        (summary_df['Manager'] == selected_manager)
    ]
    
    # Reshape data cleanly for visualization
    melted_df = subset_df.melt(
        id_vars=['Manager', 'Year', 'Position'],
        value_vars=['Total_Cap_Percent', 'Total_Consensus_AAV_Cap_Percent'],
        var_name='Cap_Metric',
        value_name='Cap_Value'
    )
    melted_df['Cap_Metric'] = melted_df['Cap_Metric'].map({
        'Total_Cap_Percent': 'Actual Budget Spent %',
        'Total_Consensus_AAV_Cap_Percent': 'Consensus Value %'
    })

    # Render customized visualization
    if not melted_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.lineplot(
            data=melted_df,
            x='Year',
            y='Cap_Value',
            style='Cap_Metric',
            hue='Cap_Metric',
            markers=True,
            dashes=[(1, 0), (2, 2)],
            palette=['#1E3A8A', '#94A3B8'], # Crisp contrast colors
            linewidth=2.5,
            ax=ax
        )
        ax.set_title(f'{selected_manager}: Budget Spent vs Market Value ({selected_position})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Cap Percentage (%)')
        plt.xticks(all_years)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title='Metric Details', loc='best')
        st.pyplot(fig)
    else:
        st.warning(f"No data available for {selected_manager} drafting {selected_position} positions.")

    # --- NEW FEATURE: AI-GENERATED INGENUITY FROM GEMINI ---
    st.markdown("---")
    st.subheader(f"🤖 Gemini Analysis: {selected_manager}'s Draft Profile")
    
    # Securely retrieve your user api key if configured, or check for system configurations
    api_key = st.secrets.get("GEMINI_API_KEY") or None
    
    if not api_key:
        st.info("💡 To generate live insights with Gemini, go to your repository settings on Streamlit Cloud and add `GEMINI_API_KEY` to your app Secrets.")
        st.markdown(f"**Draft Notes for {selected_manager}:** Reviewing the trendline chart highlights periods of aggressive premiums or heavy roster value accumulation across the 2021–2025 timelines.")
    else:
        try:
            # Construct a clear data table string to feed directly to Gemini
            data_summary_text = subset_df[['Year', 'Total_Cap_Percent', 'Total_Consensus_AAV_Cap_Percent']].to_string(index=False)
            
            # Initialize client and query the flash model
            client = genai.Client(api_key=api_key)
            
            prompt = (
                f"You are a high-level fantasy football league analyst assessing an auction draft league. "
                f"Analyze this multi-year draft data for fantasy football manager '{selected_manager}' focusing specifically on the '{selected_position}' position. "
                f"Data table (shows how much actual draft cap % they spent vs what the consensus market baseline values were for those players):\n"
                f"{data_summary_text}\n\n"
                f"Provide a punchy, 3-sentence summary in a conversational tone. "
                f"Sentence 1: State whether they typically overspend (aggressive) or underspend (bargain hunting) compared to consensus on this position. "
                f"Sentence 2: Identify any noticeable spikes, years of heavy shift, or tactical behaviors. "
                f"Sentence 3: Provide a quick tactical piece of advice for playing against them in future drafts based on this data. Use bold elements for scannability."
            )
            
            with st.spinner("Gemini is dissecting the auction data room..."):
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt,
                )
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Could not reach Gemini API. Technical Check: {str(e)}")

    with st.expander("View Filtered Spreadsheet Breakdown"):
        st.dataframe(subset_df, use_container_width=True)

# ----------------------------------------------------
# VIEW 2: DRAFT POSITION LULLS
# ----------------------------------------------------
elif view_option == "Draft Position Lulls":
    st.header("⏳ Draft Runs and Market Lulls")
    st.markdown("Identify exactly where in the draft flow specific positions are heavily targeted or completely ignored.")
    
    selected_year = st.selectbox("Select Draft Year", all_years)
    subset_year_df = df_clean[df_clean['Year'] == selected_year]
    
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
    
    final_player_df = melted_players[
        (melted_players['Year'] == selected_year) & 
        (melted_players['Position'] == selected_position)
    ]
    
    max_val = melted_players['Cap_Value_Percent'].max() * 1.1
    
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
        height=max(400, len(final_player_df) * 15),
        range_x=[0, max_val]
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        legend_title="Valuation Type",
        margin=dict(l=150, r=20, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
