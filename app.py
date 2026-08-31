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
# ----------------------------------------------------
# NEW FUNCTION: CACHED GEMINI ANALYSIS ENGINE
# ----------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_ai_analysis(api_key, prompt):
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-3.7-flash', 
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig(thinking_level="MEDIUM")
            )
        )
        return response.text
    except Exception as e:
        return f"Error connecting to the scouting network: {str(e)}"

df_clean = load_and_clean_data()

if df_clean.empty:
    st.stop()

# ----------------------------------------------------
# 2. SIDEBAR NAVIGATION & FILTERS
# ----------------------------------------------------
st.sidebar.title("🏈 Auction Analysis Engine")
st.sidebar.markdown("Analyzing historical draft inefficiencies to deliver a structural championship blueprint.")

all_years = sorted(df_clean['Year'].unique())
all_positions = sorted(df_clean['Position'].unique())
all_managers = sorted(df_clean['Manager'].unique())

view_option = st.sidebar.radio(
    "Select Analysis View",
    ["Executive Blueprint (2026 Plan)", "Manager Spending Habits", "Draft Position Lulls", "Player Market Value"]
)

# ----------------------------------------------------
# VIEW 0: EXECUTIVE BLUEPRINT (2026 DRAFT PLAN)
# ----------------------------------------------------
if view_option == "Executive Blueprint (2026 Plan)":
    st.title("📋 Executive Blueprint: 2026 Draft Strategy Room")
    
    with st.expander("📖 Quick-Start How-To Guide (Click to Expand)", expanded=True):
        st.markdown("""
        Welcome to the **Auction Analysis Engine**! This tool is engineered to break down your home league's historical trends (2021–2025) and help you exploit manager tendencies. Here is how to navigate the strategy room:
        *   **1. Executive Blueprint (Home Page):** View the lifetime capital ROI leaderboard to instantly pinpoint who overspends and who finds bargains. Review the custom multi-column tactical blueprint for the 2026 draft.
        *   **2. Manager Spending Habits:** Use the filters to select a specific manager and position. A custom chart will display their budget habits, followed by a live **Gemini AI Scouting Report** detailing their Superflex anomalies and how to beat them.
        *   **3. Draft Position Lulls:** Toggle draft years to look at the historical flow of when positions fly off the board. Target the 'valleys' to secure players where market competition cools down.
        *   **4. Player Market Value:** Analyze individual asset pricing. Grouped horizontal bars let you compare actual league paid percentages directly against baseline market consensus values.
        """)
    
    st.markdown("### 📈 Macro Trends & Manager Anomalies")
    st.markdown("A unified analysis overview breaking down historical draft capital flow and tactical advantages.")

    df_clean['Premium_Paid'] = df_clean['Cap_Percent'] - df_clean['Consensus_AAV_Cap_Percent']
    avg_premium_by_manager = df_clean.groupby('Manager')['Premium_Paid'].mean().reset_index()
    
    most_aggressive = avg_premium_by_manager.sort_values(by='Premium_Paid', ascending=False).iloc[0]['Manager']
    biggest_bargain_hunter = avg_premium_by_manager.sort_values(by='Premium_Paid', ascending=True).iloc[0]['Manager']
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="League Economy", value="10 Teams / $200 Cap", delta="Superflex (2 QB Config)")
    with m2:
        st.metric(label="Most Aggressive Bidder", value=most_aggressive, delta="Pays Highest Asset Premiums", delta_color="inverse")
    with m3:
        st.metric(label="Top Value Exploiter", value=biggest_bargain_hunter, delta="Consistently Underbuys Market")
        
    st.markdown("---")
    
    st.subheader("📊 Career Draft Capital Value Leaderboard")
    st.markdown("This tracker displays the cumulative cap percentage saved (Bargain) or overpaid (Premium) by each manager across all drafted positions relative to consensus market rates.")
    
    roi_df = df_clean.groupby('Manager').agg(
        Total_Spent_Cap=('Cap_Percent', 'sum'),
        Total_Consensus_Value=('Consensus_AAV_Cap_Percent', 'sum'),
        Total_Players_Drafted=('Player', 'count')
    ).reset_index()
    
    roi_df['Net_Value_Differential'] = roi_df['Total_Consensus_Value'] - roi_df['Total_Spent_Cap']
    roi_df = roi_df.sort_values(by='Net_Value_Differential', ascending=False).reset_index(drop=True)
    
    display_roi_df = roi_df.copy()
    display_roi_df['Net_Value_Differential'] = display_roi_df['Net_Value_Differential'].apply(lambda x: f"🟢 +{x:.1f}% Saved" if x >= 0 else f"🔴 {x:.1f}% Overpaid")
    display_roi_df.columns = ['Manager Name', 'Total Cap Spent (%)', 'Market Value Secured (%)', 'Total Assets Drafted', 'Lifetime Capital Efficiency ROI']
    
    st.dataframe(display_roi_df, use_container_width=True, hide_index=True)
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"#### 🔍 Key Manager Behaviors")
        st.markdown(
            f"• **{most_aggressive}** repeatedly resets positional markets by driving premium asset prices significantly above baseline consensus AAV.\n"
            f"• **{biggest_bargain_hunter}** systematically drops out of early-round bidding runs, waiting for late-draft drops where visual inflation resets.\n"
            f"• Historical trendlines show that home-league managers heavily favor emotional bidding, over-indexing on high-tier starters."
        )
    with c2:
        st.markdown("#### 📉 Positional Market Trends")
        st.markdown(
            "• **Quarterbacks (QB):** Due to the Superflex structure, early tiers face massive inflationary bidding curves. The mid-tier values are consistently squeezed.\n"
            f"• **Running Backs & Wide Receivers:** Highly volatile valleys exist between pick ranges 35-70 where league draft velocity cools down significantly.\n"
            "• **Tight Ends (TE):** Consistently draft below international consensus values unless targeting the elite top-3 overall stars."
        )
    with c3:
        st.markdown("#### 🎯 2026 Draft Game Plan")
        st.markdown(
            "1. **Exploit the Squeeze:** Let aggressive managers wipe out their capital matching high-premium bidding wars early.\n"
            "2. **Target the Pick 40-70 Lull:** Bank multiple high-floor WRs and RB2s in the dead zones where the league economy traditionally dries up.\n"
            "3. **Superflex Asset Shielding:** Do not leave the draft without locking down 3 starting QBs; mid-tier league value provides an elite cost-adjusted window."
        )

# ----------------------------------------------------
# VIEW 1: MANAGER SPENDING HABITS
# ----------------------------------------------------
elif view_option == "Manager Spending Habits":
    st.header("💰 How do Managers Value Positions?")
    st.markdown("Track historical budget allocation versus market consensus to break down positional investment habits.")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_manager = st.selectbox("Select Manager", all_managers)
    with col_f2:
        selected_position = st.selectbox("Select Position", all_positions)
    
    summary_df = df_clean.groupby(['Manager', 'Year', 'Position']).agg(
        Total_Cap_Percent=('Cap_Percent', 'sum'),
        Total_Consensus_AAV_Cap_Percent=('Consensus_AAV_Cap_Percent', 'sum')
    ).reset_index()
    
    subset_df = summary_df[
        (summary_df['Position'] == selected_position) & 
        (summary_df['Manager'] == selected_manager)
    ]
    
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
    
    if not melted_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        sns.lineplot(
            data=melted_df,
            x='Year',
            y='Cap_Value',
            style='Cap_Metric',
            hue='Cap_Metric',
            markers=True,
            dashes=[(1, 0), (2, 2)],
            palette='tab10', 
            linewidth=3,
            ax=ax
        )
        ax.set_title(f'{selected_manager}: Budget Spent vs Market Value ({selected_position})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Cap Percentage (%)')
        plt.xticks(all_years)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(title='Metric Details', loc='best')
        st.pyplot(fig)
    else:
        st.warning(f"No data available for {selected_manager} drafting {selected_position} positions.")

    st.markdown("---")
    st.subheader(f"🤖 Gemini Superflex Scouting Profile: {selected_manager}")
    
    api_key = st.secrets.get("GEMINI_API_KEY") or None
    
    if not api_key:
        st.info("💡 Add your `GEMINI_API_KEY` to app Secrets to activate live dynamic scouting reports here.")
    else:
        try:
            client = genai.Client(api_key=api_key)
            prompt = (
                f"You are a sharp, elite fantasy football analytics expert specializing in high-stakes Superflex auction leagues. "
                f"Deconstruct this multi-year positional data for manager '{selected_manager}' regarding the '{selected_position}' position.\n"
                f"{data_summary_text}\n\n"
                f"Write a comprehensive scouting report tailored for the 2026 draft. Your analysis must cover:\n"
                f"1. **Positional Economy Evaluation**: Compare their real cap% spend trends versus consensus market metrics. Are they a hyper-aggressive bidder or a value hunter?\n"
                f"2. **Superflex Draft Anomalies**: Evaluate if this manager panics during aggressive early position runs (like overspending on mid-tier QBs to secure a second starter) or if they successfully weaponize capital during cross-positional value dips.\n"
                f"3. **2026 Tactical Counter-Strategy**: Give a specific rule or psychological trap to use against this exact manager in the upcoming draft room to force them into bad math or exhaustion of budget.\n"
                f"Tone: Dense, direct, analytical, and highly strategic. Use bold headers and clean structure for maximum scannability."
            )
            
            with st.spinner("Analyzing high-stakes Superflex market patterns..."):
                # CALLED THE NEW CACHED ENGINE HERE
                analysis_text = get_cached_ai_analysis(api_key, prompt)
                st.write(analysis_text)
                
        except Exception as e:
            st.error(f"Scouting database connection timed out: {str(e)}")
            
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
        palette='muted',
        linewidth=2,
        ax=ax
    )
    ax.set_title(f'Distribution of Pick Numbers by Position for {selected_year}', fontsize=14)
    ax.set_xlabel('Pick Number (Higher = Picked Later)')
    ax.set_ylabel('Draft Frequency Density')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

    st.markdown("---")
    st.subheader(f"🤖 Gemini AI Draft Flow Assessment ({selected_year})")
    
    api_key = st.secrets.get("GEMINI_API_KEY") or None
    
    if not api_key:
        st.info("💡 Add your `GEMINI_API_KEY` to app Secrets to activate live dynamic draft flow insights here.")
    else:
        try:
            lull_summary = subset_year_df.groupby('Position')['Pick Number'].agg(['min', 'mean', 'max']).reset_index()
            data_summary_text = lull_summary.to_string(index=False)
            
            client = genai.Client(api_key=api_key)
            prompt = (
                f"You are a fantasy football data scientist studying draft economy curves. "
                f"Analyze these draft distribution stats for the year {selected_year} detailing the pick numbers when positions went off the board:\n"
                f"{data_summary_text}\n\n"
                f"Provide a 3-sentence macro summary of the draft room environment.\n"
                f"Sentence 1: Detail where the heaviest positional run took place based on the pick numbers.\n"
                f"Sentence 2: Identify any obvious drafting lulls or windows where value dropped significantly.\n"
                f"Sentence 3: Provide a distinct tactical rule of thumb for exploiting this dynamic in future drafts. Keep it scannable with bold highlights."
            )
            with st.spinner("Analyzing drafting waves and valleys..."):
                # CALLED THE NEW CACHED ENGINE HERE
                analysis_text = get_cached_ai_analysis(api_key, prompt)
                st.write(analysis_text)
        except Exception as e:
            st.error(f"Could not load AI draft analysis: {str(e)}")


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

    st.markdown("---")
    st.subheader(f"🤖 Gemini Player Value Audit ({selected_year} - {selected_position}s)")
    
    api_key = st.secrets.get("GEMINI_API_KEY") or None
    
    if not api_key:
        st.info("💡 Add your `GEMINI_API_KEY` to app Secrets to activate live dynamic asset valuations here.")
    else:
        try:
            audit_df = df_clean[(df_clean['Year'] == selected_year) & (df_clean['Position'] == selected_position)].copy()
            audit_df['Discrepancy'] = audit_df['Cap_Percent'] - audit_df['Consensus_AAV_Cap_Percent']
            
            top_overpaid = audit_df.sort_values(by='Discrepancy', ascending=False).head(2)[['Player', 'Manager', 'Discrepancy']].to_string(index=False)
            top_bargains = audit_df.sort_values(by='Discrepancy', ascending=True).head(2)[['Player', 'Manager', 'Discrepancy']].to_string(index=False)
            
            client = genai.Client(api_key=api_key)
            prompt = (
                f"You are a fantasy football financial ledger auditor reviewing an auction draft league. "
                f"Analyze these top pricing anomalies for the position {selected_position} in the draft year {selected_year}.\n\n"
                f"Top overpaid assets (Positive means they paid a huge premium above market value):\n{top_overpaid}\n\n"
                f"Top bargain assets (Negative means they saved money below market value):\n{top_bargains}\n\n"
                f"Provide a crisp 3-sentence economic teardown.\n"
                f"Sentence 1: Highlight who the biggest overpayment was and why that manager compromised their budget economy.\n"
                f"Sentence 2: Identify the best value bargain won in the room and the manager who secured it.\n"
                f"Sentence 3: Outline a pricing strategy warning for handling this player tier in future draft rooms based on these behaviors. Use bold text elements."
            )
            with st.spinner("Auditing individual player transaction ledgers..."):
                # CALLED THE NEW CACHED ENGINE HERE
                analysis_text = get_cached_ai_analysis(api_key, prompt)
                st.write(analysis_text)
        except Exception as e:
            st.error(f"Could not load player price audit: {str(e)}")





