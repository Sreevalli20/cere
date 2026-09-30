"""
ToneTrace - Conversation Intelligence
CEREBRO 2026 - PS-01: Tone Intelligence in Conversations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parser import ConversationParser
from src.engine import ToneEngine
from src.temporal import TemporalAnalyzer
from src.explainability import ExplainabilityEngine
from src.demo import get_demo_conversation


# Page configuration
st.set_page_config(
    page_title="ToneTrace",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #9D4EDD, #00D4FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .tagline {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1A1F3A;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #2A3040;
    }
    .stAlert {
        background-color: #1A1F3A;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'conversation_df' not in st.session_state:
    st.session_state.conversation_df = None
if 'analyzed_df' not in st.session_state:
    st.session_state.analyzed_df = None
if 'summary' not in st.session_state:
    st.session_state.summary = None


def render_landing_page():
    """Render landing page"""
    st.markdown('<h1 class="main-header">TONETRACE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Conversation Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">"See the conversation before it becomes a conflict."</p>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Conversation")
        uploaded_file = st.file_uploader(
            "Supported formats: CSV, JSON, WhatsApp TXT, Slack-style, Discord-style",
            type=['csv', 'json', 'txt'],
            help="Maximum file size: 5 MB"
        )

        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                parser = ConversationParser()
                df = parser.parse(content, uploaded_file.name)

                st.session_state.conversation_df = df
                st.success(f"Loaded {len(df)} messages successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error parsing file: {str(e)}")

    with col2:
        st.subheader("Try Demo")
        if st.button("Load Demo Conversation"):
            try:
                df = get_demo_conversation()
                st.session_state.conversation_df = df
                st.success("Demo conversation loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading demo: {str(e)}")

    st.markdown("---")
    st.info("ToneTrace analyzes conversational progression, not just individual messages. "
            "It tracks how tone evolves across turns to identify escalation and turning points.")


def render_overview():
    """Render overview dashboard"""
    st.header("Conversation Overview")

    df = st.session_state.analyzed_df
    summary = st.session_state.summary

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Messages", summary['total_messages'])

    with col2:
        escalation = summary['estimated_escalation']
        st.metric("Est. Escalation", f"{escalation:.1f}")

    with col3:
        peak = summary['peak_escalation']
        st.metric("Peak Escalation", f"{peak:.1f}")

    with col4:
        st.metric("Turning Points", summary['turning_points'])

    with col5:
        st.metric("Speakers", summary['speakers'])

    st.markdown("---")

    # Signal counts
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Sarcasm Signals", summary['sarcasm_signals'])

    with col2:
        st.metric("Passive-Aggression", summary['passive_aggression_signals'])

    with col3:
        st.metric("Hostility Signals", summary['hostility_signals'])

    with col4:
        st.metric("Dominant Sentiment", summary['dominant_sentiment'].capitalize())

    st.markdown("---")

    # Emotion Arc
    st.subheader("Emotion Arc")

    fig = go.Figure()

    # Add sentiment trace
    fig.add_trace(go.Scatter(
        x=df['turn_index'],
        y=df['sentiment_score'],
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='#00D4FF', width=2),
        marker=dict(size=6)
    ))

    # Add escalation trace
    fig.add_trace(go.Scatter(
        x=df['turn_index'],
        y=df['escalation'] / 100.0,  # Normalize to -1 to 1 range
        mode='lines+markers',
        name='Escalation',
        line=dict(color='#FF6B6B', width=2),
        marker=dict(size=6)
    ))

    # Add turning point markers
    turning_points = df[df['is_turning_point']]
    if len(turning_points) > 0:
        fig.add_trace(go.Scatter(
            x=turning_points['turn_index'],
            y=turning_points['sentiment_score'],
            mode='markers',
            name='Turning Points',
            marker=dict(color='#FF4444', size=12, symbol='star'),
            text=turning_points['speaker'],
            hovertemplate='<b>Message %{x}</b><br>Speaker: %{text}<br>Sentiment: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title="Tone Evolution Over Time",
        xaxis_title="Message Turn",
        yaxis_title="Signal Intensity",
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Escalation Timeline
    st.subheader("Escalation Timeline")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df['turn_index'],
        y=df['escalation'],
        mode='lines+markers',
        name='Escalation',
        line=dict(color='#FF6B6B', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.2)'
    ))

    # Add threshold line
    fig2.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Elevated")

    fig2.update_layout(
        title="Estimated Conversation Escalation",
        xaxis_title="Message Turn",
        yaxis_title="Escalation Score (0-100)",
        hovermode='x unified',
        template='plotly_dark',
        height=300
    )

    st.plotly_chart(fig2, use_container_width=True)


def render_conversation_view():
    """Render conversation with tone analysis"""
    st.header("Conversation View")

    df = st.session_state.analyzed_df

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        speaker_filter = st.multiselect(
            "Filter by Speaker",
            options=df['speaker'].unique().tolist(),
            default=df['speaker'].unique().tolist()
        )

    with col2:
        sentiment_filter = st.multiselect(
            "Filter by Sentiment",
            options=['positive', 'neutral', 'negative'],
            default=['positive', 'neutral', 'negative']
        )

    # Apply filters
    filtered_df = df[
        df['speaker'].isin(speaker_filter) &
        df['sentiment'].isin(sentiment_filter)
    ].copy()

    # Display messages
    for idx, row in filtered_df.iterrows():
        with st.expander(f"Message {row['turn_index'] + 1} - {row['speaker']}", expanded=False):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**{row['text']}**")
                st.caption(f"Timestamp: {row.get('timestamp', 'N/A')}")

            with col2:
                sentiment_emoji = {
                    'positive': '😊',
                    'neutral': '😐',
                    'negative': '😠'
                }
                st.markdown(f"**Sentiment:** {sentiment_emoji.get(row['sentiment'], '😐')} {row['sentiment'].capitalize()}")

            # Signal badges
            signals = []
            if row['sarcasm'] > 0.5:
                signals.append("🔸 Sarcasm")
            if row['passive_aggression'] > 0.5:
                signals.append("🔸 Passive-Aggression")
            if row['hostility'] > 0.5:
                signals.append("🔸 Hostility")
            if row['irony'] > 0.5:
                signals.append("🔸 Irony")

            if signals:
                st.markdown(" ".join(signals))

            # Detailed analysis
            with st.expander("View Analysis", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Estimated Tone Values:**")
                    st.write(f"Sentiment Score: {row['sentiment_score']:.2f}")
                    st.write(f"Joy: {row['joy']:.2f}")
                    st.write(f"Sadness: {row['sadness']:.2f}")
                    st.write(f"Anger: {row['anger']:.2f}")
                    st.write(f"Fear: {row['fear']:.2f}")
                    st.write(f"Frustration: {row['frustration']:.2f}")

                with col2:
                    st.markdown("**Signal Intensity:**")
                    st.write(f"Sarcasm: {row['sarcasm']:.2f}")
                    st.write(f"Irony: {row['irony']:.2f}")
                    st.write(f"Passive Aggression: {row['passive_aggression']:.2f}")
                    st.write(f"Hostility: {row['hostility']:.2f}")
                    st.write(f"Escalation: {row['escalation']:.1f}")

                if idx > 0:
                    st.markdown("**Change from Previous Message:**")
                    st.write(f"Sentiment Delta: {row['sentiment_score_delta']:.2f}")
                    st.write(f"Anger Delta: {row['anger_delta']:.2f}")
                    st.write(f"Frustration Delta: {row['frustration_delta']:.2f}")
                    st.write(f"Hostility Delta: {row['hostility_delta']:.2f}")


def render_conflict_signals():
    """Render conflict signals and turning points"""
    st.header("Conflict Signals")

    df = st.session_state.analyzed_df
    summary = st.session_state.summary

    # Overall escalation
    st.subheader("Overall Estimated Escalation")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current", f"{summary['estimated_escalation']:.1f}")
    with col2:
        st.metric("Peak", f"{summary['peak_escalation']:.1f}")
    with col3:
        st.metric("Turning Points", summary['turning_points'])

    st.markdown("---")

    # Turning points
    st.subheader("Candidate Turning Points")

    turning_points = df[df['is_turning_point']]

    if len(turning_points) == 0:
        st.info("No significant turning points detected in this conversation.")
    else:
        for idx, row in turning_points.iterrows():
            with st.expander(f"Turning Point at Message {row['turn_index'] + 1}", expanded=False):
                st.markdown(f"**Speaker:** {row['speaker']}")
                st.markdown(f"**Message:** {row['text']}")
                st.markdown(f"**Timestamp:** {row.get('timestamp', 'N/A')}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Previous Tone:**")
                    if idx > 0:
                        prev = df.iloc[idx - 1]
                        st.write(f"Sentiment: {prev['sentiment']} ({prev['sentiment_score']:.2f})")
                        st.write(f"Frustration: {prev['frustration']:.2f}")
                        st.write(f"Hostility: {prev['hostility']:.2f}")

                with col2:
                    st.markdown("**Current Tone:**")
                    st.write(f"Sentiment: {row['sentiment']} ({row['sentiment_score']:.2f})")
                    st.write(f"Frustration: {row['frustration']:.2f}")
                    st.write(f"Hostility: {row['hostility']:.2f}")

                st.markdown("**Tone Delta:**")
                st.write(f"Sentiment: {row['sentiment_score_delta']:.2f}")
                st.write(f"Frustration: {row['frustration_delta']:.2f}")
                st.write(f"Hostility: {row['hostility_delta']:.2f}")

                st.markdown("**Escalation Score:**")
                st.write(f"{row['escalation']:.1f}/100")

    st.markdown("---")

    # Largest tone changes
    st.subheader("Largest Tone Changes")

    df['tone_change_magnitude'] = (
        df['sentiment_score_delta'].abs() +
        df['frustration_delta'].abs() +
        df['hostility_delta'].abs() +
        df['anger_delta'].abs()
    )

    top_changes = df.nlargest(5, 'tone_change_magnitude')

    for idx, row in top_changes.iterrows():
        if row['tone_change_magnitude'] > 0:
            with st.expander(f"Change at Message {row['turn_index'] + 1} (Magnitude: {row['tone_change_magnitude']:.2f})", expanded=False):
                st.markdown(f"**Speaker:** {row['speaker']}")
                st.markdown(f"**Message:** {row['text']}")
                st.write(f"Sentiment Delta: {row['sentiment_score_delta']:.2f}")
                st.write(f"Frustration Delta: {row['frustration_delta']:.2f}")
                st.write(f"Hostility Delta: {row['hostility_delta']:.2f}")


def render_speaker_analysis():
    """Render per-speaker analysis"""
    st.header("Speaker Analysis")

    df = st.session_state.analyzed_df

    analyzer = TemporalAnalyzer()
    speaker_df = analyzer.get_speaker_analysis(df)

    for idx, row in speaker_df.iterrows():
        with st.expander(f"Speaker: {row['speaker']}", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Messages", row['message_count'])
            with col2:
                st.metric("Avg Sentiment", f"{row['avg_sentiment']:.2f}")
            with col3:
                st.metric("Tone Volatility", f"{row['tone_volatility']:.2f}")

            st.markdown("**Signal Counts:**")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Frustration Signals", f"{row['avg_frustration']:.2f}")
            with col2:
                st.metric("Hostility Signals", f"{row['avg_hostility']:.2f}")
            with col3:
                st.metric("Sarcasm Signals", row['sarcasm_signals'])

            st.markdown("**Passive-Aggression Signals:**")
            st.metric("Count", row['passive_aggression_signals'])


def render_evidence():
    """Render evidence view"""
    st.header("Evidence")

    df = st.session_state.analyzed_df

    explainability = ExplainabilityEngine()

    # Strongest tone change
    st.subheader("Strongest Tone Change")

    strongest = explainability.get_strongest_tone_change(df)

    if strongest:
        st.markdown(f"**Message {strongest['message_number']}**")
        st.markdown(f"**Speaker:** {strongest['speaker']}")
        st.markdown(f"**Message:** {strongest['message']}")
        st.markdown(f"**Timestamp:** {strongest['timestamp']}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Previous Tone:**")
            for key, value in strongest['previous_tone'].items():
                st.write(f"{key}: {value}")

        with col2:
            st.markdown("**Current Tone:**")
            for key, value in strongest['current_tone'].items():
                st.write(f"{key}: {value}")

        st.markdown("**Tone Delta:**")
        for key, value in strongest['tone_delta'].items():
            st.write(f"{key}: {value}")

        st.markdown("**Supporting Signals:**")
        for signal in strongest['supporting_signals']:
            st.write(f"- {signal}")

        st.markdown("**Why Detected:**")
        st.write(strongest['why_detected'])
    else:
        st.info("No significant tone changes detected.")

    st.markdown("---")

    # Limitations
    st.subheader("Limitations")
    st.warning("""
    **Important Limitations:**

    - ToneTrace estimates linguistic signals, not psychological states
    - Sarcasm and passive-aggression detection are uncertain and context-dependent
    - Language, culture, dialect, and code-switching can affect results
    - Results should not be used for psychological diagnosis
    - Results should not be used for employment, disciplinary, legal, or other high-stakes decisions
    - The system is intentionally lightweight and API-free
    """)


def render_methodology():
    """Render methodology page"""
    st.header("Methodology")

    st.markdown("""
    ## Analysis Pipeline

    ToneTrace follows this analysis pipeline:

    1. **Input**
       - User uploads conversation (CSV, JSON, WhatsApp TXT, Slack-style, Discord-style)

    2. **Parsing**
       - Normalize conversation into standard format
       - Extract speaker, text, timestamp, turn index

    3. **Feature Extraction**
       - Lexical features (positive/negative words)
       - Structural features (punctuation, capitalization)
       - Pattern features (disagreement markers, blame indicators)
       - Cue features (sarcasm, irony, passive-aggression cues)

    4. **Tone Estimation**
       - Sentiment (positive/neutral/negative, -1 to +1 score)
       - Emotion signals (joy, sadness, anger, fear, frustration)
       - Sarcasm signal
       - Irony signal
       - Passive-aggression signal
       - Hostility signal

    5. **Temporal Comparison**
       - Calculate deltas (change from previous message)
       - Rolling averages for temporal smoothing
       - Recent context window (5 messages)

    6. **Escalation Estimation**
       - Combine multiple signals (sentiment decrease, anger increase, frustration increase, hostility increase, sarcasm, passive-aggression, volatility)
       - Weighted formula documented in code
       - Score 0-100

    7. **Turning-Point Detection**
       - Detect significant tone changes relative to recent context
       - Requires substantial change in multiple dimensions
       - Elevated escalation threshold

    8. **Evidence Extraction**
       - Identify supporting linguistic cues
       - Generate explanations for detections
       - Show tone deltas and context

    9. **Visualization**
       - Emotion arc over time
       - Escalation timeline
       - Turning point markers
       - Per-speaker statistics

    ## Architecture

    ToneTrace is intentionally:
    - **Lightweight**: No external models, no GPU required
    - **API-free**: No external AI services, no API keys
    - **Memory-efficient**: Designed for ~512 MB RAM constraint
    - **Fast**: No model downloads, no training at startup

    ## Signal Interpretation

    All signals are **estimated linguistic signals**, not claims about a person's actual emotional state.

    - **Sentiment**: Estimated positive/negative/neutral tone (-1 to +1)
    - **Emotions**: Estimated joy, sadness, anger, fear, frustration (0 to 1)
    - **Sarcasm**: Possible sarcasm signal based on contextual patterns
    - **Irony**: Possible irony signal based on semantic contrast
    - **Passive-Aggression**: Possible passive-aggressive signal based on dismissive patterns
    - **Hostility**: Possible hostility signal based on aggressive vocabulary
    - **Escalation**: Estimated conversation escalation (0 to 100)

    ## Responsible AI

    ToneTrace is designed for exploratory conversation analysis, not for high-stakes decision-making.

    See the **Evidence** page for detailed limitations.
    """)


def analyze_conversation(df):
    """Run full analysis pipeline"""
    engine = ToneEngine()
    analyzer = TemporalAnalyzer()

    # Analyze tone
    df = engine.analyze(df)

    # Analyze temporal patterns
    df = analyzer.analyze(df)

    # Get summary
    summary = analyzer.get_conversation_summary(df)

    return df, summary


def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("ToneTrace")
    st.sidebar.markdown("Conversation Intelligence")

    if st.session_state.conversation_df is None:
        page = "landing"
    else:
        page = st.sidebar.radio(
            "Navigate",
            ["Overview", "Conversation", "Conflict Signals", "Speaker Analysis", "Evidence", "Methodology"]
        )

    # Render page
    if page == "landing":
        render_landing_page()
    elif page == "Overview":
        if st.session_state.analyzed_df is None:
            # Analyze conversation
            with st.spinner("Analyzing conversation..."):
                df, summary = analyze_conversation(st.session_state.conversation_df)
                st.session_state.analyzed_df = df
                st.session_state.summary = summary
        render_overview()
    elif page == "Conversation":
        if st.session_state.analyzed_df is None:
            with st.spinner("Analyzing conversation..."):
                df, summary = analyze_conversation(st.session_state.conversation_df)
                st.session_state.analyzed_df = df
                st.session_state.summary = summary
        render_conversation_view()
    elif page == "Conflict Signals":
        if st.session_state.analyzed_df is None:
            with st.spinner("Analyzing conversation..."):
                df, summary = analyze_conversation(st.session_state.conversation_df)
                st.session_state.analyzed_df = df
                st.session_state.summary = summary
        render_conflict_signals()
    elif page == "Speaker Analysis":
        if st.session_state.analyzed_df is None:
            with st.spinner("Analyzing conversation..."):
                df, summary = analyze_conversation(st.session_state.conversation_df)
                st.session_state.analyzed_df = df
                st.session_state.summary = summary
        render_speaker_analysis()
    elif page == "Evidence":
        if st.session_state.analyzed_df is None:
            with st.spinner("Analyzing conversation..."):
                df, summary = analyze_conversation(st.session_state.conversation_df)
                st.session_state.analyzed_df = df
                st.session_state.summary = summary
        render_evidence()
    elif page == "Methodology":
        render_methodology()

    # Sidebar actions
    st.sidebar.markdown("---")
    if st.session_state.conversation_df is not None:
        if st.sidebar.button("Upload New Conversation"):
            st.session_state.conversation_df = None
            st.session_state.analyzed_df = None
            st.session_state.summary = None
            st.rerun()


if __name__ == "__main__":
    main()
