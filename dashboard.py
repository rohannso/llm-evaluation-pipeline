"""
Streamlit Dashboard for LLM Evaluation Results
Beautiful UI to visualize evaluation metrics

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 LLM Evaluation Dashboard</h1>', unsafe_allow_html=True)

# Sidebar - File Upload
st.sidebar.header("📁 Load Evaluation Results")
uploaded_file = st.sidebar.file_uploader(
    "Upload JSON file", 
    type=['json'],
    help="Upload your evaluation_results.json file"
)

# Load default file if no upload
if uploaded_file is None:
    default_files = [
        'data/evaluation_results.json',
        'data/results/sample-chat-conversation-01_hybrid_evaluation.json',
        'evaluation_results.json'
    ]
    
    data = None
    for file_path in default_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            st.sidebar.success(f"✅ Loaded: {file_path}")
            break
    
    if data is None:
        st.warning("⚠️ No evaluation file found. Please upload a JSON file.")
        st.stop()
else:
    data = json.load(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

# Extract data
summary = data['evaluation_summary']
turns = data['turn_evaluations']

# === SUMMARY METRICS ===
st.header("📊 Overall Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Relevance",
        f"{summary['avg_relevance_score']:.2%}",
        delta=f"{(summary['avg_relevance_score'] - 0.7) * 100:.1f}% vs target (70%)",
        delta_color="normal"
    )

with col2:
    hallucination = summary['avg_hallucination_rate']
    st.metric(
        "Hallucination Rate",
        f"{hallucination:.2%}",
        delta=f"{-(hallucination - 0.2) * 100:.1f}% vs target (20%)",
        delta_color="inverse"
    )

with col3:
    st.metric(
        "Total Latency",
        f"{summary['total_latency_ms']:.0f} ms",
        delta=f"{summary['total_latency_ms'] / len(turns):.0f} ms/turn"
    )

with col4:
    st.metric(
        "Total Cost",
        f"${summary['total_cost_usd']:.4f}",
        delta=f"${summary['total_cost_usd'] / len(turns):.4f}/turn"
    )

# === EVALUATION MODE ===
st.divider()
mode = data.get('evaluation_mode', 'unknown')
mode_emoji = "🔄" if "hybrid" in mode else "🧠"
st.info(f"{mode_emoji} **Evaluation Mode:** {mode.replace('_', ' ').title()}")

# === TOKEN USAGE ===
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💰 Token Usage Breakdown")
    
    tokens = summary['token_usage']
    token_df = pd.DataFrame([
        {"Type": "Embedding Tokens", "Count": tokens['embedding_tokens'], "Cost": "$0 (Local)"},
        {"Type": "LLM Input Tokens", "Count": tokens['input_tokens'], "Cost": f"${tokens['input_tokens'] * 0.59 / 1_000_000:.4f}"},
        {"Type": "LLM Output Tokens", "Count": tokens['output_tokens'], "Cost": f"${tokens['output_tokens'] * 0.79 / 1_000_000:.4f}"}
    ])
    
    st.dataframe(token_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🤖 LLM Usage")
    
    if 'llm_usage' in summary:
        llm = summary['llm_usage']
        st.metric("Total LLM Calls", llm['total_llm_calls'])
        st.metric("Usage Percentage", f"{llm['percentage']:.1f}%")
        
        st.caption("Breakdown:")
        st.write(f"- Relevance: {llm['relevance_calls']}")
        st.write(f"- Claims: {llm['claim_extraction_calls']}")
        st.write(f"- Grounding: {llm['grounding_calls']}")

# === TURN-BY-TURN ANALYSIS ===
st.divider()
st.header("📝 Turn-by-Turn Analysis")

# Create DataFrame
turn_data = []
for turn in turns:
    turn_data.append({
        "Turn": turn['turn'],
        "Relevance": turn['relevance_score'],
        "Hallucination": turn['hallucination_score'],
        "Claims": f"{turn['grounded_claims']}/{turn['total_claims']}",
        "Latency (ms)": turn['latency_ms'],
        "Cost ($)": turn['cost_usd']
    })

df = pd.DataFrame(turn_data)

# Plot: Relevance & Hallucination over turns
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Turn'],
    y=df['Relevance'],
    mode='lines+markers',
    name='Relevance',
    line=dict(color='#667eea', width=3),
    marker=dict(size=10)
))

fig.add_trace(go.Scatter(
    x=df['Turn'],
    y=df['Hallucination'],
    mode='lines+markers',
    name='Hallucination',
    line=dict(color='#f093fb', width=3),
    marker=dict(size=10)
))

fig.update_layout(
    title="Relevance vs Hallucination Across Turns",
    xaxis_title="Turn Number",
    yaxis_title="Score",
    hovermode='x unified',
    template='plotly_white',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# === DETAILED TURN INSPECTION ===
st.divider()
st.header("🔍 Detailed Turn Inspection")

# Turn selector
selected_turn = st.selectbox(
    "Select Turn to Inspect",
    options=[t['turn'] for t in turns],
    format_func=lambda x: f"Turn {x}"
)

# Get selected turn data
turn_detail = next(t for t in turns if t['turn'] == selected_turn)

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("💬 Conversation")
    
    # User query
    st.markdown("**👤 User:**")
    st.info(turn_detail['user_query'])
    
    # AI response
    st.markdown("**🤖 AI Response:**")
    st.success(turn_detail['message'])

with col2:
    st.subheader("📊 Metrics")
    
    # Relevance
    rel_color = "🟢" if turn_detail['relevance_score'] >= 0.7 else "🟡" if turn_detail['relevance_score'] >= 0.3 else "🔴"
    st.write(f"{rel_color} **Relevance:** {turn_detail['relevance_score']:.2%}")
    st.write(f"**Label:** {turn_detail['relevance_label'].title()}")
    st.write(f"**Method:** {turn_detail.get('relevance_method', 'N/A')}")
    
    st.divider()
    
    # Hallucination
    hal_color = "🟢" if turn_detail['hallucination_score'] <= 0.2 else "🟡" if turn_detail['hallucination_score'] <= 0.5 else "🔴"
    st.write(f"{hal_color} **Hallucination:** {turn_detail['hallucination_score']:.2%}")
    st.write(f"**Grounded Claims:** {turn_detail['grounded_claims']}/{turn_detail['total_claims']}")
    
    st.divider()
    
    # Performance
    st.write(f"⚡ **Latency:** {turn_detail['latency_ms']:.0f} ms")
    st.write(f"💰 **Cost:** ${turn_detail['cost_usd']:.6f}")

# Hallucinated claims (if any)
if turn_detail['hallucinated_claims']:
    st.warning("⚠️ **Detected Hallucinations:**")
    for i, claim in enumerate(turn_detail['hallucinated_claims'], 1):
        st.markdown(f"{i}. {claim}")
else:
    st.success("✅ No hallucinations detected in this turn!")

# === STATISTICS ===
st.divider()
st.header("📈 Statistics & Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Relevance Distribution")
    relevance_scores = [t['relevance_score'] for t in turns]
    
    fig = go.Figure(data=[go.Histogram(
        x=relevance_scores,
        nbinsx=10,
        marker_color='#667eea'
    )])
    fig.update_layout(
        xaxis_title="Relevance Score",
        yaxis_title="Count",
        showlegend=False,
        height=250
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Hallucination Distribution")
    hal_scores = [t['hallucination_score'] for t in turns]
    
    fig = go.Figure(data=[go.Histogram(
        x=hal_scores,
        nbinsx=10,
        marker_color='#f093fb'
    )])
    fig.update_layout(
        xaxis_title="Hallucination Rate",
        yaxis_title="Count",
        showlegend=False,
        height=250
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Performance Stats")
    
    avg_latency = sum(t['latency_ms'] for t in turns) / len(turns)
    avg_cost = sum(t['cost_usd'] for t in turns) / len(turns)
    
    st.metric("Avg Latency/Turn", f"{avg_latency:.0f} ms")
    st.metric("Avg Cost/Turn", f"${avg_cost:.6f}")
    
    total_claims = sum(t['total_claims'] for t in turns)
    total_grounded = sum(t['grounded_claims'] for t in turns)
    st.metric("Claims Grounded", f"{total_grounded}/{total_claims}")

# === GROUNDING ANALYSIS ===
st.divider()
st.header("🎯 Claim Grounding Analysis")

# Create grounding data
grounding_data = []
for turn in turns:
    if turn['total_claims'] > 0:
        grounding_data.append({
            "Turn": turn['turn'],
            "Total Claims": turn['total_claims'],
            "Grounded": turn['grounded_claims'],
            "Hallucinated": turn['total_claims'] - turn['grounded_claims'],
            "Grounding Rate": turn['grounded_claims'] / turn['total_claims']
        })

if grounding_data:
    grounding_df = pd.DataFrame(grounding_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=grounding_df['Turn'],
        y=grounding_df['Grounded'],
        name='Grounded',
        marker_color='#10b981'
    ))
    
    fig.add_trace(go.Bar(
        x=grounding_df['Turn'],
        y=grounding_df['Hallucinated'],
        name='Hallucinated',
        marker_color='#ef4444'
    ))
    
    fig.update_layout(
        title="Claims Grounding by Turn",
        xaxis_title="Turn",
        yaxis_title="Number of Claims",
        barmode='stack',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# === EXPORT ===
st.divider()
st.header("💾 Export Data")

col1, col2 = st.columns(2)

with col1:
    # Download full JSON
    st.download_button(
        label="📥 Download Full JSON",
        data=json.dumps(data, indent=2),
        file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

with col2:
    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Summary CSV",
        data=csv,
        file_name=f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.divider()
st.caption(f"Evaluated at: {data.get('evaluated_at', 'N/A')} | Conversation ID: {data.get('conversation_id', 'N/A')}")