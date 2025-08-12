import streamlit as st
from typing import Dict, Any
from components.ui_components import (
    render_entity_tab, render_relations_tab, 
    render_connected_terms_tab, render_json_tab, render_download_section
)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def render_analysis_tabs(result: Dict[str, Any]):
    """Render main analysis tabs"""
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧬 Entities", "🔗 Relations", "🌐 Connected Terms", 
        "📊 Visualization", "📋 JSON Output", "💾 Export"
    ])
    
    with tab1:
        render_entity_tab(result)
    
    with tab2:
        render_relations_tab(result)
    
    with tab3:
        render_connected_terms_tab(result)
    
    with tab4:
        render_visualization_tab(result)
    
    with tab5:
        render_json_tab(result)
    
    with tab6:
        render_download_section(result)

def render_visualization_tab(result: Dict[str, Any]):
    """Render visualization tab"""
    st.subheader("📊 Analysis Visualization")
    
    if PLOTLY_AVAILABLE:
        render_plotly_charts(result)
    else:
        render_basic_charts(result)
    
    render_summary_metrics(result)

def render_plotly_charts(result: Dict[str, Any]):
    """Render charts using Plotly"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Relation types distribution
        if result['relations']:
            relation_types = {}
            for rel in result['relations']:
                rel_type = rel['relation_type']
                relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
            
            fig_pie = px.pie(
                values=list(relation_types.values()),
                names=list(relation_types.keys()),
                title="🔗 Relation Types Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No relations to visualize")
    
    with col2:
        # Entity counts
        entity_counts = {
            'Drugs': len(result['drugs']),
            'Diseases': len(result['diseases']),
            'Relations': len(result['relations']),
            'Connected Terms': len(result['connected_terms'])
        }
        
        fig_bar = px.bar(
            x=list(entity_counts.keys()),
            y=list(entity_counts.values()),
            title="📈 Entity Extraction Summary",
            color=list(entity_counts.values()),
            color_continuous_scale="viridis"
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Confidence distribution
    if result['relations']:
        confidences = [rel['confidence'] for rel in result['relations']]
        
        fig_hist = px.histogram(
            x=confidences,
            nbins=10,
            title="📊 Relation Confidence Distribution",
            labels={'x': 'Confidence Score', 'y': 'Number of Relations'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Network visualization (if we have relations)
    if result['relations']:
        render_network_graph(result)

def render_network_graph(result: Dict[str, Any]):
    """Render network graph of entities and relations"""
    if not PLOTLY_AVAILABLE:
        return
    
    # Create nodes and edges
    nodes = set()
    edges = []
    
    for rel in result['relations']:
        nodes.add(rel['entity1'])
        nodes.add(rel['entity2'])
        edges.append((rel['entity1'], rel['entity2'], rel['confidence'], rel['relation_type']))
    
    if not nodes:
        return
    
    # Simple network layout
    import math
    node_list = list(nodes)
    n = len(node_list)
    
    # Circular layout
    node_positions = {}
    for i, node in enumerate(node_list):
        angle = 2 * math.pi * i / n
        node_positions[node] = (math.cos(angle), math.sin(angle))
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in edges:
        x0, y0 = node_positions[edge[0]]
        x1, y1 = node_positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_info.append(f"{edge[0]} → {edge[1]}<br>Confidence: {edge[2]:.3f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in node_list:
        x, y = node_positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        # Color nodes by type
        if node in result['drugs']:
            node_colors.append('lightblue')
        else:
            node_colors.append('lightcoral')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=30,
            color=node_colors,
            line=dict(width=2, color='black')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='🌐 Entity Relationship Network',
                        title_font_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Blue = Drugs, Red = Diseases/Effects",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor="left", yanchor="bottom",
                            font=dict(size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    st.plotly_chart(fig, use_container_width=True)

def render_basic_charts(result: Dict[str, Any]):
    """Render basic charts using Streamlit native charts"""
    st.info("📦 Install plotly for enhanced visualizations: `pip install plotly`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Entity Counts")
        chart_data = {
            'Drugs': len(result['drugs']),
            'Diseases': len(result['diseases']),
            'Relations': len(result['relations'])
        }
        st.bar_chart(chart_data)
    
    with col2:
        st.subheader("📈 Relevance Score")
        relevance_data = {'Relevance Score': [result['relevance_score']]}
        st.line_chart(relevance_data)

def render_summary_metrics(result: Dict[str, Any]):
    """Render summary metrics"""
    st.subheader("📋 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_confidence = 0
        if result['relations']:
            avg_confidence = sum(rel['confidence'] for rel in result['relations']) / len(result['relations'])
        st.metric("🎯 Avg Confidence", f"{avg_confidence:.3f}")
    
    with col2:
        unique_drugs = len(set(result['drugs']))
        st.metric("💊 Unique Drugs", unique_drugs)
    
    with col3:
        unique_diseases = len(set(result['diseases']))
        st.metric("🏥 Unique Diseases", unique_diseases)
    
    with col4:
        total_entities = len(result['drugs']) + len(result['diseases'])
        st.metric("🧬 Total Entities", total_entities)
    
    # Additional insights
    if result['relations']:
        st.subheader("🔍 Key Insights")
        
        # Most common relation type
        relation_counts = {}
        for rel in result['relations']:
            rel_type = rel['relation_type']
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
        
        if relation_counts:
            most_common = max(relation_counts.items(), key=lambda x: x[1])
            st.success(f"🏆 **Most Common Relation**: {most_common[0]} ({most_common[1]} occurrences)")
        
        # Highest confidence relation
        highest_conf_rel = max(result['relations'], key=lambda x: x['confidence'])
        st.info(f"⭐ **Highest Confidence**: {highest_conf_rel['entity1']} → {highest_conf_rel['entity2']} ({highest_conf_rel['confidence']:.3f})")