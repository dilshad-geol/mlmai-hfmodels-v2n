import streamlit as st
from typing import Tuple, Dict, Any
from config.settings import UI_CONFIG

def render_header():
    """Render application header"""
    st.title("💊 Pharmacovigilance NLP Inference")
    st.markdown("**Extract drug-event relations, NERs, and relevance scores from biomedical text**")
    
    # Add info section
    with st.expander("ℹ️ About this system"):
        st.markdown("""
        This AI-powered system analyzes biomedical literature for pharmacovigilance insights:
        
        - **🧬 Entity Recognition**: Identifies drugs, diseases, and adverse effects
        - **🔗 Relation Extraction**: Finds relationships between drugs and events
        - **📊 Relevance Scoring**: Assesses pharmacovigilance importance (0-1 scale)
        - **🌐 Connected Terms**: Discovers semantic relationships
        
        **Models Used**: BioBERT, Clinical BERT, SciBERT, XLM-RoBERTa
        """)

def render_input_section() -> Tuple[str, str]:
    """Render input section and return title and abstract"""
    st.subheader("📝 Input")
    
    # Sample data button
    if st.button("📋 Load Sample Data"):
        st.session_state.sample_title = "Cardiovascular Safety of Atorvastatin in Elderly Patients: A Retrospective Analysis"
        st.session_state.sample_abstract = "This retrospective study analyzed 1,247 elderly patients (age >65) treated with atorvastatin 20-80mg daily for hypercholesterolemia. During 24-month follow-up, we observed 23 cases of muscle pain, 8 cases of rhabdomyolysis, and 12 cases of elevated liver enzymes. Cardiovascular events decreased by 35% compared to control group. However, 15 patients discontinued treatment due to statin-induced myopathy. Risk factors included age >75 years and concomitant use of fibrates. Results suggest careful monitoring is required for elderly patients receiving high-dose atorvastatin therapy."
    
    title = st.text_input(
        "Title", 
        placeholder="Enter research paper title...",
        value=st.session_state.get('sample_title', '')
    )
    
    abstract = st.text_area(
        "Abstract", 
        height=200, 
        placeholder="Enter abstract text...",
        value=st.session_state.get('sample_abstract', '')
    )
    
    # Clear sample data
    if st.button("🗑️ Clear"):
        st.session_state.pop('sample_title', None)
        st.session_state.pop('sample_abstract', None)
        st.rerun()
    
    return title, abstract

def render_results_overview(result: Dict[str, Any]):
    """Render results overview section"""
    st.subheader("📊 Results Overview")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧬 Drugs Found", len(result['drugs']))
    with col2:
        st.metric("🏥 Diseases Found", len(result['diseases']))
    with col3:
        st.metric("🔗 Relations Found", len(result['relations']))
    
    # Relevance score with color coding
    relevance_score = result['relevance_score']
    st.metric("📈 Relevance Score", f"{relevance_score:.3f}")
    
    # Color-coded assessment
    if relevance_score >= UI_CONFIG['relevance_thresholds']['high']:
        st.success("🟢 High pharmacovigilance relevance")
    elif relevance_score >= UI_CONFIG['relevance_thresholds']['medium']:
        st.warning("🟡 Medium pharmacovigilance relevance")
    else:
        st.error("🔴 Low pharmacovigilance relevance")
    
    # Quick insights
    if result['relations']:
        most_confident_relation = max(result['relations'], key=lambda x: x['confidence'])
        st.info(f"🎯 **Most Confident Relation**: {most_confident_relation['entity1']} → {most_confident_relation['entity2']} ({most_confident_relation['confidence']:.3f})")

def render_entity_tab(result: Dict[str, Any]):
    """Render entities tab"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💊 Drugs")
        if result['drugs']:
            for drug in result['drugs']:
                st.success(f"🔹 {drug}")
        else:
            st.info("No drugs found")
    
    with col2:
        st.subheader("🏥 Diseases/Effects")
        if result['diseases']:
            for disease in result['diseases']:
                st.warning(f"🔸 {disease}")
        else:
            st.info("No diseases found")

def render_relations_tab(result: Dict[str, Any]):
    """Render relations tab"""
    st.subheader("🔗 Drug-Event Relations")
    
    if result['relations']:
        # Sort by confidence
        sorted_relations = sorted(result['relations'], key=lambda x: x['confidence'], reverse=True)
        
        for rel in sorted_relations:
            relation_color = UI_CONFIG['relation_colors'].get(rel['relation_type'], '⚪')
            
            confidence_bar = "🟩" * int(rel['confidence'] * 10) + "⬜" * (10 - int(rel['confidence'] * 10))
            
            st.markdown(f"""
            **{relation_color} {rel['entity1']}** ➡️ **{rel['entity2']}**
            - *Relation*: `{rel['relation_type']}`
            - *Confidence*: `{rel['confidence']}` {confidence_bar}
            """)
            st.divider()
    else:
        st.info("No relations found")

def render_connected_terms_tab(result: Dict[str, Any]):
    """Render connected terms tab"""
    st.subheader("🌐 Connected Terms")
    
    if result['connected_terms']:
        for conn in result['connected_terms']:
            similarity_bar = "🟦" * int(conn['similarity'] * 10) + "⬜" * (10 - int(conn['similarity'] * 10))
            
            st.markdown(f"""
            **{conn['term1']}** ↔️ **{conn['term2']}**
            - *Similarity*: `{conn['similarity']}` {similarity_bar}
            """)
    else:
        st.info("No connected terms found")

def render_json_tab(result: Dict[str, Any]):
    """Render JSON output tab"""
    st.subheader("📋 JSON Output")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📋 Copy JSON", use_container_width=True):
            st.code(str(result), language='json')
    
    st.json(result)

def render_download_section(result: Dict[str, Any]):
    """Render download section"""
    import json
    
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        json_str = json.dumps(result, indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=json_str,
            file_name="pharmacovigilance_analysis.json",
            mime="application/json"
        )
    
    with col2:
        # Create CSV format
        csv_data = "Entity Type,Entity,Confidence\n"
        for drug in result['drugs']:
            csv_data += f"Drug,{drug},1.0\n"
        for disease in result['diseases']:
            csv_data += f"Disease,{disease},1.0\n"
        for rel in result['relations']:
            csv_data += f"Relation,{rel['entity1']} -> {rel['entity2']},{rel['confidence']}\n"
        
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name="pharmacovigilance_entities.csv",
            mime="text/csv"
        )
    
    with col3:
        # Create summary report
        report = f"""
# Pharmacovigilance Analysis Report

## Input
**Title**: {result['title']}
**Abstract**: {result['abstract'][:200]}...

## Summary
- **Relevance Score**: {result['relevance_score']}
- **Drugs Found**: {len(result['drugs'])}
- **Diseases Found**: {len(result['diseases'])}
- **Relations Found**: {len(result['relations'])}

## Key Findings
### Drugs
{chr(10).join(f"- {drug}" for drug in result['drugs'][:5])}

### Relations
{chr(10).join(f"- {rel['entity1']} → {rel['entity2']} ({rel['confidence']:.3f})" for rel in result['relations'][:5])}
        """
        
        st.download_button(
            label="📝 Download Report",
            data=report,
            file_name="pharmacovigilance_report.md",
            mime="text/markdown"
        )