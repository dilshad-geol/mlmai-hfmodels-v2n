import streamlit as st
import sys
import os
import random
import numpy as np
import torch


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.nlp_system import PharmNLPSystem
    from components.ui_components import render_header, render_input_section, render_results_overview
    from components.visualization import render_analysis_tabs
    from config.settings import APP_CONFIG
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure you're running from the project root directory")
    st.stop()


def _set_deterministic_seeds(seed: int = 42):
    """Best-effort determinism for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    st.set_page_config(
        page_title=APP_CONFIG['page_title'],
        page_icon=APP_CONFIG['page_icon'],
        layout=APP_CONFIG['layout'],
        initial_sidebar_state=APP_CONFIG['sidebar_state']
    )

    # Ensure deterministic behavior across runs
    _set_deterministic_seeds(42)

    render_header()

    try:
        if 'nlp_system' not in st.session_state:
            with st.spinner("🔄 Initializing AI models..."):
                st.session_state.nlp_system = PharmNLPSystem()
            st.success("System ready!")

        col1, col2 = st.columns([2, 1])

        with col1:
            title, abstract = render_input_section()

            if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
                if title and abstract:
                    with st.spinner("🧠 Processing with AI models..."):
                        try:
                            result = st.session_state.nlp_system.process_text(title, abstract)
                            st.session_state.result = result
                            st.success("Analysis complete!")
                        except Exception as e:
                            st.error(f"Processing error: {e}")
                            st.info("💡 Try again or check your input text")
                else:
                    st.error("Please enter both title and abstract")

        with col2:
            if 'result' in st.session_state:
                render_results_overview(st.session_state.result)

        if 'result' in st.session_state:
            try:
                render_analysis_tabs(st.session_state.result)
            except Exception as e:
                st.error(f"Visualization error: {e}")
                st.info("Results are still available in the JSON tab")

                # Show basic results as fallback
                result = st.session_state.result
                st.subheader("📊 Basic Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Drugs Found:**")
                    for drug in result.get('drugs', []):
                        st.write(f"• {drug}")
                with col2:
                    st.write("**Diseases Found:**")
                    for disease in result.get('diseases', []):
                        st.write(f"• {disease}")

                if result.get('relations'):
                    st.write("**Relations Found:**")
                    for rel in result['relations']:
                        st.write(f"• {rel['entity1']} → {rel['entity2']} ({rel['confidence']:.3f})")

    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page and try again")

        # Show debug information
        with st.expander("🔧 Debug Information"):
            st.code(str(e))


if __name__ == "__main__":
    main()
