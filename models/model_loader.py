import streamlit as st
import torch
import warnings
from typing import Dict, Any
import os

warnings.filterwarnings("ignore")

try:
    
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        AutoModel,
        AutoModelForSequenceClassification,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

from config.settings import MODEL_CONFIG, PROCESSING_CONFIG

class ModelLoader:
    def __init__(self):
        self.device = PROCESSING_CONFIG['device']
        self.models = {}

    @st.cache_resource
    def load_all_models(_self) -> Dict[str, Any]:
        if not TRANSFORMERS_AVAILABLE:
            st.warning("AI models not available. Using fallback methods.")
            return {}

        models = {}

        models.update(_self._load_biomedical_ner())
        models.update(_self._load_clinical_bert())
        models.update(_self._load_scibert_ade())
        models.update(_self._load_relation_extraction())

        return models

    def _load_biomedical_ner(self) -> Dict[str, Any]:
        try:
            config = MODEL_CONFIG['biomedical_ner']
            if not os.path.exists(config['model_name']):
                st.warning(f"Biomedical NER model not found at {config['model_name']}")
                return {'biomedical_ner_tokenizer': None, 'biomedical_ner_model': None}

            tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
            model = AutoModelForTokenClassification.from_pretrained(
                config['model_name']
            ).to(self.device)
            model.eval()

            st.success("Biomedical NER model loaded")
            return {
                'biomedical_ner_tokenizer': tokenizer,
                'biomedical_ner_model': model
            }
        except Exception as e:
            st.warning(f"Biomedical NER model failed: {e}")
            return {
                'biomedical_ner_tokenizer': None,
                'biomedical_ner_model': None
            }

    def _load_clinical_bert(self) -> Dict[str, Any]:
        try:
            config = MODEL_CONFIG['clinical_bert']
            if not os.path.exists(config['model_name']):
                st.warning(f"Clinical BERT model not found at {config['model_name']}")
                return {'clinical_tokenizer': None, 'clinical_model': None}

            tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
            model = AutoModel.from_pretrained(config['model_name']).to(self.device)
            model.eval()

            st.success("Clinical BERT model loaded")
            return {
                'clinical_tokenizer': tokenizer,
                'clinical_model': model
            }
        except Exception as e:
            st.warning(f"Clinical BERT model failed: {e}")
            return {
                'clinical_tokenizer': None,
                'clinical_model': None
            }

    def _load_scibert_ade(self) -> Dict[str, Any]:
        try:
            config = MODEL_CONFIG['scibert_ade']
            if not os.path.exists(config['model_name']):
                st.warning(f"SciBERT ADE model not found at {config['model_name']}")
                return {'ade_ner_tokenizer': None, 'ade_ner_model': None}

            tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
            model = AutoModelForTokenClassification.from_pretrained(
                config['model_name'],
                num_labels=config['num_labels'],
                id2label=config['id2label']
            ).to(self.device)
            model.eval()

            st.success("SciBERT ADE model loaded")
            return {
                'ade_ner_tokenizer': tokenizer,
                'ade_ner_model': model
            }
        except Exception as e:
            st.warning(f"SciBERT ADE model failed: {e}")
            return {
                'ade_ner_tokenizer': None,
                'ade_ner_model': None
            }

    def _load_relation_extraction(self) -> Dict[str, Any]:
        try:
            config = MODEL_CONFIG['relation_extraction']
            if not os.path.exists(config['model_name']):
                st.warning(f"Relation extraction model not found at {config['model_name']}")
                return {'relation_tokenizer': None, 'relation_model': None}

            tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
            model = AutoModelForSequenceClassification.from_pretrained(config['model_name']).to(self.device)
            model.eval()

            st.success("Relation extraction model loaded")
            return {
                'relation_tokenizer': tokenizer,
                'relation_model': model
            }
        except Exception as e:
            st.warning(f"Relation extraction model failed: {e}")
            return {
                'relation_tokenizer': None,
                'relation_model': None
            }

    def get_models(self) -> Dict[str, Any]:
        if not self.models:
            self.models = self.load_all_models()
        return self.models
