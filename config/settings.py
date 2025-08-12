import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")

APP_CONFIG = {
    'page_title': "Pharmacovigilance NLP",
    'page_icon': "💊",
    'layout': "wide",
    'sidebar_state': "collapsed"
}

# Toggle behavior (used by your ModelLoader)
INFERENCE_CONFIG = {
    # Try local first, then fallback to HF if local load hits .bin on torch<2.6
    'prefer_local': True,
    'use_hf_inference_for_bin': True,
    # Optional: read token from env if you have rate limits or private models
    'hf_api_key': os.getenv("HUGGINGFACEHUB_API_TOKEN", None),
}

MODEL_CONFIG = {
    'biomedical_ner': {
        'model_name': os.path.join(MODELS_DIR, "biomedical_ner"),
        'hf_model_id': "d4data/biomedical-ner-all",            # token-classification
        'max_length': 512
    },
    'clinical_bert': {
        'model_name': os.path.join(MODELS_DIR, "clinical_bert"),
        'hf_model_id': "emilyalsentzer/Bio_ClinicalBERT",      # feature-extraction (embeddings)
        'max_length': 512
    },
    'scibert_ade': {
        'model_name': os.path.join(MODELS_DIR, "scibert_ade"),
        'hf_model_id': "jsylee/scibert_scivocab_uncased-finetuned-ner",  # token-classification
        'num_labels': 5,
        'id2label': {0: 'O', 1: 'B-DRUG', 2: 'I-DRUG', 3: 'B-EFFECT', 4: 'I-EFFECT'},
        'max_length': 512
    },
    'relation_extraction': {
        'model_name': os.path.join(MODELS_DIR, "relation_extraction"),
        'hf_model_id': "sagteam/pharm-relation-extraction",    # text-classification
        # Raise threshold slightly to prioritize precision
        'confidence_threshold': 0.7
    }
}

PROCESSING_CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'max_entities': 20,
    'max_relations': 15,
    'max_connected_terms': 10,
    'max_entity_pairs': 15
}

RELEVANCE_CONFIG = {
    'high_value_keywords': [
        'adverse', 'side effect', 'toxicity', 'safety',
        'pharmacovigilance', 'drug', 'reaction', 'adr', 'ade'
    ],
    'medium_value_keywords': [
        'clinical', 'patient', 'treatment', 'therapy',
        'monitoring', 'surveillance'
    ],
    'pharma_keywords': [
        'adverse', 'side effect', 'toxicity', 'safety', 'drug', 'medication',
        'treatment', 'therapy', 'dosage', 'administration', 'pharmacokinetics',
        'pharmacodynamics', 'clinical trial', 'efficacy', 'contraindication',
        'interaction', 'reaction', 'symptom', 'syndrome', 'disorder',
        'rhabdomyolysis', 'myopathy', 'hepatotoxicity', 'cardiovascular',
        'statin', 'monitoring', 'patient', 'retrospective', 'follow-up',
        'pharmacovigilance', 'surveillance', 'adr', 'ade'
    ],
    'scoring_weights': {
        'keyword_score': 0.35,
        'clinical_score': 0.30,
        'length_penalty': 0.15,
        'entity_density': 0.20
    }
}

UI_CONFIG = {
    'relation_colors': {
        'adverse_drug_reaction': '🔴',
        'drug_causes_effect': '🟠',
        'side_effect_relation': '🟡',
        'drug_effect_relation': '🟢',
        'drug_indication': '🔵',
        'same_entity': '⚪',
        'drug_disease_relation': '🟣',
        'contraindication': '⚫'
    },
    'relevance_thresholds': {
        'high': 0.7,
        'medium': 0.4
    }
}
