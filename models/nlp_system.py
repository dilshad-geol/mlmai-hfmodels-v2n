import re
from typing import Dict, List, Any
from models.model_loader import ModelLoader
from utils.entity_extraction import EntityExtractor
from utils.relation_extraction import RelationExtractor
from utils.relevance_scoring import RelevanceScorer
from utils.text_processing import TextProcessor
from config.settings import PROCESSING_CONFIG

class PharmNLPSystem:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.models = self.model_loader.get_models()

        self.entity_extractor = EntityExtractor(self.models)
        self.relation_extractor = RelationExtractor(self.models)
        self.relevance_scorer = RelevanceScorer(self.models)
        self.text_processor = TextProcessor()

    def process_text(self, title: str, abstract: str) -> Dict[str, Any]:
        """Main processing pipeline for pharmacovigilance analysis"""
        full_text = f"{title}. {abstract}"

        # Extract entities
        biomedical_entities = self.entity_extractor.extract_biomedical_entities(full_text)
        drug_effect_entities = self.entity_extractor.extract_drug_effects(full_text)

        # Combine and process entities (order-preserving, case-insensitive unique)
        drugs, diseases = self._combine_entities(biomedical_entities, drug_effect_entities)

        # Extract relations (rank pairs deterministically by proximity/same-sentence)
        entity_pairs = self._create_entity_pairs(full_text, drugs, diseases)
        relations = self.relation_extractor.extract_relations(full_text, entity_pairs)

        # Find connected terms
        connected_terms = self.text_processor.find_connected_terms(drugs, diseases)

        # Calculate relevance score
        relevance_score = self.relevance_scorer.calculate_score(title, abstract)

        return {
            "title": title,
            "abstract": abstract,
            "drugs": drugs[:PROCESSING_CONFIG['max_entities']],
            "diseases": diseases[:PROCESSING_CONFIG['max_entities']],
            "relations": relations[:PROCESSING_CONFIG['max_relations']],
            "connected_terms": connected_terms[:PROCESSING_CONFIG['max_connected_terms']],
            "relevance_score": round(relevance_score, 3)
        }

    def _unique_preserve_casefold(self, items: List[str]) -> List[str]:
        """Deduplicate while preserving first occurrence order (case-insensitive)."""
        seen = set()
        out = []
        for x in items:
            if not x:
                continue
            k = x.casefold()
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    def _combine_entities(self, biomedical_entities: Dict, drug_effect_entities: Dict) -> tuple:
        """Combine and deduplicate entities from different extractors (stable order)."""
        drugs_raw = (
            biomedical_entities.get('CHEMICAL', []) +
            biomedical_entities.get('DRUG', []) +
            drug_effect_entities.get('drugs', [])
        )

        diseases_raw = (
            biomedical_entities.get('DISEASE', []) +
            biomedical_entities.get('DISORDER', []) +
            biomedical_entities.get('SYMPTOM', []) +
            drug_effect_entities.get('effects', [])
        )

        drugs = self._unique_preserve_casefold(drugs_raw)
        diseases = self._unique_preserve_casefold(diseases_raw)
        return drugs, diseases

    def _create_entity_pairs(self, text: str, drugs: List[str], diseases: List[str]) -> List[tuple]:
        """Create entity pairs ranked by same-sentence occurrence and proximity (deterministic)."""
        text_l = text.lower()
        sentences = re.split(r'(?<=[.!?])\s+', text_l)

        def first_pos(term: str) -> int:
            return text_l.find(term.lower())

        scored = []
        for d in drugs:
            for dis in diseases:
                dpos = first_pos(d)
                spos = first_pos(dis)
                co_sentence = any(d.lower() in s and dis.lower() in s for s in sentences)
                distance = abs(dpos - spos) if dpos != -1 and spos != -1 else 1_000_000
                score = (1_000_000 if co_sentence else 0) - distance
                scored.append((d, dis, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return [(d, dis) for d, dis, _ in scored][:PROCESSING_CONFIG['max_entity_pairs']]
