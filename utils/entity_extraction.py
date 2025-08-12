import re
import torch
from typing import Dict, List, Any, Iterable
from collections import defaultdict, OrderedDict

from transformers import pipeline
from config.settings import MODEL_CONFIG, PROCESSING_CONFIG


def _unique_preserve_casefold(items: Iterable[str]) -> List[str]:
    """Deduplicate while preserving first occurrence order (case-insensitive)."""
    seen = set()
    out = []
    for x in items:
        if not x:
            continue
        k = x.casefold().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out


def _clean_span(text: str) -> str:
    """Light cleanup for extracted spans."""
    if not text:
        return ""
    t = text.strip()
    # Normalize spaces around hyphens/slashes
    t = re.sub(r'\s*-\s*', '-', t)
    t = re.sub(r'\s*/\s*', '/', t)
    # Trim surrounding punctuation
    t = t.strip(" ,.;:()[]{}")
    # Collapse internal whitespace
    t = re.sub(r'\s+', ' ', t)
    return t


class EntityExtractor:
    """
    Robust entity extraction:
    - Uses HF pipeline with aggregation to correctly merge wordpieces.
    - Slides over long inputs with token windows to avoid truncation.
    - Maps model labels into the project's expected categories.
    - Falls back to regex rules when models aren't available.
    """
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.device = PROCESSING_CONFIG['device']

    # ---------- Public API ----------

    def extract_biomedical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities with d4data/biomedical-ner-all (if available),
        mapped to keys: CHEMICAL, DRUG, DISEASE, DISORDER, SYMPTOM.
        """
        if not self.models.get('biomedical_ner_model') or not self.models.get('biomedical_ner_tokenizer'):
            return self._extract_entities_fallback(text, "biomedical")

        try:
            ner = self._make_token_pipeline(
                self.models['biomedical_ner_model'],
                self.models['biomedical_ner_tokenizer']
            )
            chunks = self._iter_text_chunks_by_tokens(
                text=text,
                tokenizer=self.models['biomedical_ner_tokenizer'],
                max_length=MODEL_CONFIG['biomedical_ner']['max_length'],
                stride=64
            )

            buckets: Dict[str, List[str]] = defaultdict(list)
            for chunk in chunks:
                preds = ner(chunk)
                for p in preds:
                    span = _clean_span(p.get('word') or p.get('entity') or '')
                    if not span:
                        continue
                    mapped = self._map_biomedical_label(p.get('entity_group') or p.get('entity') or '')
                    if mapped:
                        buckets[mapped].append(span)

            # Dedup in each bucket, stable
            for k in list(buckets.keys()):
                buckets[k] = _unique_preserve_casefold(buckets[k])

            # Ensure keys exist even if empty (downstream expects them)
            for needed in ['CHEMICAL', 'DRUG', 'DISEASE', 'DISORDER', 'SYMPTOM']:
                buckets.setdefault(needed, [])

            return dict(buckets)

        except Exception as e:
            print(f"Error in biomedical NER: {e}")
            return self._extract_entities_fallback(text, "biomedical")

    def extract_drug_effects(self, text: str) -> Dict[str, List[str]]:
        """
        Extract DRUG / EFFECT spans with SciBERT ADE if available.
        Falls back to rules otherwise.
        """
        if not self.models.get('ade_ner_model') or not self.models.get('ade_ner_tokenizer'):
            return self._extract_entities_fallback(text, "drug_effects")

        try:
            ade = self._make_token_pipeline(
                self.models['ade_ner_model'],
                self.models['ade_ner_tokenizer']
            )
            chunks = self._iter_text_chunks_by_tokens(
                text=text,
                tokenizer=self.models['ade_ner_tokenizer'],
                max_length=MODEL_CONFIG['scibert_ade']['max_length'],
                stride=64
            )

            drugs: List[str] = []
            effects: List[str] = []

            for chunk in chunks:
                preds = ade(chunk)
                for p in preds:
                    span = _clean_span(p.get('word') or '')
                    if not span:
                        continue
                    group = (p.get('entity_group') or p.get('entity') or '').upper()
                    if 'DRUG' in group:
                        drugs.append(span)
                    elif 'EFFECT' in group:
                        effects.append(span)

            return {
                'drugs': _unique_preserve_casefold(drugs),
                'effects': _unique_preserve_casefold(effects)
            }

        except Exception as e:
            print(f"Error in drug effects NER: {e}")
            return self._extract_entities_fallback(text, "drug_effects")

    # ---------- Pipelines & chunking ----------

    def _make_token_pipeline(self, model: Any, tokenizer: Any):
        """Create a token-classification pipeline with wordpiece aggregation."""
        device_idx = 0 if self.device.type == "cuda" else -1
        return pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            device=device_idx,
            aggregation_strategy="simple",   # merges B-/I- spans into clean text
            ignore_labels=["O"]
        )

    def _iter_text_chunks_by_tokens(self, text: str, tokenizer: Any, max_length: int, stride: int = 64) -> Iterable[str]:
        """
        Slide a token window over the input to avoid truncation.
        We decode windows back to text for the pipeline.
        """
        # Tokenize without specials to control window size
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )
        ids = enc["input_ids"]
        n = len(ids)
        if n == 0:
            return [text]

        window = max_length - 2  # leave room for specials added by pipeline/tokenizer internally
        if window <= 0 or n <= window:
            return [text]

        chunks: List[str] = []
        start = 0
        step = max(1, window - stride)
        while start < n:
            end = min(n, start + window)
            piece_ids = ids[start:end]
            chunk_text = tokenizer.decode(piece_ids, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text)
            if end == n:
                break
            start += step

        # Deduplicate identical decodes while preserving order
        seen = set()
        uniq = []
        for c in chunks:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    # ---------- Label mapping ----------

    def _map_biomedical_label(self, raw_label: str) -> str:
        """
        Map the heterogeneous labels from d4data/biomedical-ner-all
        into our 5 buckets used downstream.
        """
        l = raw_label.upper().strip()

        # Direct hits first
        if l in {"CHEMICAL", "DRUG", "DISEASE", "DISORDER", "SYMPTOM"}:
            return l

        # Common alternates seen across biomedical NER sets
        if any(k in l for k in ["CHEM", "COMPOUND", "SUBSTANCE"]):
            return "CHEMICAL"
        if any(k in l for k in ["MEDICATION", "PHARMACEUTICAL", "THERAPEUTIC"]):
            return "DRUG"

        # Disease-like
        if any(k in l for k in ["DISEASE", "DIAGNOSIS", "PATHOLOGY", "CONDITION"]):
            return "DISEASE"
        if "DISORDER" in l:
            return "DISORDER"

        # Symptom / ADE-like
        if any(k in l for k in ["SYMPTOM", "SIGN", "FINDING", "ADVERSE", "SIDE_EFFECT", "AE", "EFFECT"]):
            return "SYMPTOM"

        # Otherwise ignore (anatomy, organism, procedure, etc.)
        return ""

    # ---------- Heuristic fallbacks ----------

    def _extract_entities_fallback(self, text: str, mode: str) -> Dict[str, List[str]]:
        if mode == "biomedical":
            return self._extract_biomedical_fallback(text)
        elif mode == "drug_effects":
            return self._extract_drug_effects_fallback(text)
        return {}

    def _extract_biomedical_fallback(self, text: str) -> Dict[str, List[str]]:
        """Simple regex-based fallback, kept conservative."""
        drug_patterns = [
            r'\b(?:atorvastatin|simvastatin|pravastatin|rosuvastatin|lovastatin|fluvastatin)\b',
            r'\b(?:aspirin|ibuprofen|acetaminophen|morphine|insulin|penicillin|warfarin|metformin|omeprazole)\b',
            r'\b(?:statin|fibrate|beta-blocker|ace inhibitor|calcium channel blocker|antibiotic|analgesic)\b',
            r'\b\w+(?:cillin|mycin|prazole|statin|dipine|pril|sartan|olol|mab|zole)\b'
        ]

        disease_patterns = [
            r'\b(?:hypercholesterolemia|hypertension|diabetes|cardiovascular|myopathy|rhabdomyolysis|hepatotoxicity)\b',
            r'\b(?:muscle pain|liver enzymes|elevated liver|myalgia|arthralgia|neuropathy)\b',
            r'\b(?:cancer|pneumonia|arthritis|asthma|depression|anxiety|infection|syndrome|disorder)\b',
            r'\b\w+(?:itis|osis|emia|pathy|algia|syndrome|disorder|toxicity|carcinoma|sarcoma)\b'
        ]

        symptom_patterns = [
            r'\b(?:pain|ache|fatigue|nausea|headache|dizziness|rash|vomiting|diarrhea|constipation)\b',
            r'\b(?:muscle pain|joint pain|abdominal pain|chest pain|weakness|confusion)\b'
        ]

        drugs, diseases, symptoms = set(), set(), set()
        text_lower = text.lower()

        for pattern in drug_patterns:
            drugs.update(re.findall(pattern, text_lower))
        for pattern in disease_patterns:
            diseases.update(re.findall(pattern, text_lower))
        for pattern in symptom_patterns:
            symptoms.update(re.findall(pattern, text_lower))

        return {
            'CHEMICAL': _unique_preserve_casefold(drugs),
            'DRUG': _unique_preserve_casefold(drugs),
            'DISEASE': _unique_preserve_casefold(diseases),
            'DISORDER': _unique_preserve_casefold(diseases),
            'SYMPTOM': _unique_preserve_casefold(symptoms)
        }

    def _extract_drug_effects_fallback(self, text: str) -> Dict[str, List[str]]:
        effect_patterns = [
            r'\b(?:rhabdomyolysis|myopathy|hepatotoxicity|muscle pain|myalgia)\b',
            r'\b(?:elevated liver enzymes|liver damage|muscle weakness|joint pain)\b',
            r'\b(?:nausea|headache|dizziness|fatigue|rash|vomiting|diarrhea|constipation|drowsiness)\b',
            r'\b(?:side effect|adverse|reaction|toxicity|allergy|complication)\b'
        ]
        drug_patterns = [
            r'\b(?:atorvastatin|statin|medication|drug|therapy|treatment)\b',
            r'\b(?:aspirin|ibuprofen|warfarin|metformin|insulin)\b'
        ]
        

        effects, drugs = set(), set()
        text_lower = text.lower()
        for pattern in effect_patterns:
            effects.update(re.findall(pattern, text_lower))
        for pattern in drug_patterns:
            drugs.update(re.findall(pattern, text_lower))

        return {
            'drugs': _unique_preserve_casefold(drugs),
            'effects': _unique_preserve_casefold(effects)
        }
