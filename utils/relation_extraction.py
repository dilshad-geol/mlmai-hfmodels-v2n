import torch
import torch.nn.functional as F
import re
from typing import Dict, List, Any, Tuple
from config.settings import MODEL_CONFIG

class RelationExtractor:
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        # Use config threshold (fallback to 0.6)
        self.confidence_threshold = MODEL_CONFIG.get('relation_extraction', {}).get('confidence_threshold', 0.6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Label-specific thresholds (tunable). Stricter for causal/ADR, a bit lower for generic relations.
        self.label_thresholds = {
            'same_entity': 0.75,
            'drug_disease_relation': 0.65,
            'drug_effect_relation': 0.60,
            'adverse_drug_reaction': 0.70,
            'drug_indication': 0.70,
            'drug_causes_effect': 0.70,
            'side_effect_relation': 0.65,
            'contraindication': 0.70
        }

    # ---------- Public API ----------

    def extract_relations(self, text: str, entity_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        High-precision extractor:
        - Filters pairs to those that co-occur within a sentence (±1 sentence window optional).
        - Applies rule-based and model classifiers on local sentence context, not the whole abstract.
        """
        sentences = self._split_sentences(text)
        filtered_pairs = self._filter_pairs_by_sentence_cooccurrence(sentences, entity_pairs)

        relations: List[Dict[str, Any]] = []

        # Rule-based on local windows
        rule_based_relations = self._extract_relations_rule_based(sentences, filtered_pairs)
        relations.extend(rule_based_relations)

        # Model-based on local windows
        if self.models.get('relation_tokenizer') and self.models.get('relation_model'):
            model_relations = self._extract_relations_model(sentences, filtered_pairs)
            for model_rel in model_relations:
                exists = any(
                    r['entity1'].lower() == model_rel['entity1'].lower()
                    and r['entity2'].lower() == model_rel['entity2'].lower()
                    for r in relations
                )
                if not exists:
                    relations.append(model_rel)

        relations = self._deduplicate_relations(relations)
        relations = sorted(relations, key=lambda x: x['confidence'], reverse=True)
        return relations[:15]

    # ---------- Sentence handling ----------

    def _split_sentences(self, text: str) -> List[str]:
        """
        Lightweight sentence splitter without external deps.
        Keeps punctuation attached; trims spaces.
        """
        # Protect common abbreviations a bit (very light heuristic)
        safe = re.sub(r'\b(e\.g|i\.e|vs|fig|dr|mr|mrs)\.\s', lambda m: m.group(0).replace('.', '<DOT>'), text, flags=re.I)
        raw = re.split(r'(?<=[.!?])\s+', safe)
        sentences = [s.replace('<DOT>', '.').strip() for s in raw if s.strip()]
        return sentences

    def _contains_mention(self, sentence: str, mention: str) -> bool:
        s = sentence.lower()
        m = mention.lower().strip()
        if not m:
            return False
        # Prefer word boundaries; fall back to substring if boundaries fail (handles hyphens etc.)
        pattern = r'\b' + re.escape(m) + r'\b'
        return bool(re.search(pattern, s)) or (m in s)

    def _filter_pairs_by_sentence_cooccurrence(
        self, sentences: List[str], entity_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """
        Keep only pairs that co-occur in at least one sentence (or within a ±1 sentence window).
        This slashes the noisy Cartesian product.
        """
        kept: List[Tuple[str, str]] = []
        for drug, disease in entity_pairs:
            found = False
            for i, sent in enumerate(sentences):
                in_curr = self._contains_mention(sent, drug) and self._contains_mention(sent, disease)
                if in_curr:
                    found = True
                else:
                    # optionally allow adjacent sentence window if entities are split across a boundary
                    if i + 1 < len(sentences):
                        near_next = (self._contains_mention(sent, drug) and self._contains_mention(sentences[i + 1], disease)) or \
                                    (self._contains_mention(sent, disease) and self._contains_mention(sentences[i + 1], drug))
                        if near_next:
                            found = True
                if found:
                    kept.append((drug, disease))
                    break
        return kept

    def _contexts_for_pair(self, sentences: List[str], e1: str, e2: str) -> List[str]:
        """
        Build small context windows for (e1, e2): sentence with both, or sentence ±1 neighbor.
        """
        contexts: List[str] = []
        for i, sent in enumerate(sentences):
            if self._contains_mention(sent, e1) and self._contains_mention(sent, e2):
                ctx_parts = [sentences[i - 1]] if i - 1 >= 0 else []
                ctx_parts += [sent]
                if i + 1 < len(sentences):
                    ctx_parts += [sentences[i + 1]]
                ctx = " ".join(ctx_parts).strip()
                contexts.append(ctx)
            else:
                # split across boundary
                if i + 1 < len(sentences):
                    a, b = sentences[i], sentences[i + 1]
                    if (self._contains_mention(a, e1) and self._contains_mention(b, e2)) or \
                       (self._contains_mention(a, e2) and self._contains_mention(b, e1)):
                        ctx = (a + " " + b).strip()
                        contexts.append(ctx)
        # Dedup contexts while preserving order
        seen = set()
        uniq = []
        for c in contexts:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        return uniq[:4]  # cap to keep inference cheap

    # ---------- Model-based extraction (now sentence-scoped) ----------

    def _extract_relations_model(self, sentences: List[str], entity_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for (entity1, entity2) in entity_pairs[:20]:
            contexts = self._contexts_for_pair(sentences, entity1, entity2)
            if not contexts:
                # No local evidence → skip model guesswork
                continue

            best: Dict[str, Any] = {}
            best_conf = 0.0

            for ctx in contexts:
                input_text = f"{entity1} [SEP] {entity2} [SEP] {ctx[:400]}"
                try:
                    inputs = self.models['relation_tokenizer'](
                        input_text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self.models['relation_model'](**inputs)
                        logits = outputs.logits
                        probs = F.softmax(logits, dim=-1)
                        pred_id = torch.argmax(probs, dim=-1).item()
                        conf = probs[0][pred_id].item()

                    label = self._map_relation_labels(f'LABEL_{pred_id}')
                    # Apply label-specific threshold
                    thr = self.label_thresholds.get(label, self.confidence_threshold)

                    if conf >= thr and conf > best_conf:
                        best = {
                            'entity1': entity1,
                            'entity2': entity2,
                            'relation_type': label,
                            'confidence': round(conf, 3)
                        }
                        best_conf = conf

                except Exception:
                    continue

            if best:
                out.append(best)

        return out

    # ---------- Rule-based extraction (now sentence-scoped) ----------

    def _extract_relations_rule_based(self, sentences: List[str], entity_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        relations: List[Dict[str, Any]] = []

        for (entity1, entity2) in entity_pairs:
            contexts = self._contexts_for_pair(sentences, entity1, entity2)
            for ctx in contexts:
                rel = self._classify_relation_rules(entity1, entity2, ctx)
                if rel:
                    relations.append(rel)
                    break  # one good context is enough

        return relations

    def _classify_relation_rules(self, entity1: str, entity2: str, text_ctx: str) -> Dict[str, Any]:
        e1 = entity1.lower()
        e2 = entity2.lower()
        t = text_ctx.lower()

        # Ignore trivial overlaps
        if e1 == e2 or e1 in e2 or e2 in e1:
            return None

        # Helper for proximity-based boost
        def _gap_boost(s: str) -> float:
            try:
                p1 = s.find(e1)
                p2 = s.find(e2)
                if p1 == -1 or p2 == -1:
                    return 0.0
                gap = abs(p2 - p1)
                if gap <= 40:
                    return 0.10
                if gap <= 80:
                    return 0.05
                return 0.0
            except Exception:
                return 0.0

        # Adverse/causal patterns
        adverse_patterns = [
            rf"{re.escape(e1)}.*(?:cause|caused|causing|induce|induced|inducing|lead to|led to|result in|resulted in).*{re.escape(e2)}",
            rf"{re.escape(e2)}.*(?:due to|caused by|induced by|from|after|following).*{re.escape(e1)}",
            rf"{re.escape(e1)}.*(?:adverse|side effect|toxic|toxicity).*{re.escape(e2)}",
            rf"(?:adverse|side).*(?:effect|reaction).*(?:{re.escape(e1)}).*(?:{re.escape(e2)})",
            rf"{re.escape(e1)}.*associated.*{re.escape(e2)}",
            rf"{re.escape(e1)}.*related.*{re.escape(e2)}"
        ]

        for pat in adverse_patterns:
            if re.search(pat, t):
                conf = 0.7 + _gap_boost(t)
                return {
                    'entity1': entity1,
                    'entity2': entity2,
                    'relation_type': 'drug_causes_effect',
                    'confidence': min(conf, 0.95)
                }

        # Indication patterns
        indication_patterns = [
            rf"{re.escape(e1)}.*(?:treat|treating|treatment|therapy|for).*{re.escape(e2)}",
            rf"{re.escape(e1)}.*(?:indicated|prescribed|used).*(?:for|to treat).*{re.escape(e2)}",
            rf"(?:treatment|therapy).*(?:{re.escape(e2)}).*(?:{re.escape(e1)})"
        ]
        for pat in indication_patterns:
            if re.search(pat, t):
                conf = 0.8 + _gap_boost(t)
                return {
                    'entity1': entity1,
                    'entity2': entity2,
                    'relation_type': 'drug_indication',
                    'confidence': min(conf, 0.92)
                }

        # Co-occurrence with clinical verbs (fallback)
        clinical_verbs = ['observed', 'reported', 'developed', 'experienced', 'showed', 'presented', 'noted', 'occurred']
        if any(v in t for v in clinical_verbs) and self._contains_mention(t, e1) and self._contains_mention(t, e2):
            conf = 0.60 + _gap_boost(t)
            return {
                'entity1': entity1,
                'entity2': entity2,
                'relation_type': 'drug_effect_relation',
                'confidence': min(conf, 0.85)
            }

        return None

    # ---------- Utilities ----------

    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        dedup: List[Dict[str, Any]] = []
        for r in relations:
            key = (r['entity1'].lower(), r['entity2'].lower(), r['relation_type'])
            if key not in seen:
                seen.add(key)
                dedup.append(r)
        return dedup


    def _map_relation_labels(self, label: str) -> str:
        label_mapping = {
            'LABEL_0': 'same_entity',
            'LABEL_1': 'drug_disease_relation',
            'LABEL_2': 'drug_effect_relation',
            'LABEL_3': 'adverse_drug_reaction',
            'LABEL_4': 'drug_indication',
            'LABEL_5': 'drug_causes_effect',
            'LABEL_6': 'side_effect_relation',
            'LABEL_7': 'contraindication'
        }
        return label_mapping.get(label, label)
