# utils/relation_extraction.py
import re
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn.functional as F

from config.settings import MODEL_CONFIG

try:
    from transformers import pipeline
    _TRANSFORMERS_OK = True
except Exception:
    _TRANSFORMERS_OK = False


class RelationExtractor:
    """
    Model-only relation extractor with typed gating.

    Key ideas:
      • Use NER outputs upstream to build pairs (drug → effect/disease).
      • For each pair, build short local contexts (same sentence or ±1).
      • Gate each context with ADE NER: the context must actually contain a DRUG
        mention matching entity1 and an EFFECT/DISORDER/SYMPTOM-like mention
        matching entity2 (validated by the ADE model) before classification.
      • Run the HF relation model on all valid contexts and aggregate probabilities.
      • Apply per-label confidence thresholds; keep at most one label per pair.

    No hardcoded patterns/regex rules are used for classification; all signals come
    from the models and the typed entity lists.
    """

    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Thresholds (data-driven tuning friendly)
        self.confidence_threshold = MODEL_CONFIG.get('relation_extraction', {}).get('confidence_threshold', 0.7)
        self.label_thresholds = {
            'same_entity':           max(0.75, self.confidence_threshold),
            'drug_disease_relation': max(0.70, self.confidence_threshold - 0.05),
            'drug_effect_relation':  max(0.70, self.confidence_threshold - 0.05),
            'adverse_drug_reaction': max(0.75, self.confidence_threshold),
            'drug_indication':       max(0.75, self.confidence_threshold),
            'drug_causes_effect':    max(0.75, self.confidence_threshold),
            'side_effect_relation':  max(0.70, self.confidence_threshold - 0.05),
            'contraindication':      max(0.75, self.confidence_threshold),
        }

        # Build ADE NER pipeline for typed gating (model-based, no heuristics)
        self._ade_pipe = None
        if _TRANSFORMERS_OK and self.models.get('ade_ner_model') and self.models.get('ade_ner_tokenizer'):
            try:
                device_idx = 0 if self.device.type == "cuda" else -1
                self._ade_pipe = pipeline(
                    task="token-classification",
                    model=self.models['ade_ner_model'],
                    tokenizer=self.models['ade_ner_tokenizer'],
                    device=device_idx,
                    aggregation_strategy="simple",
                    ignore_labels=["O"]
                )
            except Exception:
                self._ade_pipe = None

        # Relation classifier bits
        self._rel_tokenizer = self.models.get('relation_tokenizer', None)
        self._rel_model = self.models.get('relation_model', None)

        # Small cache to avoid re-running ADE on the same context string repeatedly
        self._ade_cache: Dict[str, Dict[str, List[str]]] = {}

    # ---------- Public API ----------

    def extract_relations(self, text: str, entity_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Purely model-driven extraction over typed pairs.
        """
        if not (self._rel_model and self._rel_tokenizer):
            # No relation model available → nothing to do
            return []

        sentences = self._split_sentences(text)
        pairs_scoped = self._filter_pairs_by_sentence_cooccurrence(sentences, entity_pairs)

        out: List[Dict[str, Any]] = []
        for (drug, effect_like) in pairs_scoped:
            label, conf = self._classify_pair(sentences, drug, effect_like)
            if label is None:
                continue
            out.append({
                'entity1': drug,
                'entity2': effect_like,
                'relation_type': label,
                'confidence': round(conf, 3)
            })

        # Dedup and sort
        uniq = self._deduplicate_relations(out)
        uniq.sort(key=lambda r: r['confidence'], reverse=True)
        return uniq[:15]

    # ---------- Pair classification ----------

    def _classify_pair(self, sentences: List[str], e1: str, e2: str) -> Tuple[Optional[str], float]:
        """
        Build contexts → ADE-gate them → classify each → aggregate probs → decide label.
        """
        contexts = self._contexts_for_pair(sentences, e1, e2)
        if not contexts:
            return None, 0.0

        # Evaluate all valid contexts, sum probabilities
        prob_sum = None
        n_ctx_used = 0

        for ctx in contexts:
            # ADE-typed gating: must see DRUG≈e1 and EFFECT≈e2 inside the SAME context.
            if not self._ade_gate(ctx, e1, e2):
                continue

            logits = self._forward_rel(f"{e1} [SEP] {e2} [SEP] {ctx[:400]}")
            if logits is None:
                continue

            probs = F.softmax(logits, dim=-1).detach().cpu()
            prob_sum = probs if prob_sum is None else (prob_sum + probs)
            n_ctx_used += 1

        if n_ctx_used == 0 or prob_sum is None:
            return None, 0.0

        # Aggregate (mean prob across valid contexts)
        probs_mean = prob_sum / n_ctx_used
        pred_id = int(torch.argmax(probs_mean, dim=-1).item())
        conf = float(probs_mean[0, pred_id].item())
        label = self._map_relation_labels(f"LABEL_{pred_id}")

        # Per-label thresholding
        thr = self.label_thresholds.get(label, self.confidence_threshold)
        if conf < thr:
            return None, 0.0

        return label, conf

    def _forward_rel(self, input_text: str) -> Optional[torch.Tensor]:
        try:
            inputs = self._rel_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self._rel_model(**inputs)
                return outputs.logits  # [1, num_labels]
        except Exception:
            return None

    # ---------- ADE typed gating (model-based) ----------

    def _ade_gate(self, ctx: str, drug: str, eff: str) -> bool:
        """
        Accept context only if ADE NER sees a DRUG matching `drug`
        and an EFFECT matching `eff` within this context window.
        No hand-coded keywords; purely model-based.
        """
        if self._ade_pipe is None:
            # If ADE model is unavailable, fall back to robust mention check only.
            return self._contains_mention(ctx, drug) and self._contains_mention(ctx, eff)

        ctx_key = ctx.lower()
        if ctx_key not in self._ade_cache:
            preds = self._ade_pipe(ctx)
            drugs, effects = [], []
            for p in preds:
                span = (p.get('word') or p.get('entity') or '').strip()
                if not span:
                    continue
                grp = (p.get('entity_group') or p.get('entity') or '').upper()
                if 'DRUG' in grp:
                    drugs.append(span)
                elif 'EFFECT' in grp:
                    effects.append(span)
            self._ade_cache[ctx_key] = {
                'drugs': [self._norm(x) for x in drugs],
                'effects': [self._norm(x) for x in effects],
            }

        dset = self._ade_cache[ctx_key]['drugs']
        eset = self._ade_cache[ctx_key]['effects']
        d_ok = self._match_loose(self._norm(drug), dset)
        e_ok = self._match_loose(self._norm(eff), eset)

        return d_ok and e_ok

    @staticmethod
    def _norm(s: str) -> str:
        # Normalize for loose matching without hardcoded domain rules
        return re.sub(r'\s+', '', s.lower())

    @staticmethod
    def _match_loose(target: str, spans: List[str]) -> bool:
        # Accept if the ADE span contains the target or vice versa (handles partials / morphology)
        for sp in spans:
            if target in sp or sp in target:
                return True
        return False

    # ---------- Sentence & context utilities ----------

    def _split_sentences(self, text: str) -> List[str]:
        safe = re.sub(
            r'\b(e\.g|i\.e|vs|fig|dr|mr|mrs)\.\s',
            lambda m: m.group(0).replace('.', '<DOT>'),
            text,
            flags=re.I
        )
        raw = re.split(r'(?<=[.!?])\s+', safe)
        return [s.replace('<DOT>', '.').strip() for s in raw if s.strip()]

    def _contains_mention(self, sentence: str, mention: str) -> bool:
        s = sentence.lower()
        m = mention.lower().strip()
        if not m:
            return False
        pattern = r'\b' + re.escape(m) + r'\b'
        return bool(re.search(pattern, s)) or (m in s)

    def _filter_pairs_by_sentence_cooccurrence(
        self, sentences: List[str], entity_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        kept: List[Tuple[str, str]] = []
        for drug, disease in entity_pairs:
            for i, sent in enumerate(sentences):
                in_curr = self._contains_mention(sent, drug) and self._contains_mention(sent, disease)
                if in_curr:
                    kept.append((drug, disease))
                    break
                if i + 1 < len(sentences):
                    near_next = (self._contains_mention(sent, drug) and self._contains_mention(sentences[i + 1], disease)) or \
                                (self._contains_mention(sent, disease) and self._contains_mention(sentences[i + 1], drug))
                    if near_next:
                        kept.append((drug, disease))
                        break
        return kept

    def _contexts_for_pair(self, sentences: List[str], e1: str, e2: str) -> List[str]:
        contexts: List[str] = []
        for i, sent in enumerate(sentences):
            if self._contains_mention(sent, e1) and self._contains_mention(sent, e2):
                ctx = " ".join([s for s in (sentences[i - 1] if i - 1 >= 0 else "", sent, sentences[i + 1] if i + 1 < len(sentences) else "") if s]).strip()
                if ctx:
                    contexts.append(ctx)
            elif i + 1 < len(sentences):
                a, b = sentences[i], sentences[i + 1]
                if (self._contains_mention(a, e1) and self._contains_mention(b, e2)) or \
                   (self._contains_mention(a, e2) and self._contains_mention(b, e1)):
                    ctx = (a + " " + b).strip()
                    if ctx:
                        contexts.append(ctx)

        # De-duplicate, cap to keep inference inexpensive
        seen = set()
        uniq = []
        for c in contexts:
            k = c.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        return uniq[:4]

    # ---------- Post-processing ----------

    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        dedup: List[Dict[str, Any]] = []
        for r in relations:
            key = (r['entity1'].lower(), r['entity2'].lower())  # keep one label per pair (highest-conf)
            if key not in seen:
                seen.add(key)
                dedup.append(r)
            else:
                # keep the highest-confidence one if duplicate pair appears
                for i, prev in enumerate(dedup):
                    if (prev['entity1'].lower(), prev['entity2'].lower()) == key and r['confidence'] > prev['confidence']:
                        dedup[i] = r
                        break
        return dedup

    def _map_relation_labels(self, label: str) -> str:
        mapping = {
            'LABEL_0': 'same_entity',
            'LABEL_1': 'drug_disease_relation',
            'LABEL_2': 'drug_effect_relation',
            'LABEL_3': 'adverse_drug_reaction',
            'LABEL_4': 'drug_indication',
            'LABEL_5': 'drug_causes_effect',
            'LABEL_6': 'side_effect_relation',
            'LABEL_7': 'contraindication'
        }
        return mapping.get(label, label)
