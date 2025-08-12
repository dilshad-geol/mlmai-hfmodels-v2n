# utils/relation_extraction.py
import re, math
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
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = MODEL_CONFIG.get('relation_extraction', {})
        self.confidence_threshold = cfg.get('confidence_threshold', 0.7)
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
        self.margin_min = cfg.get('margin_min', 0.08)
        self.pooling = cfg.get('pooling', 'mean')  # mean | max | lse
        self.max_contexts_per_pair = int(cfg.get('max_contexts_per_pair', 4))
        self.min_entity_len = int(cfg.get('min_entity_len', 3))
        self.alpha_ratio_min = float(cfg.get('alpha_ratio_min', 0.6))

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

        self._rel_tokenizer = self.models.get('relation_tokenizer', None)
        self._rel_model = self.models.get('relation_model', None)

        self._ade_cache: Dict[str, Dict[str, List[str]]] = {}

    def extract_relations(self, text: str, entity_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        if not (self._rel_model and self._rel_tokenizer):
            return []

        entity_pairs = [(d, e) for (d, e) in entity_pairs if self._valid_entity(d) and self._valid_entity(e)]

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

        uniq = self._deduplicate_relations(out)
        uniq.sort(key=lambda r: r['confidence'], reverse=True)
        return uniq[:15]

    def _classify_pair(self, sentences: List[str], e1: str, e2: str) -> Tuple[Optional[str], float]:
        contexts = self._contexts_for_pair(sentences, e1, e2)
        if not contexts:
            return None, 0.0

        valid_ctxs = [c for c in contexts if self._ade_gate(c, e1, e2)]
        if not valid_ctxs:
            return None, 0.0

        texts = [f"{e1} [SEP] {e2} [SEP] {c[:400]}" for c in valid_ctxs[: self.max_contexts_per_pair]]
        logits = self._score_batch(texts)
        if logits is None:
            return None, 0.0

        B = logits.shape[0]
        if self.pooling == "lse":
            pooled_logits = torch.logsumexp(logits, dim=0) - math.log(B)
            probs_pool = F.softmax(pooled_logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            if self.pooling == "max":
                probs_pool, _ = probs.max(dim=0)
            else:
                probs_pool = probs.mean(dim=0)

        top2 = torch.topk(probs_pool, k=2)
        conf = float(top2.values[0].item())
        second = float(top2.values[1].item()) if top2.values.numel() > 1 else 0.0
        if (conf - second) < self.margin_min:
            return None, 0.0

        pred_id = int(top2.indices[0].item())
        label = self._map_relation_labels(f"LABEL_{pred_id}")
        thr = self.label_thresholds.get(label, self.confidence_threshold)
        if conf < thr:
            return None, 0.0
        return label, conf

    def _score_batch(self, texts: List[str]) -> Optional[torch.Tensor]:
        try:
            enc = self._rel_tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", enabled=True):
                        logits = self._rel_model(**enc).logits
                else:
                    logits = self._rel_model(**enc).logits
            return logits.detach().cpu()
        except Exception:
            return None

    def _ade_gate(self, ctx: str, drug: str, eff: str) -> bool:
        if self._ade_pipe is None:
            return self._contains_mention(ctx, drug) and self._contains_mention(ctx, eff)

        key = ctx.lower()
        if key not in self._ade_cache:
            preds = self._ade_pipe(ctx)
            drugs, effects = [], []
            for p in preds:
                span = (p.get('word') or p.get('entity') or '').strip()
                if not span:
                    continue
                grp = (p.get('entity_group') or p.get('entity') or '').upper()
                if 'DRUG' in grp:
                    drugs.append(self._norm(span))
                elif 'EFFECT' in grp:
                    effects.append(self._norm(span))
            self._ade_cache[key] = {'drugs': drugs, 'effects': effects}

        dset = self._ade_cache[key]['drugs']
        eset = self._ade_cache[key]['effects']
        d_ok = self._match_loose(self._norm(drug), dset)
        e_ok = self._match_loose(self._norm(eff), eset)
        return d_ok and e_ok

    def _valid_entity(self, s: str) -> bool:
        if not s:
            return False
        t = re.sub(r'\s+', ' ', s).strip()
        if len(t) < self.min_entity_len:
            return False
        alpha = sum(ch.isalpha() for ch in t)
        return (alpha / max(1, len(t))) >= self.alpha_ratio_min

    @staticmethod
    def _norm(s: str) -> str:
        s = s.casefold()
        s = re.sub(r'[\s\.\,\;\:\(\)\[\]\{\}\-_/\\]+', '', s)
        return s

    @staticmethod
    def _match_loose(target: str, spans: List[str]) -> bool:
        for sp in spans:
            if target in sp or sp in target:
                return True
            ta = set(re.findall(r'[a-z0-9]+', target))
            sa = set(re.findall(r'[a-z0-9]+', sp))
            if ta and sa and len(ta & sa) / len(ta | sa) >= 0.6:
                return True
        return False

    def _split_sentences(self, text: str) -> List[str]:
        safe = re.sub(r'\b(e\.g|i\.e|vs|fig|dr|mr|mrs)\.\s',
                      lambda m: m.group(0).replace('.', '<DOT>'), text, flags=re.I)
        raw = re.split(r'(?<=[.!?])\s+', safe)
        return [s.replace('<DOT>', '.').strip() for s in raw if s.strip()]

    def _contains_mention(self, sentence: str, mention: str) -> bool:
        s = sentence.lower()
        m = mention.lower().strip()
        if not m:
            return False
        pattern = r'\b' + re.escape(m) + r'\b'
        return bool(re.search(pattern, s)) or (m in s)

    def _filter_pairs_by_sentence_cooccurrence(self, sentences: List[str], entity_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
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
                parts = []
                if i - 1 >= 0:
                    parts.append(sentences[i - 1])
                parts.append(sent)
                if i + 1 < len(sentences):
                    parts.append(sentences[i + 1])
                ctx = " ".join([p for p in parts if p]).strip()
                if ctx:
                    contexts.append(ctx)
            elif i + 1 < len(sentences):
                a, b = sentences[i], sentences[i + 1]
                if (self._contains_mention(a, e1) and self._contains_mention(b, e2)) or \
                   (self._contains_mention(a, e2) and self._contains_mention(b, e1)):
                    ctx = (a + " " + b).strip()
                    if ctx:
                        contexts.append(ctx)
        seen = set()
        uniq = []
        for c in contexts:
            k = c.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        return uniq[: self.max_contexts_per_pair]

    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        dedup: List[Dict[str, Any]] = []
        for r in relations:
            key = (r['entity1'].lower(), r['entity2'].lower())
            if key not in seen:
                seen.add(key)
                dedup.append(r)
            else:
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
