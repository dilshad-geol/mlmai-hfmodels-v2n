from typing import List, Dict, Any

class TextProcessor:
    def __init__(self):
        pass
    
    def find_connected_terms(self, drugs: List[str], diseases: List[str], effects: List[str] = None) -> List[Dict[str, Any]]:
        """Find semantically connected terms"""
        connected = []
        effects = effects or []
        all_terms = drugs + diseases + effects
        
        for i, term1 in enumerate(all_terms):
            for j, term2 in enumerate(all_terms[i+1:], i+1):
                similarity = self._calculate_term_similarity(term1, term2)
                if similarity > 0.3:
                    connected.append({
                        'term1': term1,
                        'term2': term2,
                        'similarity': round(similarity, 3)
                    })
        
        return connected
    
    def _calculate_term_similarity(self, term1: str, term2: str) -> float:
        """Calculate Jaccard similarity between terms"""
        term1_words = set(term1.lower().split())
        term2_words = set(term2.lower().split())
        
        if not term1_words or not term2_words:
            return 0.0
            
        intersection = len(term1_words.intersection(term2_words))
        union = len(term1_words.union(term2_words))
        
        return intersection / union if union > 0 else 0.0
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep medical symbols
        import re
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\%\>\<\=]', ' ', text)
        
        return text
    
    def extract_dosage_info(self, text: str) -> List[str]:
        """Extract dosage information from text"""
        import re
        dosage_patterns = [
            r'\b\d+\s*(?:mg|g|ml|mcg|μg|units?)\b',
            r'\b\d+\s*(?:times?|daily|weekly|monthly)\b',
            r'\b(?:once|twice|three times?)\s*(?:daily|weekly)\b'
        ]
        
        dosages = []
        for pattern in dosage_patterns:
            dosages.extend(re.findall(pattern, text.lower()))
        
        return list(set(dosages))
    
    def extract_temporal_info(self, text: str) -> List[str]:
        """Extract temporal information"""
        import re
        temporal_patterns = [
            r'\b\d+\s*(?:days?|weeks?|months?|years?)\b',
            r'\b(?:follow-up|baseline|endpoint|duration)\b',
            r'\b(?:before|after|during|throughout)\s+(?:treatment|therapy)\b'
        ]
        
        temporal_info = []
        for pattern in temporal_patterns:
            temporal_info.extend(re.findall(pattern, text.lower()))
        
        return list(set(temporal_info))