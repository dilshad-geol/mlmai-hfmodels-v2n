import re
import torch
from typing import Dict, Any, List
from config.settings import RELEVANCE_CONFIG, PROCESSING_CONFIG

class RelevanceScorer:
    """
    Calculate pharmacovigilance relevance scores using multiple algorithms
    and biomedical context understanding.
    """
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.device = PROCESSING_CONFIG['device']
        self.config = RELEVANCE_CONFIG
    
    def calculate_score(self, title: str, abstract: str) -> float:
        """
        Calculate comprehensive pharmacovigilance relevance score
        
        Args:
            title: Research paper title
            abstract: Research paper abstract
            
        Returns:
            Float between 0.0 and 1.0 indicating relevance
        """
        text = f"{title} {abstract}".lower()
        
        # Core scoring components
        keyword_score = self._calculate_keyword_score(text)
        clinical_score = self._calculate_clinical_score(text)
        length_score = self._calculate_length_score(text)
        entity_density_score = self._calculate_entity_density(text)
        
        # Advanced scoring components
        domain_specificity_score = self._calculate_domain_specificity(text)
        temporal_relevance_score = self._calculate_temporal_relevance(text)
        
        # Clinical BERT enhancement if available
        bert_enhancement = 0.0
        if self.models.get('clinical_model') and self.models.get('clinical_tokenizer'):
            try:
                bert_enhancement = self._process_with_clinical_bert(text)
            except Exception as e:
                print(f"Clinical BERT processing error: {e}")
        
        # Weighted combination
        weights = self.config['scoring_weights']
        base_score = (
            keyword_score * weights['keyword_score'] +
            clinical_score * weights['clinical_score'] +
            length_score * weights['length_penalty'] +
            entity_density_score * weights['entity_density']
        )
        
        # Apply enhancements
        enhanced_score = base_score + (domain_specificity_score * 0.1) + (temporal_relevance_score * 0.05) + (bert_enhancement * 0.1)
        
        return min(max(enhanced_score, 0.0), 1.0)
    
    def _calculate_keyword_score(self, text: str) -> float:
        """Calculate score based on pharmacovigilance keyword matching"""
        high_value_keywords = self.config['high_value_keywords']
        medium_value_keywords = self.config['medium_value_keywords']
        pharma_keywords = self.config['pharma_keywords']
        
        # Count matches with different weights
        high_matches = sum(1 for keyword in high_value_keywords if keyword in text)
        medium_matches = sum(1 for keyword in medium_value_keywords if keyword in text)
        total_matches = sum(1 for keyword in pharma_keywords if keyword in text)
        
        # Calculate weighted score
        weighted_score = (high_matches * 0.15) + (medium_matches * 0.05) + (total_matches * 0.03)
        
        return min(weighted_score, 1.0)
    
    def _calculate_clinical_score(self, text: str) -> float:
        """Calculate clinical terminology density score"""
        clinical_patterns = [
            r'\b(?:drug|medication|adverse|effect|reaction|treatment|therapy)\b',
            r'\b(?:clinical|patient|dose|administration|safety|toxicity)\b',
            r'\b(?:side effect|pharmacokinetics|pharmacodynamics|contraindication)\b',
            r'\b(?:interaction|monitoring|follow-up|retrospective|study|trial)\b',
            r'\b(?:efficacy|effectiveness|tolerability|adverse event|adr|ade)\b'
        ]
        
        clinical_terms = []
        for pattern in clinical_patterns:
            clinical_terms.extend(re.findall(pattern, text))
        
        # Normalize by text length to get density
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
            
        density = len(clinical_terms) / word_count
        return min(density * 10, 1.0)  # Scale factor for appropriate range
    
    def _calculate_length_score(self, text: str) -> float:
        """Calculate length-based quality score"""
        word_count = len(text.split())
        
        # Optimal range: 100-300 words for abstracts
        if word_count < 50:
            return word_count / 50  # Penalty for very short texts
        elif word_count <= 300:
            return 1.0  # Optimal range
        else:
            return max(0.5, 1.0 - (word_count - 300) / 1000)  # Gentle penalty for very long texts
    
    def _calculate_entity_density(self, text: str) -> float:
        """Calculate medical entity density"""
        medical_entity_patterns = [
            r'\b(?:mg|g|ml|mcg|μg|units?|daily|weekly|monthly)\b',  # Dosage terms
            r'\b(?:patients?|cases?|subjects?|individuals?)\b',      # Population terms
            r'\b(?:study|analysis|trial|research|investigation)\b',  # Study terms
            r'\b(?:group|cohort|sample|population)\b',               # Group terms
            r'\b(?:risk|factors?|results?|outcomes?|findings?)\b',   # Result terms
            r'\b(?:suggest|observed|compared|decreased|increased)\b', # Analysis terms
            r'\b(?:elevated|reduced|significant|associated)\b',      # Statistical terms
            r'\b(?:concomitant|concurrent|simultaneous)\b'          # Temporal terms
        ]
        
        entity_matches = []
        for pattern in medical_entity_patterns:
            entity_matches.extend(re.findall(pattern, text))
        
        # Calculate density
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
            
        density = len(entity_matches) / word_count
        return min(density * 15, 1.0)  # Scale factor
    
    def _calculate_domain_specificity(self, text: str) -> float:
        """Calculate domain-specific terminology score"""
        pharmacovigilance_terms = [
            r'\b(?:pharmacovigilance|drug safety|adverse drug reaction)\b',
            r'\b(?:post-marketing|spontaneous reporting|signal detection)\b',
            r'\b(?:causality assessment|disproportionality|safety profile)\b',
            r'\b(?:hepatotoxicity|cardiotoxicity|nephrotoxicity|neurotoxicity)\b',
            r'\b(?:black box warning|contraindication|drug interaction)\b',
            r'\b(?:dose-dependent|idiosyncratic|hypersensitivity)\b'
        ]
        
        domain_matches = []
        for pattern in pharmacovigilance_terms:
            domain_matches.extend(re.findall(pattern, text))
        
        return min(len(domain_matches) * 0.2, 1.0)
    
    def _calculate_temporal_relevance(self, text: str) -> float:
        """Calculate temporal relevance for pharmacovigilance"""
        temporal_patterns = [
            r'\b(?:follow-up|followup|longitudinal|prospective|retrospective)\b',
            r'\b(?:baseline|endpoint|duration|time-to|onset)\b',
            r'\b(?:acute|chronic|short-term|long-term)\b',
            r'\b(?:before|after|during|throughout|within)\s+(?:treatment|therapy)\b',
            r'\b\d+\s*(?:days?|weeks?|months?|years?)\b'
        ]
        
        temporal_matches = []
        for pattern in temporal_patterns:
            temporal_matches.extend(re.findall(pattern, text))
        
        return min(len(temporal_matches) * 0.1, 1.0)
    
    def _process_with_clinical_bert(self, text: str) -> float:
        """
        Process text with Clinical BERT for enhanced understanding
        
        Returns:
            Enhancement score based on BERT embeddings
        """
        try:
            # Check if models are available
            if not self.models.get('clinical_model') or not self.models.get('clinical_tokenizer'):
                return 0.0
                
            # Tokenize and encode
            inputs = self.models['clinical_tokenizer'](
                text[:512], 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['clinical_model'](**inputs)
                
                # Get embeddings and calculate semantic richness
                if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    
                    # Calculate embedding-based metrics
                    embedding_norm = torch.norm(embeddings).item()
                    embedding_richness = min(embedding_norm / 100, 1.0)  # Normalize
                    
                    # Attention-based relevance (if available)
                    attention_score = 0.0
                    if hasattr(outputs, 'attentions') and outputs.attentions is not None and len(outputs.attentions) > 0:
                        # Calculate attention entropy as a measure of complexity
                        attention_weights = outputs.attentions[-1]
                        if attention_weights is not None:
                            attention_mean = attention_weights.mean(dim=(1, 2, 3))
                            attention_score = min(attention_mean.item() * 10, 1.0)
                    
                    return (embedding_richness + attention_score) / 2
                else:
                    return 0.0
                
        except Exception as e:
            print(f"BERT processing error: {e}")
            return 0.0
    
    def get_score_breakdown(self, title: str, abstract: str) -> Dict[str, float]:
        """
        Get detailed breakdown of relevance score components
        
        Returns:
            Dictionary with individual component scores
        """
        text = f"{title} {abstract}".lower()
        
        breakdown = {
            'keyword_score': self._calculate_keyword_score(text),
            'clinical_score': self._calculate_clinical_score(text),
            'length_score': self._calculate_length_score(text),
            'entity_density_score': self._calculate_entity_density(text),
            'domain_specificity_score': self._calculate_domain_specificity(text),
            'temporal_relevance_score': self._calculate_temporal_relevance(text)
        }
        
        if self.models.get('clinical_model'):
            breakdown['bert_enhancement'] = self._process_with_clinical_bert(text)
        
        breakdown['total_score'] = self.calculate_score(title, abstract)
        
        return breakdown
    
    def classify_relevance(self, score: float) -> str:
        """
        Classify relevance score into categories
        
        Args:
            score: Relevance score (0.0-1.0)
            
        Returns:
            String classification
        """
        if score >= 0.8:
            return "Very High"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def extract_key_indicators(self, text: str) -> List[str]:
        """
        Extract key pharmacovigilance indicators from text
        
        Args:
            text: Input text
            
        Returns:
            List of key indicator phrases
        """
        indicator_patterns = [
            r'\b(?:adverse (?:drug )?(?:reaction|effect|event)s?)\b',
            r'\b(?:side effects?)\b',
            r'\b(?:drug(?:\s+|-)?induced)\b',
            r'\b(?:safety (?:profile|concern|issue)s?)\b',
            r'\b(?:contraindication|drug interaction)\b',
            r'\b(?:hepatotoxic|cardiotoxic|nephrotoxic|neurotoxic)\b',
            r'\b(?:black box warning|boxed warning)\b',
            r'\b(?:dose(?:\s+|-)?dependent)\b',
            r'\b(?:pharmacovigilance|drug safety)\b'
        ]
        
        indicators = []
        text_lower = text.lower()
        
        for pattern in indicator_patterns:
            matches = re.findall(pattern, text_lower)
            indicators.extend(matches)
        
        return list(set(indicators))  # Remove duplicates