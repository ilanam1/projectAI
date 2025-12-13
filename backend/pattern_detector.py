"""
Suspicious Pattern Detection Module.

This module provides extensible, configurable pattern detection for identifying
characteristics commonly found in fake or AI-generated reviews.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ml_config import SuspiciousPatternsConfig, get_config


@dataclass
class PatternMatch:
    """Represents a detected pattern match."""
    pattern_type: str
    confidence: float
    matches: List[str]
    description: str


class SuspiciousPatternDetector:
    """
    Detects suspicious patterns in review text that may indicate fake or AI-generated content.
    
    This class is designed to be extensible and configurable, allowing easy addition
    of new pattern types without modifying core logic.
    """
    
    def __init__(self, config: SuspiciousPatternsConfig = None):
        """
        Initialize the pattern detector.
        
        Args:
            config: Configuration for pattern detection. If None, uses global config.
        """
        self.config = config or get_config().suspicious_patterns
    
    def detect_all_patterns(self, text: str) -> Dict[str, PatternMatch]:
        """
        Detect all suspicious patterns in the given text.
        
        Args:
            text: The text to analyze (can be Hebrew or English)
            
        Returns:
            Dictionary mapping pattern type to PatternMatch object
        """
        # Convert to lowercase for pattern matching (works for both Hebrew and English)
        text_lower = text.lower()
        
        patterns = {}
        
        # Excessive positive words
        positive_match = self._detect_excessive_positive(text_lower)
        if positive_match:
            patterns['excessive_positive'] = positive_match
        
        # Generic language
        generic_match = self._detect_generic_language(text_lower)
        if generic_match:
            patterns['generic_language'] = generic_match
        
        # Lack of details
        details_match = self._detect_lack_of_details(text, text_lower)
        if details_match:
            patterns['lack_of_details'] = details_match
        
        # Repetitive language
        repetitive_match = self._detect_repetitive_language(text_lower)
        if repetitive_match:
            patterns['repetitive'] = repetitive_match
        
        return patterns
    
    def _detect_excessive_positive(self, text_lower: str) -> Optional[PatternMatch]:
        """Detect excessive use of positive words."""
        matches = [word for word in self.config.positive_words if word in text_lower]
        
        if len(matches) >= self.config.excessive_positive_min_count:
            confidence = min(1.0, len(matches) / (self.config.excessive_positive_min_count * 2))
            return PatternMatch(
                pattern_type='excessive_positive',
                confidence=confidence,
                matches=matches,
                description=f"Found {len(matches)} excessive positive words"
            )
        return None
    
    def _detect_generic_language(self, text_lower: str) -> Optional[PatternMatch]:
        """
        Detect generic phrases commonly used in AI-generated and fake reviews.
        
        AI-generated reviews often use generic, template-like language.
        This is the STRONGEST indicator of AI-generated content (ChatGPT).
        """
        matches = [phrase for phrase in self.config.generic_phrases if phrase in text_lower]
        
        # CRITICAL: Even 1 generic phrase is suspicious for AI-generated content
        # Lowered threshold to 1 (was 2) to catch more AI-generated reviews
        if len(matches) >= 1:  # Changed from generic_phrases_min_count (2) to 1
            # Higher confidence calculation - more matches = higher confidence
            # For 1 match: confidence = 0.67, for 2: 0.89, for 3+: 1.0
            confidence = min(1.0, len(matches) / 1.5)  # More aggressive confidence
            return PatternMatch(
                pattern_type='generic_language',
                confidence=confidence,
                matches=matches,
                description=f"Found {len(matches)} generic phrases (AI indicator)"
            )
        return None
    
    def _detect_lack_of_details(self, text: str, text_lower: str) -> Optional[PatternMatch]:
        """
        Detect lack of specific details (short + generic = suspicious).
        
        AI-generated reviews are often vague and lack specific details like:
        - Numbers (prices, dates, quantities)
        - Specific names or places
        - Concrete experiences
        - Detailed descriptions
        """
        text_length = len(text.strip())
        text_words = text.split()
        
        # Check for specific details that indicate authentic reviews
        has_numbers = any(char.isdigit() for char in text)
        has_specific_words = any(word in text_lower for word in [
            'explained', 'told', 'said', 'asked', 'received', 'went', 'came', 'saw',
            'הסביר', 'אמר', 'ביקשתי', 'קיבלתי', 'הגעתי', 'ראיתי', 'מספר', 'שעה'
        ])
        has_names = sum(1 for word in text_words if word and word[0].isupper() and len(word) > 2) > 1
        
        # If review lacks specific details AND is short/generic, it's suspicious
        lacks_specifics = not (has_numbers or has_specific_words or has_names)
        
        # Check for generic language
        generic_matches = [p for p in self.config.generic_phrases if p in text_lower]
        
        if text_length < self.config.short_review_threshold:
            # Short review with generic language
            if len(generic_matches) >= 1:
                confidence = 0.8 if lacks_specifics else 0.6  # Higher confidence if lacks specifics
                return PatternMatch(
                    pattern_type='lack_of_details',
                    confidence=confidence,
                    matches=[f"Length: {text_length} chars", f"Generic phrases: {len(generic_matches)}", 
                            f"Lacks specifics: {lacks_specifics}"],
                    description=f"Short review ({text_length} chars) with generic language and no specific details"
                )
        elif lacks_specifics and len(generic_matches) >= 2:  # Even longer reviews can be suspicious if generic
            # Longer review but lacks specifics and has generic language
            confidence = 0.7
            return PatternMatch(
                pattern_type='lack_of_details',
                confidence=confidence,
                matches=[f"Length: {text_length} chars", f"Generic phrases: {len(generic_matches)}",
                        "Lacks specific details"],
                description=f"Review lacks specific details despite length ({text_length} chars)"
            )
        return None
    
    def _detect_repetitive_language(self, text_lower: str) -> Optional[PatternMatch]:
        """Detect repetitive language patterns."""
        words = text_lower.split()
        if len(words) < 10:
            return None
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find words that appear more than 3 times
        repetitive_words = [word for word, count in word_counts.items() if count > 3]
        
        if len(repetitive_words) > 0:
            confidence = min(0.8, len(repetitive_words) * 0.2)
            return PatternMatch(
                pattern_type='repetitive',
                confidence=confidence,
                matches=repetitive_words[:5],  # Limit to top 5
                description=f"Found {len(repetitive_words)} repetitive words"
            )
        return None
    
    def _has_authentic_details(self, text: str) -> bool:
        """
        Check if text contains authentic details (specifics, not generic).
        
        Authentic reviews (positive OR negative) contain:
        - Specific details (numbers, names, places, dates)
        - Concrete experiences (what happened, when, where)
        - Personal observations (not generic templates)
        
        This is NOT about sentiment - it's about specificity vs. generic AI patterns.
        Authentic = specific details indicating human experience.
        """
        # Check for specific details that indicate human experience (regardless of sentiment)
        has_numbers = any(char.isdigit() for char in text)
        has_specific_references = any(word in text.lower() for word in [
            'explained', 'asked', 'received', 'told', 'said', 'went', 'came', 'saw',
            'tried', 'used', 'bought', 'visited', 'called', 'contacted', 'met',
            'הסביר', 'ביקשתי', 'קיבלתי', 'אמר', 'הגעתי', 'ראיתי', 'היה', 'מספר',
            'קניתי', 'ניסיתי', 'ביקרתי', 'פניתי', 'פגשתי'
        ])
        has_names_or_places = sum(1 for word in text.split() if word and word[0].isupper()) > 2
        has_concrete_actions = sum(1 for word in text.split() if len(word) > 8) > 2
        
        return has_numbers or has_specific_references or has_names_or_places or has_concrete_actions
    
    def _has_specific_details(self, text: str) -> bool:
        """Check if text contains specific details (suggests genuine review)."""
        # Check for specific details: numbers, locations, specific features
        has_numbers = any(char.isdigit() for char in text)
        has_capitalized = sum(1 for word in text.split() if word and word[0].isupper()) > 3
        has_long_words = sum(1 for word in text.split() if len(word) > 8) > 2
        return has_numbers or has_capitalized or has_long_words
    
    def calculate_suspicious_score(self, patterns: Dict[str, PatternMatch], text: str = "") -> float:
        """
        Calculate an overall suspicious score from detected patterns.
        
        Args:
            patterns: Dictionary of detected patterns
            text: Original text (for context checks)
            
        Returns:
            Suspicious score between 0.0 and 1.0
        """
        if not patterns:
            return 0.0
        
        text_lower = text.lower() if text else ""
        
        # If review has authentic details (specifics, not generic), reduce suspicious score
        # This applies to BOTH positive and negative authentic reviews
        # We're detecting AUTHENTICITY (specificity), not sentiment
        # BUT: If review is VERY generic (multiple generic phrases), don't reduce too much
        generic_count = len([p for p in self.config.generic_phrases if p in text_lower]) if text else 0
        
        if text and self._has_authentic_details(text):
            # Authentic reviews have specific details regardless of sentiment
            # But if it's also very generic, reduce less
            if generic_count >= 3:  # Very generic despite having some details
                reduction_factor = 0.7  # Less reduction
            else:
                reduction_factor = 0.4  # Normal reduction for authentic details
        else:
            reduction_factor = 1.0
        
        # Weight different patterns differently
        # CRITICAL: generic_language is the STRONGEST indicator of AI-generated content
        weights = {
            'excessive_positive': 0.15,   # Reduced
            'generic_language': 0.7,       # DRAMATICALLY increased - strongest ChatGPT indicator
            'lack_of_details': 0.2,       # Reduced (generic_language is more important)
            'repetitive': 0.05             # Reduced
        }
        
        weighted_sum = sum(
            weights.get(pattern_type, 0.25) * match.confidence
            for pattern_type, match in patterns.items()
        )
        
        # Normalize to 0-1 range
        total_weight = sum(weights.get(pt, 0.25) for pt in patterns.keys())
        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Apply reduction for genuine reviews
        score = score * reduction_factor
        
        return min(1.0, score)

