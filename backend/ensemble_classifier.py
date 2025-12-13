"""
Ensemble Classification Module.

This module implements a robust, calibrated ensemble algorithm that combines
multiple ML models with pattern detection for accurate fake review classification.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ml_config import SystemConfig, get_config
from pattern_detector import SuspiciousPatternDetector, PatternMatch


@dataclass
class ModelOutput:
    """Structured output from a single model."""
    fake_score: float
    real_score: float
    confidence: float
    model_name: str
    raw_output: Any = None


@dataclass
class ClassificationResult:
    """Final classification result with full metadata."""
    classification: str  # 'REAL', 'FAKE', 'FAKE (AI Detected)', 'FAKE (Suspicious Patterns)', 'UNCERTAIN'
    fake_probability: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    model_used: str
    model_outputs: Dict[str, ModelOutput]
    suspicious_patterns: Dict[str, PatternMatch]
    suspicious_score: float
    reasoning: str  # Human-readable explanation


class EnsembleClassifier:
    """
    Robust ensemble classifier that combines multiple models with pattern detection.
    
    This class implements a calibrated, production-ready ensemble algorithm that:
    1. Combines outputs from multiple ML models
    2. Incorporates suspicious pattern detection
    3. Applies calibrated thresholds for decision making
    4. Provides explainable results
    """
    
    def __init__(self, config: SystemConfig = None):
        """
        Initialize the ensemble classifier.
        
        Args:
            config: System configuration. If None, uses global config.
        """
        self.config = config or get_config()
        self.pattern_detector = SuspiciousPatternDetector(self.config.suspicious_patterns)
        self.thresholds = self.config.thresholds
    
    def classify(
        self,
        model1_output: ModelOutput,
        model2_output: ModelOutput,
        suspicious_patterns: Dict[str, PatternMatch],
        suspicious_score: float,
        translated_text: str,
        original_text: str = ""
    ) -> ClassificationResult:
        """
        Perform ensemble classification with pattern detection.
        
        Args:
            model1_output: Output from review-specific classifier
            model2_output: Output from AI detector
            suspicious_patterns: Detected suspicious patterns
            suspicious_score: Overall suspicious score (0.0-1.0)
            translated_text: Translated text (for logging)
            
        Returns:
            ClassificationResult with full metadata
        """
        # Step 1: Combine model outputs with adaptive weighting
        combined_scores = self._combine_model_outputs(model1_output, model2_output)
        
        # Step 2: Apply suspicious pattern adjustments
        adjusted_scores = self._apply_pattern_adjustments(
            combined_scores,
            suspicious_score,
            suspicious_patterns
        )
        
        # Step 3: Make final classification decision (with confidence calibration)
        classification, confidence, reasoning = self._make_decision(
            adjusted_scores,
            model1_output,
            model2_output,
            suspicious_score,
            suspicious_patterns,
            original_text=original_text
        )
        
        # Step 4: Calculate final fake probability (using calibrated confidence)
        fake_probability = self._calculate_fake_probability(
            adjusted_scores,
            classification,
            suspicious_score,
            confidence  # Pass calibrated confidence
        )
        
        # FINAL VALIDATION: Ensure all values are in valid ranges
        fake_probability = max(0.0, min(1.0, float(fake_probability)))
        confidence = max(0.0, min(1.0, float(confidence)))
        suspicious_score = max(0.0, min(1.0, float(suspicious_score)))
        
        return ClassificationResult(
            classification=classification,
            fake_probability=fake_probability,
            confidence=confidence,
            model_used=self._determine_model_used(model1_output, model2_output, suspicious_score),
            model_outputs={
                'model1': model1_output,
                'model2': model2_output
            },
            suspicious_patterns=suspicious_patterns,
            suspicious_score=suspicious_score,
            reasoning=reasoning
        )
    
    def _combine_model_outputs(
        self,
        model1: ModelOutput,
        model2: ModelOutput
    ) -> Tuple[float, float]:
        """
        Combine model outputs with adaptive weighting based on confidence.
        
        Returns:
            Tuple of (combined_fake_score, combined_real_score)
        """
        weights = self.config.ensemble_weights
        
        # Give Model 2 (AI Detector) more weight when it detects ANY AI signal
        if model2.fake_score > self.thresholds.ai_detection_threshold:
            m1_weight = 0.15  # Reduced from 0.2
            m2_weight = 0.85  # Increased from 0.8
        elif model2.fake_score > 0.20:  # Lowered from 0.25
            m1_weight = 0.3   # Reduced from 0.4
            m2_weight = 0.7   # Increased from 0.6
        elif model2.fake_score > 0.12:  # Lowered from 0.15
            m1_weight = 0.4   # Reduced from 0.5
            m2_weight = 0.6   # Increased from 0.5
        else:
            m1_weight = weights.model1_weight
            m2_weight = weights.model2_weight
        
        combined_fake = (model1.fake_score * m1_weight) + (model2.fake_score * m2_weight)
        combined_real = (model1.real_score * m1_weight) + (model2.real_score * m2_weight)
        
        total = combined_fake + combined_real
        if total > 0:
            combined_fake /= total
            combined_real /= total
        
        return combined_fake, combined_real
    
    def _apply_pattern_adjustments(
        self,
        combined_scores: Tuple[float, float],
        suspicious_score: float,
        patterns: Dict[str, PatternMatch]
    ) -> Tuple[float, float]:
        """
        Apply adjustments based on suspicious patterns.
        
        Returns:
            Tuple of (adjusted_fake_score, adjusted_real_score)
        """
        fake_score, real_score = combined_scores
        
        # More aggressive adjustments to catch AI-generated content
        # Especially for generic/vague reviews (common in ChatGPT output)
        if suspicious_score > 0.55:  # Lowered from 0.6 - catch more AI-generated content
            if real_score > 0.75:
                fake_score = max(fake_score, 0.85)  # Increased from 0.8
                real_score = min(real_score, 0.15)  # Decreased from 0.2
            elif real_score > 0.6:
                fake_score = max(fake_score, 0.75)  # Increased from 0.7
                real_score = min(real_score, 0.25)  # Decreased from 0.3
            else:
                fake_score = max(fake_score, 0.7)  # Increased from 0.65
                real_score = min(real_score, 0.3)  # Decreased from 0.35
        elif suspicious_score > self.thresholds.suspicious_patterns_threshold:
            # More aggressive adjustment for moderate suspicious patterns
            # Generic language + lack of details = likely AI-generated
            adjustment = suspicious_score * 0.7  # Increased from 0.6
            fake_score = min(1.0, fake_score + adjustment)
            real_score = max(0.0, real_score - adjustment)
        
        return fake_score, real_score
    
    def _calibrate_confidence(
        self,
        base_confidence: float,
        model1: ModelOutput,
        model2: ModelOutput,
        suspicious_score: float,
        fake_score: float,
        real_score: float
    ) -> float:
        """
        Calibrate confidence to prevent overconfidence in ambiguous cases.
        """
        calibrated = base_confidence
        model_disagreement = abs(model1.fake_score - model2.fake_score)
        
        if suspicious_score > 0.5:
            if real_score > 0.7:
                overconfidence_penalty = (suspicious_score - 0.5) * 0.6
                calibrated = max(0.4, calibrated - overconfidence_penalty)
        
        if model_disagreement > self.thresholds.model_disagreement_threshold:
            disagreement_penalty = model_disagreement * 0.3
            calibrated = max(0.3, calibrated - disagreement_penalty)
        
        if (model1.real_score > self.thresholds.overconfidence_penalty_threshold and 
            model2.fake_score > 0.2):
            overconfidence_penalty = (model1.real_score - 0.9) * 0.5
            calibrated = max(0.4, calibrated - overconfidence_penalty)
        
        if suspicious_score > 0.6 and real_score > fake_score:
            calibrated = min(0.7, calibrated)
        
        return min(1.0, max(0.0, calibrated))
    
    def _make_decision(
        self,
        adjusted_scores: Tuple[float, float],
        model1: ModelOutput,
        model2: ModelOutput,
        suspicious_score: float,
        patterns: Dict[str, PatternMatch],
        original_text: str = ""
    ) -> Tuple[str, float, str]:
        """
        Make final classification decision with reasoning and calibrated confidence.
        
        Returns:
            Tuple of (classification, confidence, reasoning)
        """
        fake_score, real_score = adjusted_scores
        base_confidence = max(fake_score, real_score)
        
        # PRIORITY 0: Generic language = STRONG indicator of AI-generated content (ChatGPT)
        # BUT: Check for authentic details first - if review has specific details, it might be authentic
        # Generic language + lack of authentic details = FAKE
        # Generic language + authentic details = Check Model 2 and other signals
        if 'generic_language' in patterns:
            # Check if review has authentic details (specifics that indicate human experience)
            has_authentic_details = False
            if original_text:
                # Use pattern detector to check for authentic details
                has_authentic_details = self.pattern_detector._has_authentic_details(original_text)
            
            generic_confidence = patterns['generic_language'].confidence
            generic_phrase_count = len(patterns['generic_language'].matches) if 'generic_language' in patterns else 0
            
            # Decision logic:
            # 1. If has authentic details + Model 2 suggests human → REAL (authentic review with common phrase)
            # 2. If no authentic details + multiple generic phrases → FAKE (AI pattern)
            # 3. If no authentic details + single generic phrase + Model 2 uncertain → FAKE
            # 4. If Model 2 VERY confident (>0.85) → Trust Model 2 (unless very generic)
            
            if has_authentic_details and model2.real_score > 0.65:
                # Generic language BUT has authentic details + Model 2 suggests human = REAL
                classification = 'REAL'
                base_confidence = model2.real_score
                reasoning = f"Specific personal experience with natural Hebrew phrasing and clear contextual detail. Contains authentic details (numbers, specific references, names) indicating genuine human experience. AI detector confirms human-written content ({model2.real_score:.2%})."
            elif not has_authentic_details and generic_phrase_count >= 2:
                # Multiple generic phrases + no authentic details = likely AI-generated
                if 'lack_of_details' in patterns:
                    classification = 'FAKE (Generic Language + Lack of Details - AI Pattern)'
                    base_confidence = max(0.85, suspicious_score * 0.98)
                else:
                    classification = 'FAKE (Generic Language - AI Pattern)'
                    base_confidence = max(0.8, suspicious_score * 0.95)
                reasoning = f"Generic, repetitive phrasing without personal detail; resembles AI-generated promotional content. Found {generic_phrase_count} generic phrases with no authentic details (numbers, names, specific experiences). AI detector suggests generated content ({model2.fake_score:.2%})."
            elif not has_authentic_details and generic_phrase_count == 1:
                # Single generic phrase - check Model 2 more carefully
                if model2.real_score > 0.75:  # Model 2 suggests human
                    classification = 'REAL'
                    base_confidence = model2.real_score
                    reasoning = f"Natural Hebrew phrasing with authentic personal experience. Single generic phrase detected but AI detector strongly indicates human-written content ({model2.real_score:.2%}). Likely authentic review using common expression."
                elif model2.fake_score > 0.3:  # Model 2 suggests AI
                    classification = 'FAKE (Generic Language - AI Pattern)'
                    base_confidence = max(0.75, suspicious_score * 0.9)
                    reasoning = f"Generic phrasing combined with AI detection signal ({model2.fake_score:.2%}) indicates AI-generated content. Lacks specific personal details or authentic experiences."
                else:
                    # Uncertain - check other signals (fall through)
                    pass
            elif model2.real_score > 0.85:  # Model 2 VERY confident it's human
                # Very high Model 2 confidence overrides generic language (unless very generic)
                if generic_phrase_count >= 3:  # Very generic (3+ phrases)
                    classification = 'FAKE (Generic Language - AI Pattern)'
                    base_confidence = max(0.8, suspicious_score * 0.95)
                    reasoning = f"Very strong generic language pattern ({generic_phrase_count} phrases) indicates AI-generated content despite Model 2 uncertainty. Model 2: {model2.fake_score:.2%} fake, {model2.real_score:.2%} real."
                else:
                    classification = 'REAL'
                    base_confidence = model2.real_score
                    reasoning = f"AI detector very strongly indicates human-written content ({model2.real_score:.2%}) despite generic language pattern."
            else:
                # Generic language + uncertain = check other signals (fall through to other priorities)
                pass
        
        # PRIORITY 1: Model 2 (AI Detector) - strongest signal for AI-generated content
        # Lowered threshold to catch more AI content
        elif model2.fake_score > self.thresholds.ai_detection_threshold:
            if suspicious_score > 0.2:  # Lowered from 0.25
                classification = 'FAKE (AI Detected + Suspicious Patterns)'
                base_confidence = max(model2.fake_score, 0.8)  # Increased from 0.75
                reasoning = f"AI detector identified generated content ({model2.fake_score:.2%}) combined with suspicious patterns ({suspicious_score:.2f})."
            else:
                classification = 'FAKE (AI Detected)'
                base_confidence = max(model2.fake_score, 0.7)  # Increased from 0.65
                reasoning = f"AI-generated content detected with {model2.fake_score:.2%} confidence. Generic phrasing and lack of authentic personal details indicate synthetic origin."
        
        # PRIORITY 2: Model 2 moderate AI signal + suspicious patterns (lowered thresholds)
        elif model2.fake_score > 0.18 and suspicious_score > 0.4:  # Lowered thresholds
            classification = 'FAKE (AI Detected + Suspicious Patterns)'
            base_confidence = max(model2.fake_score * 1.3, suspicious_score * 0.85)  # Increased multipliers
            base_confidence = min(0.92, base_confidence)  # Increased max
            reasoning = f"AI detector suggests generated content ({model2.fake_score:.2%}) combined with suspicious patterns ({suspicious_score:.2f})."
        
        # PRIORITY 2.5: Even lower AI signal + high suspicious patterns
        elif model2.fake_score > 0.12 and suspicious_score > 0.55:  # New priority level
            classification = 'FAKE (AI Signal + High Suspicious Patterns)'
            base_confidence = max(model2.fake_score * 1.5, suspicious_score * 0.9)
            base_confidence = min(0.9, base_confidence)
            reasoning = f"AI detector shows signal ({model2.fake_score:.2%}) combined with very high suspicious patterns ({suspicious_score:.2f})."
        
        # PRIORITY 3: High suspicious patterns BUT only if Model 2 doesn't strongly suggest REAL
        # Model 2 (AI Detector) is the authority on human vs. AI origin
        # BUT: If patterns are VERY suspicious (generic + lacks details), trust patterns over Model 2
        elif suspicious_score > 0.6 and model2.fake_score > 0.12 and real_score > 0.7:  # Lowered model2.fake_score threshold
            # Check if this is a very generic pattern (ChatGPT-like)
            is_very_generic = 'generic_language' in patterns and 'lack_of_details' in patterns
            
            if is_very_generic and suspicious_score > 0.65:
                # Very generic pattern = likely ChatGPT, even if Model 2 is uncertain
                classification = 'FAKE (Very Generic Pattern - Likely ChatGPT)'
                base_confidence = max(0.75, suspicious_score * 0.9)
                reasoning = f"Very generic language pattern ({suspicious_score:.2f}) suggests AI-generated content (ChatGPT). Model 2 uncertainty ({model2.real_score:.2%}) overridden by strong pattern evidence."
            elif model2.real_score > 0.8:  # Increased from 0.75 - need VERY high confidence
                classification = 'REAL'
                base_confidence = real_score
                reasoning = f"Natural Hebrew phrasing with authentic personal experience. AI detector strongly indicates human-written content ({model2.real_score:.2%}) despite some suspicious patterns. Origin is authentic."
            else:
                # Even if Model 2 suggests human, high suspicious patterns + any AI signal = FAKE
                classification = 'FAKE (Suspicious Patterns + AI Signal)'
                base_confidence = min(0.85, max(suspicious_score * 0.95, fake_score, model2.fake_score * 1.3))  # Increased multipliers
                reasoning = f"Very suspicious patterns ({suspicious_score:.2f}) detected with AI signal ({model2.fake_score:.2%}). Likely AI-generated despite Model 2 uncertainty."
        
        # PRIORITY 4: Moderate suspicious patterns + Model 2 AI signal (lowered thresholds)
        elif suspicious_score > self.thresholds.suspicious_patterns_threshold and model2.fake_score > 0.15:  # Lowered from 0.2
            # Model 2 is the authority on origin - if it says human VERY confidently, trust it
            if model2.real_score > 0.8:  # Increased from 0.7 to 0.8
                classification = 'REAL'
                base_confidence = model2.real_score
                reasoning = f"AI detector strongly indicates human-written content ({model2.real_score:.2%}) despite suspicious patterns ({suspicious_score:.2f}). Origin is authentic."
            elif fake_score > self.thresholds.fake_classification_threshold:
                classification = 'FAKE (Suspicious Patterns + AI Signal)'
                base_confidence = max(fake_score, model2.fake_score * 1.1, suspicious_score * 0.7)  # Increased multipliers
                reasoning = f"Suspicious patterns ({suspicious_score:.2f}) and AI detector signal ({model2.fake_score:.2%}) indicate non-human origin."
            else:
                classification = 'FAKE (Suspicious Patterns + AI Signal)'
                base_confidence = max(0.65, (suspicious_score * 0.6 + model2.fake_score * 0.4) * 1.2)  # Increased calculation
                reasoning = f"Suspicious patterns ({suspicious_score:.2f}) and AI detector signal ({model2.fake_score:.2%}) suggest non-human origin."
        
        # PRIORITY 5: High suspicious patterns alone (but only if Model 2 doesn't strongly disagree)
        elif suspicious_score > 0.55 and model2.real_score < 0.85:
            # Model 2 is the authority - if it suggests human, trust it
            if model2.real_score > 0.65:
                classification = 'REAL'
                base_confidence = model2.real_score
                reasoning = f"AI detector indicates human-written content ({model2.real_score:.2%}) despite suspicious patterns ({suspicious_score:.2f}). Origin is authentic."
            elif fake_score > self.thresholds.fake_classification_threshold:
                classification = 'FAKE (Suspicious Patterns)'
                base_confidence = fake_score
                reasoning = f"Suspicious patterns ({suspicious_score:.2f}) combined with model scores indicate non-human origin."
            else:
                classification = 'FAKE (Suspicious Patterns)'
                base_confidence = max(0.55, suspicious_score * 0.75)
                reasoning = f"Suspicious patterns ({suspicious_score:.2f}) suggest non-human origin."
        
        # PRIORITY 6: Ensemble scores favor FAKE
        elif fake_score > real_score:
            classification = 'FAKE'
            base_confidence = fake_score
            reasoning = f"Ensemble scores favor FAKE ({fake_score:.2%} vs {real_score:.2%})."
        
        # PRIORITY 7: Moderate indicators
        elif fake_score > 0.3 and suspicious_score > 0.4 and model2.fake_score > 0.15:
            # Check Model 2 first - it's the authority on origin
            if model2.real_score > 0.65:
                classification = 'REAL'
                base_confidence = model2.real_score
                reasoning = f"AI detector indicates human-written content ({model2.real_score:.2%}). Origin is authentic."
            else:
                classification = 'FAKE (Multiple Indicators)'
                base_confidence = min(0.75, fake_score + (suspicious_score * 0.2))
                reasoning = f"Multiple indicators suggest non-human origin: fake score ({fake_score:.2%}), suspicious patterns ({suspicious_score:.2%}), AI signal ({model2.fake_score:.2%})."
        
        # DEFAULT: REAL (only if no strong indicators of FAKE)
        # Model 2 is the final authority - if it says human, it's REAL
        else:
            if model2.real_score > 0.6:
                classification = 'REAL'
                base_confidence = real_score
                reasoning = f"AI detector indicates human-written content ({model2.real_score:.2%}). Ensemble scores favor REAL ({real_score:.2%} vs {fake_score:.2%})."
            else:
                classification = 'REAL'
                base_confidence = real_score
                reasoning = f"Ensemble scores favor REAL ({real_score:.2%} vs {fake_score:.2%})."
        
        calibrated_confidence = self._calibrate_confidence(
            base_confidence,
            model1,
            model2,
            suspicious_score,
            fake_score,
            real_score
        )
        
        calibrated_confidence = max(0.0, min(1.0, float(calibrated_confidence)))
        
        if calibrated_confidence < base_confidence * 0.8:
            reasoning += f" Confidence calibrated down from {base_confidence:.2%} to {calibrated_confidence:.2%} due to model disagreement and suspicious patterns."
        
        return classification, calibrated_confidence, reasoning
    
    def _calculate_fake_probability(
        self,
        adjusted_scores: Tuple[float, float],
        classification: str,
        suspicious_score: float,
        calibrated_confidence: float
    ) -> float:
        """
        Calculate final fake probability with confidence calibration.
        """
        fake_score, real_score = adjusted_scores
        
        if 'FAKE' in classification:
            if suspicious_score > 0.6:
                base_prob = max(fake_score, 0.65)
                if calibrated_confidence < 0.7:
                    return min(0.85, base_prob)
                return min(1.0, base_prob)
            if calibrated_confidence < fake_score * 0.8:
                return fake_score * 0.9
            return fake_score
        else:
            base_real_prob = real_score
            if calibrated_confidence < base_real_prob * 0.8:
                uncertainty_boost = (base_real_prob - calibrated_confidence) * 0.5
                fake_prob = 1.0 - base_real_prob + uncertainty_boost
                return min(0.5, fake_prob)
            return 1.0 - real_score
    
    def _determine_model_used(
        self,
        model1: ModelOutput,
        model2: ModelOutput,
        suspicious_score: float
    ) -> str:
        """Determine which model(s) were primarily used."""
        if suspicious_score > 0.6:
            return 'Suspicious Patterns (Override)'
        elif model2.fake_score > self.thresholds.ai_detection_threshold:
            return 'AI Detector (Model 2)'
        elif model1.confidence > 0.9:
            return 'Review Classifier (Model 1)'
        else:
            return 'Ensemble (Model 1 + Model 2)'

