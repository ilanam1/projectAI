"""
Configuration module for Fake Review Detection System.

This module provides centralized, environment-aware configuration management
for all ML models, thresholds, and system parameters.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a single ML model."""
    name: str
    model_id: str
    device: int = -1  # -1 for CPU, 0+ for GPU
    return_all_scores: bool = True
    timeout: float = 30.0  # seconds


@dataclass
class EnsembleWeights:
    """Weights for ensemble model combination."""
    model1_weight: float = 0.5
    model2_weight: float = 0.5
    suspicious_patterns_weight: float = 0.3


@dataclass
class Thresholds:
    """Classification thresholds for decision making."""
    high_confidence: float = 0.90
    low_confidence: float = 0.60
    ai_detection_threshold: float = 0.25  # Lowered to catch more AI-generated content
    suspicious_patterns_threshold: float = 0.4
    fake_classification_threshold: float = 0.5
    # Confidence calibration thresholds
    overconfidence_penalty_threshold: float = 0.95  # Penalize confidence above this
    model_disagreement_threshold: float = 0.3  # Significant disagreement threshold
    uncertainty_boost_threshold: float = 0.2  # Boost uncertainty when below this
    # Bias/Fraud detection thresholds
    fraud_detection_threshold: float = 0.95  # Threshold for fraud detection
    bias_detection_threshold: float = 0.90  # Threshold for bias detection


@dataclass
class SuspiciousPatternsConfig:
    """Configuration for suspicious pattern detection."""
    excessive_positive_min_count: int = 2
    generic_phrases_min_count: int = 1  # Lowered to 1 - even 1 generic phrase is suspicious
    short_review_threshold: int = 100
    
    positive_words: List[str] = None
    generic_phrases: List[str] = None
    
    def __post_init__(self):
        if self.positive_words is None:
            self.positive_words = [
                'perfect', 'amazing', 'excellent', 'wonderful', 'fantastic',
                'great', 'love', 'best', 'awesome', 'outstanding', 'incredible',
                'simply perfect', 'very good', 'works well', 'works smoothly'
            ]
        
        if self.generic_phrases is None:
            self.generic_phrases = [
                'very good', 'works well', 'no problems', 'looks good',
                'arrived quickly', 'arrived very quickly', 'highly recommend',
                'worth it', 'simply perfect', 'works smoothly', 'had no problems',
                'no issues', 'works perfectly', 'very fast', 'very quickly',
                # Additional AI-generated patterns
                'exactly as described', 'as expected', 'meets expectations',
                'would recommend', 'definitely recommend', 'highly recommended',
                'great product', 'good quality', 'fast shipping', 'quick delivery',
                'satisfied with', 'happy with', 'pleased with', 'exceeded expectations',
                'better than expected', 'worth the money', 'good value', 'great value',
                # Hebrew generic phrases (common in AI-generated reviews)
                'מוצר משוגע', 'כל הצוות', 'עף על זה', 'באמת מוצר', 'מוצר מעולה',
                'ממליץ בחום', 'שווה את הכסף', 'איכות מעולה', 'מאוד מרוצה',
                'עובד מצוין', 'נראה טוב', 'אין בעיות', 'מושלם', 'מעולה',
                'כל כך טוב', 'אהבתי מאוד', 'מומלץ מאוד', 'בהחלט ממליץ',
                # Additional Hebrew AI-generated patterns
                'חייב כל בית', 'חייב בכל בית', 'לא יודע איך הסתדרתי',
                'פתר לי בעיה', 'הציקה לי', 'הציק לי', 'לא יודע איך',
                'חייב להיות', 'חייב להיות בכל בית', 'פתר בעיה',
                'עד היום', 'עד עכשיו', 'לא ידעתי', 'לא האמנתי',
                'משנה חיים', 'שינה לי את החיים', 'הכי טוב', 'הטוב ביותר'
            ]


@dataclass
class SystemConfig:
    """Main system configuration."""
    # Model configurations
    review_classifier: ModelConfig
    ai_detector: ModelConfig
    hebrew_ai_detector: Optional[ModelConfig] = None  # Hebrew AI detector (runs on original text before translation)
    fraud_detector: Optional[ModelConfig] = None  # Model A: Paid/Fraud detection
    bias_detector: Optional[ModelConfig] = None  # Model B: Bias/Non-objective detection
    translation_enabled: bool = True
    translation_source_lang: str = 'iw'  # Hebrew
    translation_target_lang: str = 'en'
    
    # Ensemble configuration
    ensemble_weights: EnsembleWeights = None
    thresholds: Thresholds = None
    suspicious_patterns: SuspiciousPatternsConfig = None
    
    # Performance settings
    enable_caching: bool = True
    max_cache_size: int = 1000
    request_timeout: float = 60.0
    
    # Logging
    verbose_logging: bool = True
    log_model_outputs: bool = True
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = EnsembleWeights()
        if self.thresholds is None:
            self.thresholds = Thresholds()
        if self.suspicious_patterns is None:
            self.suspicious_patterns = SuspiciousPatternsConfig()


def load_config_from_env() -> SystemConfig:
    """
    Load configuration from environment variables with sensible defaults.
    
    Environment variables:
    - ML_MODEL_DEVICE: GPU device ID (-1 for CPU)
    - ML_VERBOSE_LOGGING: Enable verbose logging (true/false)
    - ML_TRANSLATION_ENABLED: Enable translation (true/false)
    - ML_AI_DETECTION_THRESHOLD: AI detection threshold (0.0-1.0)
    - ML_SUSPICIOUS_PATTERNS_THRESHOLD: Suspicious patterns threshold (0.0-1.0)
    """
    import torch
    
    device = int(os.getenv('ML_MODEL_DEVICE', '-1'))
    if device == -1:
        device = 0 if torch.cuda.is_available() else -1
    
    verbose = os.getenv('ML_VERBOSE_LOGGING', 'true').lower() == 'true'
    translation_enabled = os.getenv('ML_TRANSLATION_ENABLED', 'true').lower() == 'true'
    
    ai_threshold = float(os.getenv('ML_AI_DETECTION_THRESHOLD', '0.40'))
    suspicious_threshold = float(os.getenv('ML_SUSPICIOUS_PATTERNS_THRESHOLD', '0.5'))
    
    thresholds = Thresholds(
        ai_detection_threshold=ai_threshold,
        suspicious_patterns_threshold=suspicious_threshold
    )
    
    return SystemConfig(
        review_classifier=ModelConfig(
            name="Review Classifier",
            model_id="debojit01/fake-review-detector",
            device=device
        ),
        ai_detector=ModelConfig(
            name="AI Detector",
            model_id="roberta-base-openai-detector",
            device=device
        ),
        # Hebrew AI detector - runs on original Hebrew text before translation
        # CRITICAL: This runs BEFORE translation to catch Hebrew-specific AI patterns
        # Using multilingual model - xlm-roberta-base supports Hebrew
        # Note: Pattern detection is still the primary Hebrew detection method
        # This model provides additional signal on original Hebrew text
        hebrew_ai_detector=None,  # Disabled for now - pattern detection handles Hebrew patterns
        # If you find a Hebrew-specific AI detection model, uncomment and set:
        # hebrew_ai_detector=ModelConfig(
        #     name="Hebrew AI Detector",
        #     model_id="your-hebrew-ai-detector-model-id",
        #     device=device
        # ),
        fraud_detector=ModelConfig(
            name="Fraud Detector",
            model_id="text-fraud-detection/T5-Base_Fraud_Detection",
            device=device
        ),
        # Bias detector removed - autism-bias-detection-roberta is not relevant for review detection
        bias_detector=None,
        translation_enabled=translation_enabled,
        thresholds=thresholds,
        verbose_logging=verbose
    )


# Global configuration instance
_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config_from_env()
    return _config


def set_config(config: SystemConfig):
    """Set the global configuration instance (for testing)."""
    global _config
    _config = config

