"""
Bias and Fraud Detection Module.

This module implements specialized detection for:
1. Paid/Fraudulent reviews (Model A: T5-Base_Fraud_Detection)

Note: Bias detector (autism-bias-detection-roberta) was removed as it's not relevant
for commercial review detection - it's designed for autism-related bias detection.

The module is triggered when AI detection returns UNCERTAIN or high-confidence REAL,
to catch human-written but fraudulent content.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ml_config import SystemConfig, get_config
from ensemble_classifier import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class BiasFraudResult:
    """Result from bias/fraud detection."""
    classification: str  # 'FRAUD (PAID/DECEPTIVE)', 'HIGHLY BIASED (Non-Objective)', or None
    score: float
    model_used: str
    confidence: float


class BiasFraudDetector:
    """
    Detects fraudulent and biased reviews that may not be caught by AI detection.
    
    This module is specifically designed to catch:
    - Paid endorsements
    - Deceptive commercial text
    - Bot-posted spam content
    - Non-objective, biased language
    """
    
    def __init__(self, config: SystemConfig = None):
        """
        Initialize the bias/fraud detector.
        
        Args:
            config: System configuration. If None, uses global config.
        """
        self.config = config or get_config()
        self.thresholds = self.config.thresholds
        self.fraud_model = None
        self.bias_model = None
    
    def load_models(self, fraud_pipeline, bias_pipeline=None):
        """
        Load the fraud detection model.
        
        Args:
            fraud_pipeline: Loaded fraud detection pipeline
            bias_pipeline: Not used (kept for backward compatibility)
        """
        self.fraud_model = fraud_pipeline
        self.bias_model = None  # Removed - not relevant for review detection
        logger.info("✅ Fraud detection model loaded")
    
    def should_run_bias_fraud_check(
        self,
        model1_output: ModelOutput,
        model2_output: ModelOutput
    ) -> bool:
        """
        Determine if bias/fraud check should be executed.
        
        Trigger conditions:
        - AI detection returns UNCERTAIN (score < 0.60)
        - AI detection returns high-confidence REAL (score > 0.90)
        
        Args:
            model1_output: Output from review classifier
            model2_output: Output from AI detector
            
        Returns:
            True if bias/fraud check should run
        """
        # Check if Model 2 (AI detector) is uncertain
        if model2_output.confidence < self.thresholds.low_confidence:
            logger.info("Triggering bias/fraud check: AI detection uncertain")
            return True
        
        # Check if Model 1 or Model 2 shows high confidence in REAL
        if (model1_output.real_score > self.thresholds.high_confidence or
            model2_output.real_score > self.thresholds.high_confidence):
            logger.info("Triggering bias/fraud check: High confidence in REAL")
            return True
        
        return False
    
    def detect_fraud(self, text: str) -> Optional[BiasFraudResult]:
        """
        Run Model A: Fraud/Paid Review Detection.
        
        Args:
            text: Translated English text to analyze
            
        Returns:
            BiasFraudResult if fraud detected, None otherwise
        """
        if not self.fraud_model:
            logger.warning("Fraud detection model not loaded")
            return None
        
        try:
            # Run fraud detection model
            results = self.fraud_model(text)
            
            # Parse results - T5 models may return different formats
            fraud_score = self._parse_fraud_results(results)
            
            # CRITICAL: Clamp fraud_score to [0.0, 1.0] range
            fraud_score = max(0.0, min(1.0, float(fraud_score)))
            
            if fraud_score >= self.thresholds.fraud_detection_threshold:
                logger.info(f"🚨 FRAUD DETECTED: Score {fraud_score:.4f}")
                return BiasFraudResult(
                    classification='FRAUD (PAID/DECEPTIVE)',
                    score=fraud_score,
                    model_used='T5-Base_Fraud_Detection',
                    confidence=fraud_score
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fraud detection: {e}", exc_info=True)
            return None
    
    def detect_bias(self, text: str) -> Optional[BiasFraudResult]:
        """
        Run Model B: Bias/Non-Objective Detection.
        
        Args:
            text: Translated English text to analyze
            
        Returns:
            BiasFraudResult if bias detected, None otherwise
        """
        if not self.bias_model:
            logger.warning("Bias detection model not loaded")
            return None
        
        try:
            # Run bias detection model
            results = self.bias_model(text)
            
            # Parse results - adapt autism-bias model output for non-objective detection
            bias_score = self._parse_bias_results(results)
            
            # CRITICAL: Clamp bias_score to [0.0, 1.0] range
            bias_score = max(0.0, min(1.0, float(bias_score)))
            
            if bias_score >= self.thresholds.bias_detection_threshold:
                logger.info(f"⚠️ BIAS DETECTED: Score {bias_score:.4f}")
                return BiasFraudResult(
                    classification='HIGHLY BIASED (Non-Objective)',
                    score=bias_score,
                    model_used='autism-bias-detection-roberta',
                    confidence=bias_score
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in bias detection: {e}", exc_info=True)
            return None
    
    def check(self, text: str, model1_output: ModelOutput, model2_output: ModelOutput) -> Optional[BiasFraudResult]:
        """
        Execute the full bias/fraud check pipeline.
        
        Logic:
        1. Check if bias/fraud check should run
        2. Run Model A (Fraud Detection)
        3. If no fraud, run Model B (Bias Detection)
        
        Args:
            text: Translated English text
            model1_output: Output from review classifier
            model2_output: Output from AI detector
            
        Returns:
            BiasFraudResult if fraud or bias detected, None otherwise
        """
        # Check if we should run the bias/fraud check
        if not self.should_run_bias_fraud_check(model1_output, model2_output):
            return None
        
        logger.info("🔍 Running Bias/Fraud Detection Module...")
        
        # Step 1: Check for Fraud (Model A)
        fraud_result = self.detect_fraud(text)
        if fraud_result:
            return fraud_result
        
        # Step 2: Bias detection removed - not relevant for commercial reviews
        # The autism-bias-detection-roberta model was designed for autism-related bias,
        # not for detecting biased commercial reviews
        
        logger.info("✅ No fraud detected")
        return None
    
    def _parse_fraud_results(self, results: Any) -> float:
        """
        Parse results from T5-Base_Fraud_Detection model.
        
        The model may return different formats, so we handle multiple cases.
        """
        fraud_score = 0.0
        
        # Handle different result formats
        if isinstance(results, list):
            # List of results
            for result in results:
                if isinstance(result, dict):
                    label = result.get('label', '').upper()
                    score = float(result.get('score', 0.0))
                    
                    # Look for fraud-related labels
                    if any(x in label for x in ['FRAUD', 'DECEPTIVE', 'PAID', 'FAKE']):
                        fraud_score = max(fraud_score, score)
                    elif 'REAL' in label or 'LEGITIMATE' in label:
                        # If it's a binary classifier, fraud = 1 - real
                        fraud_score = max(fraud_score, 1.0 - score)
                elif isinstance(result, list):
                    # Nested list
                    for sub_result in result:
                        if isinstance(sub_result, dict):
                            label = sub_result.get('label', '').upper()
                            score = float(sub_result.get('score', 0.0))
                            if any(x in label for x in ['FRAUD', 'DECEPTIVE', 'PAID']):
                                fraud_score = max(fraud_score, score)
        
        elif isinstance(results, dict):
            # Single result dictionary
            label = results.get('label', '').upper()
            score = float(results.get('score', 0.0))
            if any(x in label for x in ['FRAUD', 'DECEPTIVE', 'PAID']):
                fraud_score = score
        
        # If no fraud label found, check if it's a binary classifier
        # and use the higher score as fraud probability
        if fraud_score == 0.0 and isinstance(results, list):
            all_scores = []
            for result in results:
                if isinstance(result, dict):
                    all_scores.append(float(result.get('score', 0.0)))
                elif isinstance(result, list):
                    for sub_result in result:
                        if isinstance(sub_result, dict):
                            all_scores.append(float(sub_result.get('score', 0.0)))
            
            if all_scores:
                # Use the maximum score as fraud probability
                fraud_score = max(all_scores)
        
        return fraud_score
    
    def _parse_bias_results(self, results: Any) -> float:
        """
        Parse results from autism-bias-detection-roberta model.
        
        We adapt this model's output for non-objective language detection.
        """
        bias_score = 0.0
        
        # Handle different result formats
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    label = result.get('label', '').upper()
                    score = float(result.get('score', 0.0))
                    
                    # Look for bias-related labels
                    # The model may return labels like 'BIASED', 'AUTISM', 'NEUROTYPICAL', etc.
                    # We interpret high scores for bias-related labels as non-objective
                    if any(x in label for x in ['BIAS', 'BIASED', 'AUTISM', 'NON-OBJECTIVE']):
                        bias_score = max(bias_score, score)
                    # If it's a binary classifier, bias might be the inverse
                    elif 'NEUROTYPICAL' in label or 'OBJECTIVE' in label:
                        bias_score = max(bias_score, 1.0 - score)
                elif isinstance(result, list):
                    for sub_result in result:
                        if isinstance(sub_result, dict):
                            label = sub_result.get('label', '').upper()
                            score = float(sub_result.get('score', 0.0))
                            if any(x in label for x in ['BIAS', 'BIASED', 'NON-OBJECTIVE']):
                                bias_score = max(bias_score, score)
        
        elif isinstance(results, dict):
            label = results.get('label', '').upper()
            score = float(results.get('score', 0.0))
            if any(x in label for x in ['BIAS', 'BIASED', 'NON-OBJECTIVE']):
                bias_score = score
        
        # If no clear bias label, use maximum score as bias probability
        if bias_score == 0.0 and isinstance(results, list):
            all_scores = []
            for result in results:
                if isinstance(result, dict):
                    all_scores.append(float(result.get('score', 0.0)))
                elif isinstance(result, list):
                    for sub_result in result:
                        if isinstance(sub_result, dict):
                            all_scores.append(float(sub_result.get('score', 0.0)))
            
            if all_scores:
                bias_score = max(all_scores)
        
        return bias_score

