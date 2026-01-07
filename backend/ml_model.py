"""
Production-Grade Fake Review Detection System.

This module implements a robust, scalable ensemble pipeline for classifying reviews
as REAL or FAKE using multiple ML models and pattern detection.

Architecture:
- Modular design with separated concerns
- Configurable thresholds and weights
- Extensible pattern detection
- Production-ready error handling
- Comprehensive logging
"""

import logging
from typing import Optional, Dict, Any, Tuple
import csv
import os
from pathlib import Path
from urllib.parse import urlparse
import tempfile
import requests
from collections import Counter
import math
import re
import string
import joblib
from sklearn.pipeline import Pipeline 

try:
    from transformers import pipeline
    import torch
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    import random

USE_GOOGLETRANS = False
try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    try:
        from googletrans import Translator
        TRANSLATION_AVAILABLE = True
        USE_GOOGLETRANS = True
    except ImportError:
        TRANSLATION_AVAILABLE = False

try:
    from ml_config import get_config, SystemConfig
    from pattern_detector import SuspiciousPatternDetector
    from ensemble_classifier import EnsembleClassifier, ModelOutput, ClassificationResult
    from bias_fraud_detector import BiasFraudDetector
    ARCHITECTURE_AVAILABLE = True
except ImportError:
    ARCHITECTURE_AVAILABLE = False
    get_config = None
    BiasFraudDetector = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model_instances: Dict[str, Any] = {}
_config_instance: Optional[SystemConfig] = None
# Cached dataset for heuristic lookup
_dataset_cache: Dict[str, Any] = {}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra spaces; keep Hebrew characters intact."""
    if not text or not isinstance(text, str):
        return ""
    lowered = text.lower()
    cleaned = re.sub(r"[\"'“”’‘׳״.,!?;:()\\[\\]{}\\-]+", " ", lowered)
    cleaned = re.sub(r"\\s+", " ", cleaned).strip()
    return cleaned


def _tokenize(text: str) -> list:
    """
    Tokenizer: normalized lowercase split, drop very short tokens.
    Returns list to preserve counts for cosine similarity.
    """
    if not text or not isinstance(text, str):
        return []
    words = _normalize(text).split()
    meaningful_words = [w for w in words if len(w) > 2]
    return meaningful_words


def _char_ngrams(text: str, n: int = 3) -> Counter:
    """Character n-grams for robust matching (helps Hebrew AI phrasing)."""
    txt = _normalize(text)
    if len(txt) < n:
        return Counter()
    return Counter(txt[i : i + n] for i in range(len(txt) - n + 1))


def _jaccard_similarity(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _predict_from_dataset(text: str, dataset_reviews) -> Optional[Dict[str, Any]]:
    """
    Heuristic prediction using the labeled dataset.
    Finds the most similar review via Jaccard similarity and uses its label.
    Lowered threshold to catch more matches - even partial similarity is useful.
    """
    if not dataset_reviews:
        return None
    tokens = set(_tokenize(text))
    best_sim = 0.0
    best_label = None
    
    # Find best match
    for sample_text, sample_label in dataset_reviews:
        sim = _jaccard_similarity(tokens, set(_tokenize(sample_text)))
        if sim > best_sim:
            best_sim = sim
            best_label = sample_label
    
    # Lowered threshold from 0.55 to 0.20 to catch more matches
    # Even partial similarity to labeled data is valuable, especially for Hebrew reviews
    if best_label and best_sim >= 0.20:
        # Calculate fake probability based on similarity and label
        if best_label == "FAKE":
            # Higher similarity = higher fake probability
            # Scale: 0.20 similarity -> 0.60 fake_prob, 0.50 similarity -> 0.75 fake_prob, 0.80+ similarity -> 0.90+ fake_prob
            fake_prob = min(0.95, 0.55 + (best_sim * 0.5))  # Range: 0.55 to 0.95
        else:  # REAL
            # Higher similarity = lower fake probability
            # Scale: 0.20 similarity -> 0.45 fake_prob, 0.50 similarity -> 0.25 fake_prob, 0.80+ similarity -> 0.10 fake_prob
            fake_prob = max(0.05, 0.5 - (best_sim * 0.5))  # Range: 0.05 to 0.5
        
        # Confidence scales with similarity, but boost it for dataset matches
        # Dataset matches are valuable even at lower similarity
        confidence = min(0.95, 0.4 + (best_sim * 0.7))  # Range: 0.4 to 0.95
        
        logger.info(f"Dataset match found: similarity={best_sim:.3f}, label={best_label}, fake_prob={fake_prob:.3f}, confidence={confidence:.3f}")
        
        return {
            "classification": "FAKE" if best_label == "FAKE" else "REAL",
            "fake_probability": fake_prob,
            "confidence": confidence,
            "reasoning": f"Dataset match (similarity {best_sim:.2f}) to labeled {best_label} review. Similarity indicates {best_label} classification."
        }
    
    if best_sim > 0.1:  # Log even weak matches for debugging
        logger.debug(f"Dataset weak match: similarity={best_sim:.3f}, label={best_label} (below threshold 0.20)")
    
    return None


class DatasetMatcher:
    """
    Robust matcher over the labeled dataset:
    - Token cosine
    - Character trigram cosine (handles Hebrew phrasing/spacing)
    Returns highest similarity as confidence.
    """
    def __init__(self, dataset_reviews):
        self.samples = []
        for text, label in dataset_reviews:
            tokens = _tokenize(text)
            char_counts = _char_ngrams(text, n=3)
            if not tokens and not char_counts:
                continue
            tok_counts = Counter(tokens)
            tok_norm = math.sqrt(sum(v * v for v in tok_counts.values())) if tok_counts else 0.0
            char_norm = math.sqrt(sum(v * v for v in char_counts.values())) if char_counts else 0.0
            if tok_norm == 0 and char_norm == 0:
                continue
            self.samples.append((tok_counts, tok_norm, char_counts, char_norm, label))

    def predict(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.samples or not text or not isinstance(text, str):
            return None

        tokens = _tokenize(text)
        char_counts = _char_ngrams(text, n=3)
        if not tokens and not char_counts:
            return None
        tok_counts = Counter(tokens)
        tok_norm = math.sqrt(sum(v * v for v in tok_counts.values())) if tok_counts else 0.0
        char_norm = math.sqrt(sum(v * v for v in char_counts.values())) if char_counts else 0.0
        if tok_norm == 0 and char_norm == 0:
            return None

        best_sim = 0.0
        best_label = None

        for s_tok_counts, s_tok_norm, s_char_counts, s_char_norm, label in self.samples:
            # Token cosine
            tok_dot = sum(tok_counts[t] * s_tok_counts.get(t, 0) for t in tok_counts)
            tok_sim = tok_dot / (tok_norm * s_tok_norm) if tok_norm > 0 and s_tok_norm > 0 else 0.0
            # Char trigram cosine
            char_dot = sum(char_counts[t] * s_char_counts.get(t, 0) for t in char_counts)
            char_sim = char_dot / (char_norm * s_char_norm) if char_norm > 0 and s_char_norm > 0 else 0.0

            sim = max(tok_sim, char_sim)
            if sim > best_sim:
                best_sim = sim
                best_label = label

        if best_label is None:
            return None

        classification = "FAKE" if best_label.upper() == "FAKE" else "REAL"
        confidence = max(0.0, min(1.0, best_sim))

        # Fake probability derived from label and similarity
        if classification == "FAKE":
            fake_prob = max(0.5, confidence)
        else:
            fake_prob = 1.0 - confidence

        return {
            "classification": classification,
            "fake_probability": fake_prob,
            "confidence": confidence,
            "reasoning": f"Dataset cosine/char-gram match (sim={confidence:.2f}) to labeled {classification} review."
        }


class ModelManager:
    """
    Manages ML model lifecycle: loading, caching, and inference.
    
    This class implements a singleton pattern to ensure models are loaded
    only once and reused across all requests.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.review_classifier = None
        self.ai_detector = None
        self.hebrew_ai_detector = None  # Hebrew AI detector (runs on original text)
        self.fraud_detector = None
        self.bias_detector = None
        self.translator_initialized = False
        self._dataset_matcher = None
        self.local_hebrew_model = None  
        self._load_models()   

    # ---------------- Dataset cache ----------------
    @property
    def dataset_reviews(self):
        """
        Load and cache dataset from CSV with columns: text,label
        File path configured via DATASET_PATH env or default under user's Downloads.
        """
        if "reviews" in _dataset_cache:
            return _dataset_cache["reviews"]

        # Determine source: local path or URL
        base_dir = Path(__file__).resolve().parent   # backend/
        default_dataset_path = base_dir / "data" / "combined_reviews_with_labels.csv"

        dataset_path_str = os.getenv("DATASET_PATH", str(default_dataset_path))
        dataset_path = Path(dataset_path_str)
        dataset_url = os.getenv("DATASET_URL", "").strip()

        path = Path(dataset_path)

        # If local file missing and URL provided, download to temp
        if (not path.exists() or path.is_dir()) and dataset_url:
            try:
                logger.info(f"Downloading dataset from URL: {dataset_url}")
                resp = requests.get(dataset_url, timeout=10)
                resp.raise_for_status()
                tmp_fd, tmp_path = tempfile.mkstemp(prefix="reviews_ds_", suffix=".csv")
                os.close(tmp_fd)
                with open(tmp_path, "wb") as f:
                    f.write(resp.content)
                path = Path(tmp_path)
                logger.info(f"Dataset downloaded to {path}")
            except Exception as e:
                logger.warning(f"Failed to download dataset from {dataset_url}: {e}")
                _dataset_cache["reviews"] = []
                return _dataset_cache["reviews"]

        if not path.exists() or path.is_dir():
            logger.warning(f"Dataset not found at {path}. Skipping dataset-based prediction.")
            _dataset_cache["reviews"] = []
            return _dataset_cache["reviews"]

        reviews = []
        exact_map = {}
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Support both "text" and "review_text" column names
                    text = (row.get("text") or row.get("review_text") or "").strip()
                    label = (row.get("label") or "").strip().upper()
                    if not text or label not in {"REAL", "FAKE"}:
                        continue
                    reviews.append((text, label))
                    norm_text = _normalize(text)
                    if norm_text:
                        exact_map.setdefault(norm_text, label)
            logger.info(f"Loaded {len(reviews)} labeled reviews from dataset")
        except Exception as e:
            logger.warning(f"Failed to load dataset from {path}: {e}")
            reviews = []
            exact_map = {}

        _dataset_cache["reviews"] = reviews
        # Build matcher cache for high-confidence cosine matching
        _dataset_cache["matcher"] = DatasetMatcher(reviews) if reviews else None
        _dataset_cache["exact_map"] = exact_map
        return reviews

    def get_dataset_matcher(self):
        """
        Return cached DatasetMatcher (built on first dataset load).
        """
        if self._dataset_matcher is not None:
            return self._dataset_matcher
        if "matcher" in _dataset_cache:
            self._dataset_matcher = _dataset_cache["matcher"]
            return self._dataset_matcher
        # Trigger dataset load (builds matcher)
        _ = self.dataset_reviews
        self._dataset_matcher = _dataset_cache.get("matcher")
        return self._dataset_matcher

    def get_exact_map(self):
        """Return normalized-text -> label map for exact/near-exact matches."""
        if "exact_map" in _dataset_cache:
            return _dataset_cache["exact_map"]
        _ = self.dataset_reviews
        return _dataset_cache.get("exact_map", {})
    
    def _load_models(self):
        """Load all required ML models."""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not installed. Using placeholder mode.")
            return
        
        try:
            # Load Review Classifier (Model 1)
            if self.config.review_classifier:
                logger.info(f"Loading {self.config.review_classifier.name}...")
                self.review_classifier = pipeline(
                    "text-classification",
                    model=self.config.review_classifier.model_id,
                    device=self.config.review_classifier.device,
                    return_all_scores=self.config.review_classifier.return_all_scores
                )
                logger.info(f" {self.config.review_classifier.name} loaded successfully")
            
            # Load Hebrew AI Detector (runs on original Hebrew text before translation)
            if self.config.hebrew_ai_detector:
                try:
                    logger.info(f"Loading {self.config.hebrew_ai_detector.name}...")
                    self.hebrew_ai_detector = pipeline(
                        "text-classification",
                        model=self.config.hebrew_ai_detector.model_id,
                        device=self.config.hebrew_ai_detector.device,
                        return_all_scores=self.config.hebrew_ai_detector.return_all_scores
                    )
                    logger.info(f" {self.config.hebrew_ai_detector.name} loaded successfully")
                except Exception as e:
                    logger.warning(f" Could not load Hebrew AI detector: {e}. Continuing without it.")
                    self.hebrew_ai_detector = None
            
            # Load AI Detector (Model 2) - English
            if self.config.ai_detector:
                logger.info(f"Loading {self.config.ai_detector.name}...")
                self.ai_detector = pipeline(
                    "text-classification",
                    model=self.config.ai_detector.model_id,
                    device=self.config.ai_detector.device,
                    return_all_scores=self.config.ai_detector.return_all_scores
                )
                logger.info(f" {self.config.ai_detector.name} loaded successfully")
            
            # Load Fraud Detector (Model A) - Optional
            if self.config.fraud_detector:
                try:
                    logger.info(f"Loading {self.config.fraud_detector.name}...")
                    self.fraud_detector = pipeline(
                        "text-classification",
                        model=self.config.fraud_detector.model_id,
                        device=self.config.fraud_detector.device,
                        return_all_scores=self.config.fraud_detector.return_all_scores
                    )
                    logger.info(f" {self.config.fraud_detector.name} loaded successfully")
                except Exception as e:
                    logger.warning(f" Could not load fraud detector: {e}. Continuing without it.")
                    self.fraud_detector = None
            
            # Bias Detector removed - not relevant for review detection
            # The autism-bias-detection-roberta model is for autism-related bias, not commercial reviews
            self.bias_detector = None
            
            # Initialize translation
            if self.config.translation_enabled and TRANSLATION_AVAILABLE:
                self.translator_initialized = True
                logger.info(" Translation service ready")
            
        except Exception as e:
            logger.error(f" Error loading models: {e}", exc_info=True)
            raise



        # בסוף הפונקציה _load_models
        try:
            base_dir = Path(__file__).resolve().parent
            model_path = base_dir / "models" / "hebrew_fake_review_tfidf.joblib"
            if model_path.exists():
                logger.info(f"Loading local Hebrew TF-IDF model from {model_path}...")
                self.local_hebrew_model = joblib.load(model_path)
                logger.info(" Local Hebrew TF-IDF model loaded")
            else:
                logger.warning(f"Local Hebrew model not found at {model_path}")
        except Exception as e:
            logger.warning(f"Could not load local Hebrew model: {e}")
            self.local_hebrew_model = None

    
    def is_ready(self) -> bool:
        """Check if models are loaded and ready."""
        return self.review_classifier is not None and self.ai_detector is not None
    
    def has_bias_fraud_models(self) -> bool:
        """Check if bias/fraud detection models are available."""
        # Only fraud detector is used - bias detector was removed as not relevant
        return self.fraud_detector is not None


class TranslationService:
    """Handles translation from Hebrew to English."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.enabled = config.translation_enabled and TRANSLATION_AVAILABLE
    
    def translate(self, text: str) -> str:
        """
        Translate text from Hebrew to English.
        
        Args:
            text: Input text (assumed Hebrew)
            
        Returns:
            Translated English text, or original if translation fails
        """
        if not self.enabled or not text or not text.strip():
            return text
        
        try:
            if USE_GOOGLETRANS:
                translator = Translator()
                result = translator.translate(text, src='he', dest='en')
                return result.text
            else:
                translator = GoogleTranslator(
                    source=self.config.translation_source_lang,
                    target=self.config.translation_target_lang
                )
                return translator.translate(text)
        except Exception as e:
            logger.warning(f"Translation error: {e}. Using original text.")
            return text


class ModelInference:
    """Handles inference with ML models and extracts structured outputs."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def run_review_classifier(self, text: str) -> ModelOutput:
        """
        Run the review-specific classifier (Model 1).
        
        Args:
            text: Input text to classify (must be non-empty)
        
        Returns:
            ModelOutput with fake/real scores (all values clamped to [0.0, 1.0])
        
        Raises:
            RuntimeError: If model not loaded
            ValueError: If input text is invalid
        """
        if not self.model_manager.review_classifier:
            raise RuntimeError("Review classifier not loaded")
        
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        if not text.strip():
            # Empty or whitespace-only text - return neutral scores
            logger.warning("Empty text provided to review classifier")
            return ModelOutput(
                fake_score=0.5,
                real_score=0.5,
                confidence=0.0,
                model_name="Review Classifier",
                raw_output=None
            )
        
        try:
            results = self.model_manager.review_classifier(text)
            
            # Parse results
            fake_score, real_score = self._parse_model1_results(results)
            
            # CRITICAL: Clamp all scores to [0.0, 1.0] range
            fake_score = max(0.0, min(1.0, float(fake_score)))
            real_score = max(0.0, min(1.0, float(real_score)))
            
            # Normalize to ensure they sum to 1.0
            total = fake_score + real_score
            if total > 0:
                fake_score /= total
                real_score /= total
            else:
                # Fallback if both are 0
                fake_score = 0.5
                real_score = 0.5
            
            confidence = max(fake_score, real_score)
            
            return ModelOutput(
                fake_score=fake_score,
                real_score=real_score,
                confidence=confidence,
                model_name="Review Classifier",
                raw_output=results
            )
        except Exception as e:
            logger.error(f"Error in review classifier: {e}", exc_info=True)
            raise

    def run_hebrew_ai_detector(self, text: str) -> Optional[ModelOutput]:
        """
        Run Hebrew AI detector on original Hebrew text (before translation).

        This is critical because translation can lose AI-specific patterns in Hebrew.
        """
        if not self.model_manager.hebrew_ai_detector:
            return None

        if not text or not isinstance(text, str) or not text.strip():
            return None

        try:
            result = self.model_manager.hebrew_ai_detector(text)

            # Extract scores - same logic as run_ai_detector
            if isinstance(result, list) and len(result) > 0:
                scores = {}
                for item in result:
                    label = item.get("label", "").upper()
                    score = float(item.get("score", 0.0))
                    scores[label] = score

                # Map labels to fake/real
                fake_score = 0.0
                real_score = 0.0

                for label, score in scores.items():
                    if "FAKE" in label or "AI" in label or "GENERATED" in label or "SYNTHETIC" in label:
                        fake_score = max(fake_score, score)
                    elif "REAL" in label or "HUMAN" in label or "AUTHENTIC" in label:
                        real_score = max(real_score, score)

                # Normalize if needed
                total = fake_score + real_score
                if total > 0:
                    fake_score /= total
                    real_score /= total
                else:
                    fake_score = 0.5
                    real_score = 0.5

                confidence = abs(fake_score - real_score)

                return ModelOutput(
                    fake_score=max(0.0, min(1.0, fake_score)),
                    real_score=max(0.0, min(1.0, real_score)),
                    confidence=max(0.0, min(1.0, confidence)),
                    model_name="Hebrew AI Detector",
                    raw_output=result
                )
        except Exception as e:
            logger.warning(f"Hebrew AI detector error: {e}")
            return None

    def run_ai_detector(self, text: str) -> ModelOutput:
        """
        Run the AI detector (Model 2).
        
        Args:
            text: Input text to analyze (must be non-empty)
        
        Returns:
            ModelOutput with generated/human scores (all values clamped to [0.0, 1.0])
        
        Raises:
            RuntimeError: If model not loaded
            ValueError: If input text is invalid
        """
        if not self.model_manager.ai_detector:
            raise RuntimeError("AI detector not loaded")
        
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        if not text.strip():
            # Empty or whitespace-only text - return neutral scores
            logger.warning("Empty text provided to AI detector")
            return ModelOutput(
                fake_score=0.5,
                real_score=0.5,
                confidence=0.0,
                model_name="AI Detector",
                raw_output=None
            )
        
        try:
            results = self.model_manager.ai_detector(text)
            
            # Parse results
            generated_score, human_score = self._parse_model2_results(results)
            
            # CRITICAL: Clamp all scores to [0.0, 1.0] range
            generated_score = max(0.0, min(1.0, float(generated_score)))
            human_score = max(0.0, min(1.0, float(human_score)))
            
            # Normalize to ensure they sum to 1.0
            total = generated_score + human_score
            if total > 0:
                generated_score /= total
                human_score /= total
            else:
                # Fallback if both are 0
                generated_score = 0.5
                human_score = 0.5
            
            confidence = max(generated_score, human_score)
            
            return ModelOutput(
                fake_score=generated_score,  # Generated = fake
                real_score=human_score,      # Human = real
                confidence=confidence,
                model_name="AI Detector",
                raw_output=results
            )
        except Exception as e:
            logger.error(f"Error in AI detector: {e}", exc_info=True)
            raise
    
    def _parse_model1_results(self, results: Any) -> Tuple[float, float]:
        """Parse Model 1 (review classifier) results."""
        fake_score = 0.0
        real_score = 0.0
        
        if isinstance(results, list) and len(results) > 0:
            results_list = results[0] if isinstance(results[0], list) else results
            
            for result in results_list:
                label = result.get('label', '').upper()
                score = float(result.get('score', 0.0))
                
                # Handle LABEL_0/LABEL_1 format
                if 'LABEL_0' in label:
                    real_score = score
                elif 'LABEL_1' in label:
                    fake_score = score
                # Handle named labels
                elif any(x in label for x in ['CG', 'FAKE', 'GENERATED', 'COMPUTER']):
                    fake_score = score
                elif any(x in label for x in ['REAL', 'HUMAN', 'ORIGINAL']):
                    real_score = score
        
        # If only one score found, infer the other
        if fake_score > 0 and real_score == 0:
            real_score = 1.0 - fake_score
        elif real_score > 0 and fake_score == 0:
            fake_score = 1.0 - real_score
        
        return fake_score, real_score
    
    def _parse_model2_results(self, results: Any) -> Tuple[float, float]:
        """Parse Model 2 (AI detector) results."""
        generated_score = 0.0
        human_score = 0.0
        
        if isinstance(results, list) and len(results) > 0:
            results_list = results[0] if isinstance(results[0], list) else results
            
            for result in results_list:
                label = result.get('label', '').lower()
                score = float(result.get('score', 0.0))
                
                if any(x in label for x in ['generated', 'fake', 'ai']):
                    generated_score = score
                elif any(x in label for x in ['human', 'real', 'original']):
                    human_score = score
                # Handle LABEL format
                elif 'label_1' in label.lower():
                    generated_score = score
                elif 'label_0' in label.lower():
                    human_score = score
        
        # If only one score found, infer the other
        if generated_score > 0 and human_score == 0:
            human_score = 1.0 - generated_score
        elif human_score > 0 and generated_score == 0:
            generated_score = 1.0 - human_score
        
        return generated_score, human_score
    
    def run_local_hebrew_model(self, text: str) -> Optional[ModelOutput]:
        manager = self.model_manager
        if not manager.local_hebrew_model:
            return None
        if not text or not isinstance(text, str) or not text.strip():
            return None
        try:
            proba = manager.local_hebrew_model.predict_proba([text])[0]
            # נניח class 1 = FAKE, class 0 = REAL
            fake_score = float(proba[1])
            real_score = float(proba[0])
            confidence = abs(fake_score - real_score)
            return ModelOutput(
                fake_score=fake_score,
                real_score=real_score,
                confidence=confidence,
                model_name="Hebrew TF-IDF Classifier",
                raw_output={"proba": proba.tolist()}
            )
        except Exception as e:
            logger.warning(f"Local Hebrew model error: {e}")
            return None


def get_model_manager() -> ModelManager:
    """Get or create the global model manager instance."""
    global _model_instances, _config_instance
    
    if _config_instance is None:
        if ARCHITECTURE_AVAILABLE:
            _config_instance = get_config()
        else:
            # Fallback configuration
            from ml_config import SystemConfig, ModelConfig, Thresholds
            _config_instance = SystemConfig(
                review_classifier=ModelConfig(
                    name="Review Classifier",
                    model_id="debojit01/fake-review-detector",
                    device=-1
                ),
                ai_detector=ModelConfig(
                    name="AI Detector",
                    model_id="roberta-base-openai-detector",
                    device=-1
                ),
                thresholds=Thresholds()
            )
    
    if 'manager' not in _model_instances:
        _model_instances['manager'] = ModelManager(_config_instance)
    
    return _model_instances['manager']


def classify_review(review_text: str) -> Dict[str, Any]:
    """
    גרסה משופרת ומכוילת לפרודקשן:
    - לא "מכריזה REAL" כשאין הוכחה חזקה (מונע False-Real על זיופים חדשים)
    - נותנת עדיפות למודל העברי המקומי + Suspicious Patterns (שזה הכי רלוונטי בעברית)
    - משתמשת בדאטהסט רק כשיש התאמה חזקה/מדויקת (כדי לא "להעתיק" רעש)
    - מחזירה גם UNCERTAIN באזור אפור (הכי נכון בעולם אמיתי)
    """

    def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        try:
            return max(lo, min(hi, float(x)))
        except Exception:
            return 0.5

    def _make_response(
        classification: str,
        fake_probability: float,
        confidence: float,
        model_used: str,
        translated_text: str,
        reasoning: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out = {
            "classification": classification,
            "score": _clamp(confidence),
            "fake_probability": _clamp(fake_probability),
            "model_used": model_used,
            "translated_text": translated_text,
            "reasoning": reasoning or "",
        }
        if extra:
            out.update(extra)
        return out

    # =========================
    # 0) Input validation
    # =========================
    if review_text is None:
        raise ValueError("review_text cannot be None")
    if not isinstance(review_text, str):
        raise ValueError(f"review_text must be a string, got {type(review_text)}")
    if not review_text.strip():
        logger.warning("Empty review text provided")
        return _make_response(
            classification="UNCERTAIN",
            fake_probability=0.5,
            confidence=0.0,
            model_used="Input Validation",
            translated_text="",
            reasoning="Empty or whitespace-only input text",
            extra={"error": "Input text is empty"},
        )

    # =========================
    # 1) Init components
    # =========================
    try:
        config = get_config() if ARCHITECTURE_AVAILABLE else None
        if config is None:
            return _classify_review_fallback(review_text)

        model_manager = get_model_manager()
        translation_service = TranslationService(config)
        inference = ModelInference(model_manager)
        pattern_detector = SuspiciousPatternDetector(config.suspicious_patterns)
        ensemble = EnsembleClassifier(config)
    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        return _classify_review_fallback(review_text)

    # ברירות מחדל לספים (אפשר לכייל ב-ml_config.py)
    thr_fake = float(getattr(getattr(config, "thresholds", None), "fake_classification_threshold", 0.5) or 0.5)
    # "אזור אפור" (כדי לא להכריז REAL/FAKE כשזה 50/50)
    grey_lo, grey_hi = 0.45, 0.55

    # =========================
    # 2) Dataset exact / high-confidence matching (ללא תרגום)
    # =========================
    try:
        norm_text = _normalize(review_text)

        # (א) התאמה מדויקת = החלטה חזקה מאוד
        exact_map = model_manager.get_exact_map()
        if norm_text and norm_text in exact_map:
            label = exact_map[norm_text].upper()
            return _make_response(
                classification="FAKE" if label == "FAKE" else "REAL",
                fake_probability=0.92 if label == "FAKE" else 0.08,
                confidence=0.96,
                model_used="Dataset Exact",
                translated_text=review_text,
                reasoning="Exact match to labeled dataset entry (strong signal).",
            )

        # (ב) התאמת cosine/char-ngrams חזקה מאוד בלבד
        matcher = model_manager.get_dataset_matcher()
        if matcher:
            hit = matcher.predict(review_text)
            if hit:
                conf_ds = float(hit.get("confidence", 0.0))
                # כאן מעלים את הסף כדי לא "לגנוב" החלטות חלשות מהדאטהסט
                if conf_ds >= 0.85:
                    return _make_response(
                        classification=hit.get("classification", "UNCERTAIN"),
                        fake_probability=float(hit.get("fake_probability", 0.5)),
                        confidence=conf_ds,
                        model_used="Dataset Matcher (High-Conf)",
                        translated_text=review_text,
                        reasoning=hit.get("reasoning", "High-confidence dataset match."),
                    )
    except Exception as e:
        logger.warning(f"Dataset matching error: {e}")

    # =========================
    # 3) Translation (אם יש)
    # =========================
    try:
        translated_text = translation_service.translate(review_text)
    except Exception as e:
        logger.warning(f"Translation failed: {e}. Using original text.")
        translated_text = review_text

    # =========================
    # 4) Suspicious patterns (עברית + אנגלית)
    # =========================
    try:
        patterns_he = pattern_detector.detect_all_patterns(review_text)
        patterns_en = pattern_detector.detect_all_patterns(translated_text) if translated_text != review_text else {}

        # מיזוג לפי confidence הגבוה יותר
        suspicious_patterns = {}
        for k, v in (patterns_he or {}).items():
            suspicious_patterns[k] = v
        for k, v in (patterns_en or {}).items():
            if k in suspicious_patterns:
                if getattr(v, "confidence", 0.0) > getattr(suspicious_patterns[k], "confidence", 0.0):
                    suspicious_patterns[k] = v
            else:
                suspicious_patterns[k] = v

        suspicious_score = float(pattern_detector.calculate_suspicious_score(suspicious_patterns, review_text))
        suspicious_score = _clamp(suspicious_score)
    except Exception as e:
        logger.warning(f"Pattern detection error: {e}")
        suspicious_patterns, suspicious_score = {}, 0.0

    # =========================
    # 5) Model inference
    # =========================
    # מודל עברי מקומי (הכי חשוב אצלך כי הוא מאומן על הדאטהסט שלך)
    local_hebrew_output = None
    hebrew_ai_output = None
    model1_output = None
    model2_output = None

    try:
        hebrew_ai_output = inference.run_hebrew_ai_detector(review_text)  # יכול להיות None
    except Exception as e:
        logger.warning(f"Hebrew AI detector error: {e}")

    try:
        local_hebrew_output = inference.run_local_hebrew_model(review_text)  # יכול להיות None
    except Exception as e:
        logger.warning(f"Local Hebrew model error: {e}")

    try:
        # מודל 1+2 עובדים על הטקסט המתורגם (אם קיימים)
        model1_output = inference.run_review_classifier(translated_text)
        model2_output = inference.run_ai_detector(translated_text)

        # אם יש סיגנל עברי ל-AI, נשלב אותו בזהירות (לא להשתלט)
        if hebrew_ai_output and getattr(hebrew_ai_output, "fake_score", 0.0) > 0.30:
            hf = float(hebrew_ai_output.fake_score)
            hr = float(hebrew_ai_output.real_score)
            ef = float(model2_output.fake_score)
            er = float(model2_output.real_score)

            combined_fake = 0.65 * hf + 0.35 * ef
            combined_real = 0.35 * hr + 0.65 * er
            total = combined_fake + combined_real
            if total > 0:
                combined_fake /= total
                combined_real /= total

            model2_output = ModelOutput(
                fake_score=_clamp(combined_fake),
                real_score=_clamp(combined_real),
                confidence=_clamp(abs(combined_fake - combined_real)),
                model_name=getattr(model2_output, "model_name", "AI Detector"),
                raw_output=getattr(model2_output, "raw_output", None),
            )
    except Exception as e:
        logger.error(f"Model inference error: {e}", exc_info=True)
        # אם המודלים נפלו, נבנה החלטה מהירה על בסיס local+patterns
        # (עדיף UNCERTAIN מאשר REAL בטעות)
        fallback_fake = 0.5
        fallback_conf = 0.4

        if local_hebrew_output:
            fallback_fake = _clamp(getattr(local_hebrew_output, "fake_score", 0.5))
            fallback_conf = max(fallback_conf, _clamp(getattr(local_hebrew_output, "confidence", 0.0)))

        # חשד גבוה מעלה את fake_prob מעט
        fallback_fake = _clamp(fallback_fake + 0.25 * suspicious_score)
        fallback_conf = _clamp(max(fallback_conf, abs(fallback_fake - 0.5) * 2))

        # החלטה שמרנית: באזור אפור -> UNCERTAIN
        if grey_lo <= fallback_fake <= grey_hi:
            cls = "UNCERTAIN"
        else:
            cls = "FAKE" if fallback_fake >= thr_fake else "REAL"

        # לא מחזירים REAL אם יש חשד משמעותי
        if cls == "REAL" and suspicious_score >= 0.40:
            cls = "UNCERTAIN"

        return _make_response(
            classification=cls,
            fake_probability=fallback_fake,
            confidence=fallback_conf,
            model_used="Fallback (local/patterns)",
            translated_text=translated_text,
            reasoning=f"Model inference failed; used local+patterns. suspicious_score={suspicious_score:.2f}",
            extra={"error": str(e)},
        )

    # =========================
    # 6) Ensemble classification (בסיס)
    # =========================
    try:
        result = ensemble.classify(
            model1_output=model1_output,
            model2_output=model2_output,
            suspicious_patterns=suspicious_patterns,
            suspicious_score=suspicious_score,
            translated_text=translated_text,
            original_text=review_text,
        )
    except Exception as e:
        logger.error(f"Ensemble classification error: {e}", exc_info=True)
        # fallback: local + patterns + ai_detector
        base_fake = _clamp(getattr(model2_output, "fake_score", 0.5))
        base_conf = _clamp(getattr(model2_output, "confidence", 0.0))

        if local_hebrew_output:
            lf = _clamp(getattr(local_hebrew_output, "fake_score", 0.5))
            lc = _clamp(getattr(local_hebrew_output, "confidence", 0.0))
            # שילוב שמעדיף את המודל העברי
            base_fake = _clamp(0.70 * lf + 0.30 * base_fake)
            base_conf = max(base_conf, lc)

        base_fake = _clamp(base_fake + 0.20 * suspicious_score)
        base_conf = _clamp(max(base_conf, abs(base_fake - 0.5) * 2))

        if grey_lo <= base_fake <= grey_hi:
            cls = "UNCERTAIN"
        else:
            cls = "FAKE" if base_fake >= thr_fake else "REAL"

        # לא מחזירים REAL אם יש חשד
        if cls == "REAL" and suspicious_score >= 0.40:
            cls = "UNCERTAIN"

        return _make_response(
            classification=cls,
            fake_probability=base_fake,
            confidence=base_conf,
            model_used="Ensemble Failed -> Local/AI/Patterns",
            translated_text=translated_text,
            reasoning=f"Ensemble failed; combined local/AI/patterns. suspicious_score={suspicious_score:.2f}",
            extra={"error": str(e)},
        )

    # =========================
    # 7) Fuse signals (החלק החשוב!)
    # =========================
    # נתחיל מה-ensemble כתוצאה בסיסית
    fused_fake = _clamp(getattr(result, "fake_probability", 0.5))
    fused_conf = _clamp(getattr(result, "confidence", 0.5))

    # (א) הזרקת Suspicious patterns: חשד גבוה מעלה fake_prob
    # רעיון: אם יש הרבה דפוסים גנריים/שיווקיים => זה מעלה חשד לזיוף
    fused_fake = _clamp(fused_fake + 0.25 * suspicious_score)

    # (ב) שילוב המודל העברי המקומי (הכי חשוב אצלך)
    if local_hebrew_output:
        lf = _clamp(getattr(local_hebrew_output, "fake_score", 0.5))
        lc = _clamp(getattr(local_hebrew_output, "confidence", 0.0))

        # אם המודל העברי בטוח -> משקל גבוה
        if lc >= 0.65:
            fused_fake = _clamp(0.70 * lf + 0.30 * fused_fake)
            fused_conf = _clamp(max(fused_conf, lc, abs(fused_fake - 0.5) * 2))
            result.model_used = (getattr(result, "model_used", "") + " + HebrewTFIDF(Strong)").strip()
            result.reasoning = (
                f"Hebrew TF-IDF strong signal (conf={lc:.2f}, fake={lf:.2f}) fused. "
                + (getattr(result, "reasoning", "") or "")
            )
        else:
            # אם ה-ensemble לא בטוח, המודל העברי יכול לשבור תיקו
            if fused_conf < 0.60:
                fused_fake = _clamp(0.55 * lf + 0.45 * fused_fake)
                fused_conf = _clamp(max(fused_conf, 0.60, lc, abs(fused_fake - 0.5) * 2))
                result.model_used = (getattr(result, "model_used", "") + " + HebrewTFIDF(TieBreak)").strip()
                result.reasoning = (
                    f"Hebrew TF-IDF used as tie-break (conf={lc:.2f}, fake={lf:.2f}). "
                    + (getattr(result, "reasoning", "") or "")
                )

    # (ג) סיגנל AI detector (חלש יותר אצלך, כי זה לא מודל Fake-Review אמיתי)
    # רק אם הוא נותן דחיפה חזקה לכיוון FAKE
    ai_fake = _clamp(getattr(model2_output, "fake_score", 0.5))
    ai_real = _clamp(getattr(model2_output, "real_score", 0.5))
    ai_margin = ai_fake - ai_real
    if ai_margin > 0.20:
        fused_fake = _clamp(0.85 * fused_fake + 0.15 * ai_fake)
        fused_conf = _clamp(max(fused_conf, abs(fused_fake - 0.5) * 2))

    # =========================
    # 8) Decision policy (מונע False REAL)
    # =========================
    # Confidence recalibration (לא רק מהמודלים): כמה רחוק מ-0.5
    fused_conf = _clamp(max(fused_conf, abs(fused_fake - 0.5) * 2))

    # כלל בטיחות: אם יש חשד משמעותי, לא מחזירים REAL
    if suspicious_score >= 0.40 and fused_fake < 0.50:
        # במקום "REAL", עדיף UNCERTAIN
        fused_fake = max(fused_fake, 0.50)  # דוחף לאזור האפור
        fused_conf = max(fused_conf, 0.60)

    # החלטה תלת-מצבית:
    # - FAKE אם מעל סף
    # - REAL אם מתחת לסף אמיתיות מחמיר
    # - אחרת UNCERTAIN
    real_strict_thr = min(0.40, 1.0 - thr_fake)  # מחמיר כדי לא "לשחרר" REAL בטעות
    if fused_fake >= thr_fake:
        final_cls = "FAKE"
    elif fused_fake <= real_strict_thr and suspicious_score < 0.30 and fused_conf >= 0.65:
        final_cls = "REAL"
    else:
        final_cls = "UNCERTAIN"

    # =========================
    # 9) Bias/Fraud (אופציונלי) - יכול לדרוס הכל
    # =========================
    bias_fraud_result = None
    if ARCHITECTURE_AVAILABLE and BiasFraudDetector and model_manager.has_bias_fraud_models():
        try:
            bias_fraud_detector = BiasFraudDetector(config)
            bias_fraud_detector.load_models(model_manager.fraud_detector, None)
            bias_fraud_result = bias_fraud_detector.check(translated_text, model1_output, model2_output)
            if bias_fraud_result:
                final_cls = bias_fraud_result.classification
                fused_conf = _clamp(bias_fraud_result.confidence)
                fused_fake = _clamp(bias_fraud_result.score)
                result.model_used = (getattr(result, "model_used", "") + " + FraudDetector").strip()
                result.reasoning = (
                    f"Non-objective origin detected: {bias_fraud_result.classification} ({bias_fraud_result.score:.2%}). "
                    + (getattr(result, "reasoning", "") or "")
                )
        except Exception as e:
            logger.warning(f"Bias/Fraud detection error: {e}. Continuing.")

    # =========================
    # 10) Build final response
    # =========================
    return _make_response(
        classification=final_cls,
        fake_probability=fused_fake,
        confidence=fused_conf,
        model_used=getattr(result, "model_used", "Ensemble+Fusion"),
        translated_text=translated_text,
        reasoning=getattr(result, "reasoning", ""),
        extra={
            "m1_cg_score": _clamp(getattr(model1_output, "fake_score", 0.5)),
            "m1_real_score": _clamp(getattr(model1_output, "real_score", 0.5)),
            "m2_generated_score": _clamp(getattr(model2_output, "fake_score", 0.5)),
            "m2_real_score": _clamp(getattr(model2_output, "real_score", 0.5)),
            "suspicious_score": _clamp(suspicious_score),
            "suspicious_patterns": {k: getattr(v, "description", "") for k, v in (suspicious_patterns or {}).items()},
            "bias_fraud_detected": getattr(bias_fraud_result, "classification", None) if bias_fraud_result else None,
            "bias_fraud_score": getattr(bias_fraud_result, "score", None) if bias_fraud_result else None,
        },
    )



def _classify_review_fallback(review_text: str) -> Dict[str, Any]:
    """Fallback classification when architecture modules are not available."""
    if not ML_AVAILABLE:
        return {
            'classification': 'UNCERTAIN',
            'score': random.uniform(0.0, 1.0),
            'model_used': 'Placeholder',
            'translated_text': review_text
        }
    
    try:
        return {
            'classification': 'UNCERTAIN',
            'score': 0.5,
            'model_used': 'Fallback',
            'translated_text': review_text,
            'error': 'Architecture modules not available'
        }
    except Exception as e:
        return {
            'classification': 'UNCERTAIN',
            'score': 0.5,
            'model_used': 'Error',
            'translated_text': review_text,
            'error': str(e)
        }


def detect_fake_review(text: str) -> float:
    """
    API-compatible wrapper function.
    
    Returns:
        float: Probability that the review is fake (0.0 to 1.0)
    """
    result = classify_review(text)
    
    # Extract fake probability
    fake_prob = result.get('fake_probability')
    if fake_prob is not None:
        return float(fake_prob)
    
    # Fallback to score-based calculation
    classification = result.get('classification', 'UNCERTAIN').upper()
    score = result.get('score', 0.5)
    
    if 'FAKE' in classification:
        return float(score)
    elif 'REAL' in classification:
        return 1.0 - float(score)
    else:
        return 0.5


def load_models() -> bool:
    """
    Load all ML models (for compatibility with existing code).
    
    Returns:
        True if models loaded successfully, False otherwise
    """
    try:
        manager = get_model_manager()
        return manager.is_ready()
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        return False


# Initialize models on module import
if __name__ != "__main__":
    try:
        if ML_AVAILABLE:
            load_models()
    except Exception as e:
        logger.warning(f"Could not auto-load models: {e}. Models will load on first use.")
