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
        self.local_hebrew_model = None  # ✅ להגדיר לפני הטעינה
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
                logger.info(f"✅ {self.config.review_classifier.name} loaded successfully")
            
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
                    logger.info(f"✅ {self.config.hebrew_ai_detector.name} loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load Hebrew AI detector: {e}. Continuing without it.")
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
                logger.info(f"✅ {self.config.ai_detector.name} loaded successfully")
            
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
                    logger.info(f"✅ {self.config.fraud_detector.name} loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load fraud detector: {e}. Continuing without it.")
                    self.fraud_detector = None
            
            # Bias Detector removed - not relevant for review detection
            # The autism-bias-detection-roberta model is for autism-related bias, not commercial reviews
            self.bias_detector = None
            
            # Initialize translation
            if self.config.translation_enabled and TRANSLATION_AVAILABLE:
                self.translator_initialized = True
                logger.info("✅ Translation service ready")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}", exc_info=True)
            raise



        # בסוף הפונקציה _load_models
        try:
            base_dir = Path(__file__).resolve().parent
            model_path = base_dir / "models" / "hebrew_fake_review_tfidf.joblib"
            if model_path.exists():
                logger.info(f"Loading local Hebrew TF-IDF model from {model_path}...")
                self.local_hebrew_model = joblib.load(model_path)
                logger.info("✅ Local Hebrew TF-IDF model loaded")
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
    Main classification function - production-ready implementation.
    
    This function orchestrates the entire pipeline:
    1. Translation (if needed)
    2. Pattern detection
    3. Model inference
    4. Ensemble classification
    5. Bias/Fraud detection (if triggered)
    
    Args:
        review_text: The review text to classify (supports Hebrew)
                    Must be a non-empty string
        
    Returns:
        Dictionary with classification results and metadata.
        All confidence and probability values are guaranteed to be in [0.0, 1.0] range.
        
    Raises:
        ValueError: If input is invalid (None, empty, or wrong type)
    """
    # CRITICAL: Input validation
    if review_text is None:
        raise ValueError("review_text cannot be None")
    
    if not isinstance(review_text, str):
        raise ValueError(f"review_text must be a string, got {type(review_text)}")
    
    if not review_text.strip():
        # Empty or whitespace-only - return uncertain result
        logger.warning("Empty review text provided")
        return {
            'classification': 'UNCERTAIN',
            'score': 0.0,
            'fake_probability': 0.5,
            'model_used': 'Input Validation',
            'translated_text': '',
            'reasoning': 'Empty or whitespace-only input text',
            'error': 'Input text is empty'
        }
    
    # Initialize components
    try:
        config = get_config() if ARCHITECTURE_AVAILABLE else None
        if config is None:
            # Fallback mode
            return _classify_review_fallback(review_text)
        
        model_manager = get_model_manager()
        translation_service = TranslationService(config)
        inference = ModelInference(model_manager)
        pattern_detector = SuspiciousPatternDetector(config.suspicious_patterns)
        ensemble = EnsembleClassifier(config)
        
    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        return _classify_review_fallback(review_text)

    # 0) Exact/normalized match + high-confidence dataset matcher (no translation)
    try:
        norm_text = _normalize(review_text)
        exact_map = model_manager.get_exact_map()
        if norm_text and norm_text in exact_map:
            label = exact_map[norm_text]
            logger.info(f"✅ Exact dataset match found. Label={label}")
            return {
                'classification': 'FAKE' if label.upper() == 'FAKE' else 'REAL',
                'fake_probability': 0.9 if label.upper() == 'FAKE' else 0.1,
                'confidence': 0.95,
                'model_used': 'Dataset Exact',
                'translated_text': review_text,
                'reasoning': "Exact match to labeled dataset entry."
            }

        matcher = model_manager.get_dataset_matcher()
        if matcher:
            high_conf_ds = matcher.predict(review_text)
            if high_conf_ds:
                conf_ds = high_conf_ds.get("confidence", 0)
                logger.info(
                    f"Dataset matcher result: conf={conf_ds:.3f}, "
                    f"class={high_conf_ds.get('classification')}, "
                    f"fake_prob={high_conf_ds.get('fake_probability', 0):.3f}"
                )
                # High-confidence gate kept at 0.80 for precision
                if conf_ds >= 0.80:
                    logger.info(
                        f"✅ Dataset matcher high-confidence hit: conf={conf_ds:.3f}, "
                        f"class={high_conf_ds['classification']}, fake_prob={high_conf_ds['fake_probability']:.3f}"
                    )
                    return {
                        'classification': high_conf_ds['classification'],
                        'fake_probability': high_conf_ds['fake_probability'],
                        'confidence': conf_ds,
                        'model_used': 'Dataset High-Confidence',
                        'translated_text': review_text,
                        'reasoning': high_conf_ds['reasoning']
                    }
                else:
                    logger.info(f"Dataset matcher below high-confidence gate (0.80): conf={conf_ds:.3f}")
    except Exception as e:
        logger.warning(f"Dataset matcher error: {e}")
    
    try:
        translated_text = translation_service.translate(review_text)
        if translated_text != review_text:
            logger.info(f"Translation successful: {translated_text[:100]}...")
    except Exception as e:
        logger.warning(f"Translation failed: {e}. Using original text.")
        translated_text = review_text
    
    # 1) Dataset-based heuristic prediction (original Hebrew text)
    dataset_result = _predict_from_dataset(review_text, model_manager.dataset_reviews)

    # 2) Pattern detection (Hebrew + English)
    try:
        patterns_hebrew = pattern_detector.detect_all_patterns(review_text)  # Original Hebrew
        patterns_english = pattern_detector.detect_all_patterns(translated_text)  # Translated English

        suspicious_patterns = {}
        for pattern_type, match in patterns_hebrew.items():
            suspicious_patterns[pattern_type] = match
        for pattern_type, match in patterns_english.items():
            if pattern_type in suspicious_patterns:
                if match.confidence > suspicious_patterns[pattern_type].confidence:
                    suspicious_patterns[pattern_type] = match
            else:
                suspicious_patterns[pattern_type] = match

        suspicious_score = pattern_detector.calculate_suspicious_score(suspicious_patterns, review_text)
        if suspicious_score > 0:
            pattern_summary = ", ".join([f"{k}:{v.confidence:.2f}" for k, v in suspicious_patterns.items()])
            logger.info(
                f"Suspicious patterns detected: {suspicious_score:.2f} "
                f"(Hebrew: {len(patterns_hebrew)}, English: {len(patterns_english)}) | {pattern_summary}"
            )
    except Exception as e:
        logger.warning(f"Pattern detection error: {e}")
        suspicious_patterns = {}
        suspicious_score = 0.0

    # 3) Model inference (Hebrew AI detector + English detectors)
    try:
        # 🔹 דטקטור עברית (קיים אצלך כבר)
        hebrew_ai_output = inference.run_hebrew_ai_detector(review_text)
        if hebrew_ai_output:
            logger.info(
                f"Hebrew AI detector: fake={hebrew_ai_output.fake_score:.2%}, "
                f"real={hebrew_ai_output.real_score:.2%}"
            )

        # 🔹 המודל המקומי החדש בעברית (TF-IDF / לוגיסטי וכו')
        local_hebrew_output = inference.run_local_hebrew_model(review_text)


        # 🔹 אם המודל העברי די בטוח בעצמו – נותנים לו להחליט עם סף FAKE רגיש יותר
        if local_hebrew_output and local_hebrew_output.confidence >= 0.65:
            fake = float(local_hebrew_output.fake_score)
            # להעדיף זיהוי זיופים → סף נמוך יותר, למשל 0.40
            threshold = 0.40  
            return {
                'classification': 'FAKE' if fake >= threshold else 'REAL',
                'score': local_hebrew_output.confidence,
                'fake_probability': fake,
                'model_used': f'Hebrew TF-IDF (high confidence, thr={threshold})',
                'translated_text': review_text,
                'reasoning': (
                    f"Local Hebrew TF-IDF model is confident "
                    f"(conf={local_hebrew_output.confidence:.2f}, fake={fake:.2f}) "
                    f"with FAKE threshold={threshold:.2f}."
                )
            }



        # 🔹 מודל 1 – Classifier על הטקסט המתורגם
        model1_output = inference.run_review_classifier(translated_text)

        # 🔹 מודל 2 – AI detector על הטקסט המתורגם
        model2_output = inference.run_ai_detector(translated_text)

        # Combine Hebrew AI detector with English AI detector (prioritize Hebrew signal)
        if hebrew_ai_output and hebrew_ai_output.fake_score > 0.3:
            combined_fake = (hebrew_ai_output.fake_score * 0.7) + (model2_output.fake_score * 0.3)
            combined_real = (hebrew_ai_output.real_score * 0.3) + (model2_output.real_score * 0.7)
            total = combined_fake + combined_real
            if total > 0:
                combined_fake /= total
                combined_real /= total
            model2_output = ModelOutput(
                fake_score=max(0.0, min(1.0, combined_fake)),
                real_score=max(0.0, min(1.0, combined_real)),
                confidence=max(0.0, min(1.0, abs(combined_fake - combined_real))),
                model_name=model2_output.model_name,
                raw_output=model2_output.raw_output
            )
            logger.info(f"Combined AI detectors (Hebrew + English): fake={model2_output.fake_score:.2%}, real={model2_output.real_score:.2%}")
    except Exception as e:
        logger.error(f"Model inference error: {e}", exc_info=True)
        return {
            'classification': 'UNCERTAIN',
            'score': 0.5,
            'model_used': 'Error',
            'translated_text': translated_text,
            'error': str(e)
        }
    
    try:
        result = ensemble.classify(
            model1_output=model1_output,
            model2_output=model2_output,
            suspicious_patterns=suspicious_patterns,
            suspicious_score=suspicious_score,
            translated_text=translated_text,
            original_text=review_text
        )


                # 🔹 שילוב המודל המקומי בעברית (TF-IDF) לפני השוואה לדאטהסט
        if local_hebrew_output:
            logger.info(
                "Local Hebrew TF-IDF model: fake=%.3f, real=%.3f, conf=%.3f",
                local_hebrew_output.fake_score,
                local_hebrew_output.real_score,
                local_hebrew_output.confidence,
            )

            # אם המודל העברי די בטוח בעצמו – נותנים לו משקל גבוה
            if local_hebrew_output.confidence >= 0.65:
                # משקל 60% למודל העברי, 40% לתוצאה של ה-ensemble
                combined_fake = (
                    0.6 * local_hebrew_output.fake_score
                    + 0.4 * result.fake_probability
                )
                result.fake_probability = combined_fake
                result.classification = "FAKE" if combined_fake >= 0.5 else "REAL"
                # הביטחון: המקסימום בין הביטחון הקודם לבין המודל העברי
                result.confidence = max(
                    result.confidence,
                    local_hebrew_output.confidence,
                    abs(combined_fake - 0.5) * 2,  # כמה רחוק מ-50%
                )
                result.model_used = (result.model_used + " + Hebrew TF-IDF").strip()
                result.reasoning = (
                    f"Hebrew TF-IDF model (conf={local_hebrew_output.confidence:.2f}, "
                    f"fake={local_hebrew_output.fake_score:.2f}) combined with ensemble. "
                    + result.reasoning
                )
            # אם ה-ensemble מתלבט (conf < 0.6), נשתמש במודל העברי כ-tie-breaker חלש יותר
            elif result.confidence < 0.6:
                combined_fake = (
                    0.5 * local_hebrew_output.fake_score
                    + 0.5 * result.fake_probability
                )
                result.fake_probability = combined_fake
                result.classification = "FAKE" if combined_fake >= 0.5 else "REAL"
                result.confidence = max(
                    result.confidence,
                    local_hebrew_output.confidence,
                    0.6
                )
                result.model_used = (result.model_used + " + Hebrew TF-IDF (tie-break)").strip()
                result.reasoning = (
                    "Ensemble was uncertain; Hebrew TF-IDF model used as tie-breaker. "
                    + result.reasoning
                )


        # Compare dataset heuristic vs ensemble result
        # Use dataset if it has higher confidence OR if ensemble is uncertain
        dataset_confidence = dataset_result.get('confidence', 0) if dataset_result else 0
        use_dataset = False
        
        if dataset_result:
            ds_class = dataset_result.get('classification')
            ds_fake = dataset_result.get('fake_probability', 0)
            logger.info(
                f"Dataset result: classification={ds_class}, confidence={dataset_confidence:.3f}, "
                f"fake_prob={ds_fake:.3f}, reasoning={dataset_result.get('reasoning')}"
            )
            logger.info(
                f"Ensemble result: classification={result.classification}, confidence={result.confidence:.3f}, "
                f"fake_prob={result.fake_probability:.3f}, suspicious_score={suspicious_score:.3f}"
            )
            
            # Strong rule: if dataset says FAKE with signal >=0.20, favor it unless ensemble is very confidently REAL
            if ds_class == "FAKE" and dataset_confidence >= 0.20:
                if not (result.classification.startswith("REAL") and result.confidence >= 0.80):
                    use_dataset = True
                    logger.info("✅ Dataset override: FAKE label with sufficient similarity (>=0.20)")
            
            # Strong rule: if dataset says REAL with signal >=0.35 and ensemble not strongly FAKE, favor REAL
            if ds_class == "REAL" and dataset_confidence >= 0.35:
                if result.fake_probability < 0.60 and model2_output.fake_score < 0.35:
                    use_dataset = True
                    logger.info("✅ Dataset override: REAL label with sufficient similarity (>=0.35) and no strong FAKE signal")
            
            # Use dataset if higher confidence than ensemble
            if not use_dataset and dataset_confidence > result.confidence:
                use_dataset = True
                logger.info(f"✅ Dataset-based decision used (confidence {dataset_confidence:.3f}) over ensemble ({result.confidence:.3f})")
            
            # If ensemble is uncertain, lean on dataset signal >0.3
            if not use_dataset and result.confidence < 0.6 and dataset_confidence > 0.3:
                use_dataset = True
                logger.info(f"✅ Dataset-based decision used (confidence {dataset_confidence:.3f}) - ensemble uncertain ({result.confidence:.3f})")
            
            # If dataset has decent confidence, blend with ensemble when both are reasonable
            if not use_dataset and dataset_confidence > 0.5:
                combined_fake = (ds_fake * 0.6) + (result.fake_probability * 0.4)
                combined_confidence = (dataset_confidence * 0.6) + (result.confidence * 0.4)
                if dataset_confidence > 0.7:
                    result.classification = ds_class
                    result.fake_probability = combined_fake
                    result.confidence = combined_confidence
                    result.model_used = "Dataset + Ensemble (weighted)"
                    result.reasoning = f"Dataset match ({dataset_confidence:.2f}) combined with ensemble analysis. {dataset_result['reasoning']}"
                    logger.info(f"✅ Combined dataset ({dataset_confidence:.3f}) + ensemble ({result.confidence:.3f}) = {combined_confidence:.3f}")
            
            if not use_dataset and dataset_confidence > 0:
                logger.info(f"⚠️ Dataset match found but not used: dataset_conf={dataset_confidence:.3f} <= ensemble_conf={result.confidence:.3f}")
        
        if use_dataset:
            result.classification = dataset_result['classification']
            result.fake_probability = dataset_result['fake_probability']
            result.confidence = max(dataset_result['confidence'], result.confidence)
            result.model_used = "Dataset Heuristic"
            # Preserve ensemble reasoning as fallback context
            result.reasoning = f"{dataset_result['reasoning']} | Ensemble: {result.reasoning}"

        # Final tie-break to avoid neutral 50/50 outcomes
        if result.confidence < 0.55:
            logger.info(
                f"Tie-break triggered (conf={result.confidence:.3f}). "
                f"AI detector: fake={model2_output.fake_score:.3f}, real={model2_output.real_score:.3f}; "
                f"suspicious_score={suspicious_score:.3f}"
            )
            # Prefer dataset if any usable signal remains
            if dataset_result and dataset_confidence >= 0.20:
                result.classification = dataset_result['classification']
                result.fake_probability = dataset_result['fake_probability']
                result.confidence = max(result.confidence, dataset_confidence, 0.6)
                result.model_used = "Dataset Heuristic (tie-break)"
                result.reasoning = f"Tie-break: dataset signal used. {dataset_result['reasoning']} | Ensemble: {result.reasoning}"
            else:
                # Fall back to strongest AI detector signal
                if model2_output.fake_score - model2_output.real_score > 0.08:
                    result.classification = 'FAKE'
                    result.fake_probability = max(result.fake_probability, model2_output.fake_score, 0.6)
                    result.confidence = max(result.confidence, abs(model2_output.fake_score - model2_output.real_score), 0.6)
                    result.reasoning = (
                        f"Tie-break: AI detector favors fake ({model2_output.fake_score:.2%} vs "
                        f"{model2_output.real_score:.2%}). Suspicious_score={suspicious_score:.2f}."
                    )
                elif model2_output.real_score - model2_output.fake_score > 0.15 and suspicious_score < 0.5:
                    result.classification = 'REAL'
                    result.fake_probability = min(result.fake_probability, 0.4)
                    result.confidence = max(result.confidence, abs(model2_output.real_score - model2_output.fake_score), 0.6)
                    result.reasoning = (
                        f"Tie-break: AI detector favors real ({model2_output.real_score:.2%} vs "
                        f"{model2_output.fake_score:.2%}) with low suspicious patterns ({suspicious_score:.2f})."
                    )
                else:
                    # If still ambiguous, lean on suspicious patterns
                    if suspicious_score >= 0.4:
                        result.classification = 'FAKE'
                        result.fake_probability = max(result.fake_probability, 0.6)
                        result.confidence = max(result.confidence, 0.6)
                        result.reasoning = (
                            f"Tie-break: suspicious patterns ({suspicious_score:.2f}) push classification to FAKE."
                        )
                    else:
                        result.classification = 'REAL'
                        result.fake_probability = min(result.fake_probability, 0.35)
                        result.confidence = max(result.confidence, 0.55)
                        result.reasoning = (
                            f"Tie-break: low suspicious patterns ({suspicious_score:.2f}) and no strong AI signal."
                        )
        
        bias_fraud_result = None
        if ARCHITECTURE_AVAILABLE and BiasFraudDetector and model_manager.has_bias_fraud_models():
            try:
                bias_fraud_detector = BiasFraudDetector(config)
                bias_fraud_detector.load_models(
                    model_manager.fraud_detector,
                    None  # Bias detector removed - not relevant
                )
                bias_fraud_result = bias_fraud_detector.check(
                    translated_text,
                    model1_output,
                    model2_output
                )
                if bias_fraud_result:
                    logger.info(f"🚨 {bias_fraud_result.classification} detected with score {bias_fraud_result.score:.4f}")
                    # Bias/Fraud detection overrides everything - it indicates non-objective origin
                    result.classification = bias_fraud_result.classification
                    result.confidence = bias_fraud_result.confidence
                    result.fake_probability = bias_fraud_result.score
                    result.model_used = f"{result.model_used} + {bias_fraud_result.model_used}"
                    result.reasoning = f"Non-objective origin detected: {bias_fraud_result.classification} ({bias_fraud_result.score:.2%}). {result.reasoning}"
            except Exception as e:
                logger.warning(f"Bias/Fraud detection error: {e}. Continuing with ensemble result.")
        
        # Convert to API format
        return {
            'classification': result.classification,
            'score': result.confidence,
            'fake_probability': result.fake_probability,
            'model_used': result.model_used,
            'translated_text': translated_text,
            'reasoning': result.reasoning,
            'm1_cg_score': model1_output.fake_score,
            'm1_real_score': model1_output.real_score,
            'm2_generated_score': model2_output.fake_score,
            'm2_real_score': model2_output.real_score,
            'suspicious_score': suspicious_score,
            'suspicious_patterns': {k: v.description for k, v in suspicious_patterns.items()},
            'bias_fraud_detected': bias_fraud_result.classification if bias_fraud_result else None,
            'bias_fraud_score': bias_fraud_result.score if bias_fraud_result else None
        }
    except Exception as e:
        logger.error(f"Ensemble classification error: {e}", exc_info=True)
        return {
            'classification': 'UNCERTAIN',
            'score': 0.5,
            'model_used': 'Error',
            'translated_text': translated_text,
            'error': str(e)
        }


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
