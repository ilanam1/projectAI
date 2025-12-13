# Bias and Fraud Detection Module

## Overview

The Bias/Fraud Detection Module extends the fake review detection system to identify **non-AI, human-written but still fraudulent or biased reviews**. This includes:

- **Paid endorsements**: Reviews written by paid reviewers
- **Deceptive commercial text**: Deliberately misleading promotional content
- **Bot-posted spam**: Automated non-LLM synthetic content
- **Biased/Non-objective language**: Reviews with linguistic bias and non-objective framing

## Architecture

### Models Integrated

#### Model A: Fraud/Paid Review Detection
- **Model**: `text-fraud-detection/T5-Base_Fraud_Detection`
- **Purpose**: Detects fraudulent commercial text and paid reviews
- **Threshold**: 0.95 (configurable)
- **Output**: `FRAUD (PAID/DECEPTIVE)` if score > threshold

#### Model B: Bias/Non-Objective Detection
- **Model**: `ayman69/autism-bias-detection-roberta`
- **Purpose**: Identifies linguistic bias and non-objective language patterns
- **Adaptation**: Model output adapted for non-objective review detection
- **Threshold**: 0.90 (configurable)
- **Output**: `HIGHLY BIASED (Non-Objective)` if score > threshold

### Trigger Conditions

The Bias/Fraud Detection Module is executed when:

1. **AI Detection Returns UNCERTAIN**: Model 2 confidence < 0.60
2. **AI Detection Returns High-Confidence REAL**: Model 1 or Model 2 real_score > 0.90

This ensures we catch fraudulent content that may not be AI-generated but is still problematic.

### Execution Flow

```
1. Run Ensemble Classification (Models 1 & 2)
   ↓
2. Check Trigger Conditions
   ↓
3. If triggered:
   a. Run Model A (Fraud Detection)
      - If fraud_score > 0.95 → Return FRAUD
   b. If no fraud, run Model B (Bias Detection)
      - If bias_score > 0.90 → Return BIASED
   c. If neither detected → Continue with ensemble result
```

## Integration

### Configuration

The models are automatically configured in `ml_config.py`:

```python
fraud_detector=ModelConfig(
    name="Fraud Detector",
    model_id="text-fraud-detection/T5-Base_Fraud_Detection",
    device=device
),
bias_detector=ModelConfig(
    name="Bias Detector",
    model_id="ayman69/autism-bias-detection-roberta",
    device=device
)
```

### Thresholds

Configurable in `ml_config.py`:

```python
@dataclass
class Thresholds:
    fraud_detection_threshold: float = 0.95
    bias_detection_threshold: float = 0.90
```

### Environment Variables

```bash
# Adjust fraud detection threshold
ML_FRAUD_DETECTION_THRESHOLD=0.95

# Adjust bias detection threshold
ML_BIAS_DETECTION_THRESHOLD=0.90
```

## Usage

### Automatic Integration

The module is automatically integrated into the classification pipeline. No additional code is required.

### Manual Usage

```python
from bias_fraud_detector import BiasFraudDetector
from ml_config import get_config

config = get_config()
detector = BiasFraudDetector(config)

# Load models (from ModelManager)
detector.load_models(fraud_pipeline, bias_pipeline)

# Check for fraud/bias
result = detector.check(
    text="This product is amazing! Best purchase ever!",
    model1_output=model1_output,
    model2_output=model2_output
)

if result:
    print(f"Detected: {result.classification}")
    print(f"Score: {result.score}")
```

## Output Format

### Classification Types

1. **`FRAUD (PAID/DECEPTIVE)`**: Paid review or deceptive commercial text detected
2. **`HIGHLY BIASED (Non-Objective)`**: Non-objective, biased language detected
3. **`None`**: No fraud or bias detected (continues with ensemble result)

### Result Structure

```python
@dataclass
class BiasFraudResult:
    classification: str  # Classification type
    score: float  # Detection score (0.0-1.0)
    model_used: str  # Which model made the detection
    confidence: float  # Confidence level
```

### API Response

The classification result includes:

```python
{
    'classification': 'FRAUD (PAID/DECEPTIVE)',  # or 'HIGHLY BIASED (Non-Objective)'
    'bias_fraud_detected': 'FRAUD (PAID/DECEPTIVE)',
    'bias_fraud_score': 0.97,
    'model_used': 'Ensemble (Model 1 + Model 2) + T5-Base_Fraud_Detection',
    'reasoning': '... Additionally, FRAUD (PAID/DECEPTIVE) detected (97.00%).'
}
```

## Model Output Parsing

### Fraud Detection (T5-Base_Fraud_Detection)

The model may return different formats. The parser handles:
- List of results with labels like 'FRAUD', 'DECEPTIVE', 'PAID'
- Binary classifiers (fraud = 1 - legitimate)
- Nested list structures

### Bias Detection (autism-bias-detection-roberta)

The model output is adapted for non-objective detection:
- Looks for labels like 'BIASED', 'BIAS', 'NON-OBJECTIVE'
- Interprets high scores as non-objective language
- Handles binary classifiers (bias = 1 - objective)

## Error Handling

The module implements robust error handling:

- **Model Loading Errors**: Logged as warnings, system continues without bias/fraud detection
- **Inference Errors**: Logged, ensemble result used as fallback
- **Parsing Errors**: Default to maximum score if label parsing fails

## Performance Considerations

### Model Loading

- Models are loaded once on startup (singleton pattern)
- First request: ~5-10 seconds (model loading)
- Subsequent requests: ~0.5-1 second (inference only)

### Conditional Execution

The module only runs when:
1. Models are available
2. Trigger conditions are met

This minimizes performance impact.

## Testing

### Test Cases

1. **Paid Review**: Should detect as `FRAUD (PAID/DECEPTIVE)`
2. **Biased Language**: Should detect as `HIGHLY BIASED (Non-Objective)`
3. **Authentic Review**: Should not trigger bias/fraud check
4. **AI-Generated Review**: Should be caught by AI detection, not bias/fraud

### Example Test

```python
from ml_model import classify_review

# Paid review example
result = classify_review("This product is simply amazing! Best purchase ever! Highly recommend!")
# Expected: May trigger fraud detection if score > 0.95

# Biased review example
result = classify_review("This product is terrible, worst thing ever, complete waste of money!")
# Expected: May trigger bias detection if score > 0.90
```

## Monitoring

### Metrics to Track

1. **Fraud Detection Rate**: How often fraud is detected
2. **Bias Detection Rate**: How often bias is detected
3. **False Positive Rate**: Authentic reviews flagged as fraud/bias
4. **Trigger Rate**: How often the module is executed

### Logging

The module provides detailed logging:
- Model loading status
- Trigger condition checks
- Detection results
- Error messages

## Future Enhancements

1. **Model Calibration**: Improve threshold calibration based on feedback
2. **Custom Models**: Train domain-specific fraud/bias detectors
3. **Multi-language Support**: Extend beyond English
4. **Real-time Learning**: Update thresholds based on user feedback

## Troubleshooting

### Models Not Loading

- Check Hugging Face model IDs are correct
- Verify internet connection for model download
- Check device availability (CPU/GPU)

### No Detections

- Verify trigger conditions are met
- Check threshold values (may be too high)
- Review model output parsing logic

### False Positives

- Adjust thresholds in configuration
- Review model output labels
- Consider adding whitelist patterns

