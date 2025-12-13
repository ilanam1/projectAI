"""
Production Validation Tests for Fake Review Detection System.

This module contains comprehensive validation tests to ensure:
1. Zero false high-confidence classifications
2. Proper edge case handling
3. Confidence calibration correctness
4. System stability

Run with: python validation_tests.py
"""

import sys
import traceback
from typing import Dict, Any

# Import the classification function
try:
    from ml_model import classify_review, detect_fake_review
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you're running from the backend directory")
    IMPORTS_AVAILABLE = False


class ValidationResults:
    """Track validation test results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.test_cases = []
    
    def add_test(self, name: str, passed: bool, message: str = "", warning: bool = False):
        """Add a test result."""
        self.test_cases.append({
            'name': name,
            'passed': passed,
            'message': message,
            'warning': warning
        })
        if warning:
            self.warnings += 1
        elif passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⚠️  Warnings: {self.warnings}")
        print(f"📊 Total: {len(self.test_cases)}")
        print("="*80)
        
        if self.failed > 0:
            print("\n❌ FAILED TESTS:")
            for test in self.test_cases:
                if not test['passed'] and not test['warning']:
                    print(f"  - {test['name']}: {test['message']}")
        
        if self.warnings > 0:
            print("\n⚠️  WARNINGS:")
            for test in self.test_cases:
                if test['warning']:
                    print(f"  - {test['name']}: {test['message']}")
        
        print("\n" + "="*80)
        if self.failed == 0:
            print("✅ ALL CRITICAL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
        else:
            print("❌ SOME TESTS FAILED - REVIEW BEFORE DEPLOYMENT")
        print("="*80)


def validate_confidence_ranges(results: ValidationResults):
    """Validate that all confidence values are in [0.0, 1.0] range."""
    print("\n🔍 Testing: Confidence Range Validation")
    
    test_cases = [
        "This is a test review.",
        "המוצר הזה פשוט מושלם!",
        "The product arrived very quickly, works smoothly, looks good and there were no problems. Simply perfect!",
        "I bought this product last week and it's been working great. The quality is excellent and I would definitely recommend it to others.",
    ]
    
    for i, text in enumerate(test_cases, 1):
        try:
            result = classify_review(text)
            
            # Check confidence
            confidence = result.get('score', 0.0)
            if not (0.0 <= confidence <= 1.0):
                results.add_test(
                    f"Confidence Range Test {i}",
                    False,
                    f"Confidence {confidence} out of range [0.0, 1.0]"
                )
            else:
                results.add_test(
                    f"Confidence Range Test {i}",
                    True,
                    f"Confidence {confidence:.4f} in valid range"
                )
            
            # Check fake_probability
            fake_prob = result.get('fake_probability', 0.5)
            if not (0.0 <= fake_prob <= 1.0):
                results.add_test(
                    f"Fake Probability Range Test {i}",
                    False,
                    f"Fake probability {fake_prob} out of range [0.0, 1.0]"
                )
            else:
                results.add_test(
                    f"Fake Probability Range Test {i}",
                    True,
                    f"Fake probability {fake_prob:.4f} in valid range"
                )
                
        except Exception as e:
            results.add_test(
                f"Confidence Range Test {i}",
                False,
                f"Exception: {str(e)}"
            )


def validate_edge_cases(results: ValidationResults):
    """Validate edge case handling."""
    print("\n🔍 Testing: Edge Case Handling")
    
    # Test empty string
    try:
        result = classify_review("")
        if result.get('classification') == 'UNCERTAIN':
            results.add_test("Empty String", True, "Correctly handles empty string")
        else:
            results.add_test("Empty String", False, f"Unexpected classification: {result.get('classification')}")
    except ValueError:
        results.add_test("Empty String", True, "Correctly raises ValueError for empty string")
    except Exception as e:
        results.add_test("Empty String", False, f"Unexpected exception: {str(e)}")
    
    # Test whitespace-only
    try:
        result = classify_review("   \n\t   ")
        if result.get('classification') == 'UNCERTAIN':
            results.add_test("Whitespace Only", True, "Correctly handles whitespace-only string")
        else:
            results.add_test("Whitespace Only", False, f"Unexpected classification: {result.get('classification')}")
    except ValueError:
        results.add_test("Whitespace Only", True, "Correctly raises ValueError for whitespace-only string")
    except Exception as e:
        results.add_test("Whitespace Only", False, f"Unexpected exception: {str(e)}")
    
    # Test None (should raise ValueError)
    try:
        result = classify_review(None)
        results.add_test("None Input", False, "Should raise ValueError for None input")
    except ValueError:
        results.add_test("None Input", True, "Correctly raises ValueError for None")
    except Exception as e:
        results.add_test("None Input", False, f"Unexpected exception type: {type(e).__name__}")


def validate_confidence_calibration(results: ValidationResults):
    """Validate confidence calibration prevents false high-confidence."""
    print("\n🔍 Testing: Confidence Calibration")
    
    # ChatGPT-style review (should not have high confidence in REAL)
    chatgpt_review = "The product arrived very quickly, works smoothly, looks good and there were no problems. Simply perfect!"
    
    try:
        result = classify_review(chatgpt_review)
        confidence = result.get('score', 0.0)
        classification = result.get('classification', '')
        suspicious_score = result.get('suspicious_score', 0.0)
        
        # If suspicious patterns exist and classification is REAL, confidence should be capped
        if suspicious_score > 0.6 and 'REAL' in classification:
            if confidence > 0.7:
                results.add_test(
                    "Confidence Calibration - ChatGPT Review",
                    False,
                    f"High confidence {confidence:.4f} in REAL despite suspicious patterns {suspicious_score:.2f}"
                )
            else:
                results.add_test(
                    "Confidence Calibration - ChatGPT Review",
                    True,
                    f"Confidence properly capped at {confidence:.4f} despite suspicious patterns"
                )
        elif 'FAKE' in classification:
            results.add_test(
                "Confidence Calibration - ChatGPT Review",
                True,
                f"Correctly classified as {classification} with confidence {confidence:.4f}"
            )
        else:
            results.add_test(
                "Confidence Calibration - ChatGPT Review",
                True,
                f"Classification: {classification}, Confidence: {confidence:.4f}"
            )
    except Exception as e:
        results.add_test(
            "Confidence Calibration - ChatGPT Review",
            False,
            f"Exception: {str(e)}"
        )


def validate_error_handling(results: ValidationResults):
    """Validate error handling and stability."""
    print("\n🔍 Testing: Error Handling")
    
    # Test that system never crashes
    test_cases = [
        "Normal review text",
        "Very long text " * 100,  # Very long text
        "Special chars: !@#$%^&*()",
        "Unicode: 你好 مرحبا שלום",
    ]
    
    for i, text in enumerate(test_cases, 1):
        try:
            result = classify_review(text)
            # If we get here, system didn't crash
            if 'error' in result and result['error']:
                results.add_test(
                    f"Error Handling Test {i}",
                    True,
                    f"Gracefully handled error: {result['error']}",
                    warning=True
                )
            else:
                results.add_test(
                    f"Error Handling Test {i}",
                    True,
                    "System handled input without crashing"
                )
        except Exception as e:
            results.add_test(
                f"Error Handling Test {i}",
                False,
                f"System crashed with exception: {str(e)}"
            )


def validate_output_structure(results: ValidationResults):
    """Validate output structure and required fields."""
    print("\n🔍 Testing: Output Structure")
    
    try:
        result = classify_review("This is a test review.")
        
        required_fields = [
            'classification',
            'score',
            'fake_probability',
            'model_used',
            'translated_text',
            'reasoning'
        ]
        
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            results.add_test(
                "Output Structure",
                False,
                f"Missing required fields: {missing_fields}"
            )
        else:
            results.add_test(
                "Output Structure",
                True,
                "All required fields present"
            )
        
        # Validate field types
        if not isinstance(result.get('classification'), str):
            results.add_test("Output Type - classification", False, "classification must be string")
        else:
            results.add_test("Output Type - classification", True, "classification is string")
        
        if not isinstance(result.get('score'), (int, float)):
            results.add_test("Output Type - score", False, "score must be numeric")
        else:
            results.add_test("Output Type - score", True, "score is numeric")
            
    except Exception as e:
        results.add_test(
            "Output Structure",
            False,
            f"Exception: {str(e)}"
        )


def main():
    """Run all validation tests."""
    print("="*80)
    print("PRODUCTION VALIDATION TESTS")
    print("Fake Review Detection System")
    print("="*80)
    
    if not IMPORTS_AVAILABLE:
        print("\n❌ Cannot run tests - imports failed")
        return 1
    
    results = ValidationResults()
    
    # Run all validation tests
    validate_confidence_ranges(results)
    validate_edge_cases(results)
    validate_confidence_calibration(results)
    validate_error_handling(results)
    validate_output_structure(results)
    
    # Print summary
    results.print_summary()
    
    # Return exit code
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)

