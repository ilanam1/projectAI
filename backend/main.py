from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from ml_model import classify_review

app = FastAPI(
    title="Fake Review Detection API",
    description="API for detecting fake reviews using Machine Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The review text to analyze (Hebrew or English)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "אני ממש אוהב את המוצר הזה, הוא עבד בדיוק כמו שציפיתי!"
            }
        }


class ReviewResponse(BaseModel):
    classification: str = Field(..., description="Classification: 'Real' or 'Fake' or 'UNCERTAIN'")
    is_fake: bool = Field(..., description="Whether the review is detected as fake")
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Confidence score (0-100%)")
    fake_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of being fake (0.0 to 1.0)")
    explanation: str = Field(..., description="Concise rationale for classification")

    class Config:
        json_schema_extra = {
            "example": {
                "classification": "Real",
                "is_fake": False,
                "confidence_score": 95.0,
                "fake_probability": 0.15,
                "explanation": "Specific personal experience with natural Hebrew phrasing and clear contextual detail."
            }
        }


class MultipleReviewsRequest(BaseModel):
    reviews: List[str] = Field(..., min_items=1, description="List of review texts to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "reviews": [
                    "אני ממש אוהב את המוצר הזה, הוא עבד בדיוק כמו שציפיתי!",
                    "מוצר מדהים! מומלץ לכולם! איכות גבוהה מאוד, תצטרכו אותו!"
                ]
            }
        }


class MultipleReviewsResponse(BaseModel):
    total_reviews: int = Field(..., description="Total number of reviews analyzed")
    real_count: int = Field(..., description="Number of real reviews")
    fake_count: int = Field(..., description="Number of fake reviews")
    uncertain_count: int = Field(..., description="Number of uncertain reviews")
    real_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of real reviews")
    fake_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of fake reviews")
    recommendation: str = Field(..., description="Overall purchase recommendation")
    individual_results: List[ReviewResponse] = Field(..., description="Individual review analysis results")
    flagged_for_review: List[int] = Field(..., description="Indices of reviews flagged for human review (uncertain cases)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_reviews": 2,
                "real_count": 1,
                "fake_count": 1,
                "uncertain_count": 0,
                "real_percentage": 50.0,
                "fake_percentage": 50.0,
                "recommendation": "Caution: 50% of reviews appear to be fake. Verify authenticity before purchase.",
                "individual_results": [],
                "flagged_for_review": []
            }
        }


@app.get("/")
async def root():
    return {
        "message": "Enterprise-Grade Hebrew Review AI Detection API",
        "version": "2.0.0",
        "description": "Analyze Hebrew product reviews and classify each as Real (human-written) or Fake/AI-generated",
        "endpoints": {
            "/predict": "POST - Analyze a single review text",
            "/predict/batch": "POST - Analyze multiple reviews with aggregated analysis"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict/batch", response_model=MultipleReviewsResponse)
async def predict_multiple_reviews(request: MultipleReviewsRequest):
    """
    Analyze multiple Hebrew product reviews and provide aggregated analysis.
    
    This endpoint:
    - Analyzes each review individually
    - Provides aggregate statistics (percentage of real vs. fake)
    - Generates overall purchase recommendation
    - Flags ambiguous cases for human review
    
    Args:
        request: MultipleReviewsRequest containing list of review texts
        
    Returns:
        MultipleReviewsResponse with aggregated statistics and individual results
    """
    try:
        individual_results = []
        real_count = 0
        fake_count = 0
        uncertain_count = 0
        flagged_indices = []
        
        # Analyze each review
        for idx, review_text in enumerate(request.reviews):
            try:
                result = classify_review(review_text)
                
                # Extract classification
                classification = result.get('classification', 'UNCERTAIN')
                if classification == 'REAL':
                    classification = 'Real'
                    real_count += 1
                elif classification == 'FAKE' or classification.startswith('FAKE'):
                    classification = 'Fake'
                    fake_count += 1
                else:
                    classification = 'UNCERTAIN'
                    uncertain_count += 1
                    flagged_indices.append(idx)
                
                # Extract scores
                fake_probability = result.get('fake_probability', 0.5)
                confidence = result.get('score', 0.5)
                confidence_score = confidence * 100.0
                
                # Get explanation
                explanation = result.get('reasoning', 'Unable to determine classification.')
                if not explanation or explanation == 'Unable to determine classification.':
                    if classification == 'Real':
                        explanation = "Natural Hebrew phrasing with authentic personal experience and contextual details."
                    elif classification == 'Fake':
                        explanation = "Generic, repetitive phrasing without personal detail; resembles AI-generated promotional content."
                    else:
                        explanation = "Ambiguous case requiring human review."
                
                is_fake = classification == 'Fake' or fake_probability >= 0.5
                
                individual_results.append(ReviewResponse(
                    classification=classification,
                    is_fake=is_fake,
                    confidence_score=confidence_score,
                    fake_probability=fake_probability,
                    explanation=explanation
                ))
                
            except Exception as e:
                # Handle individual review errors
                uncertain_count += 1
                flagged_indices.append(idx)
                individual_results.append(ReviewResponse(
                    classification="UNCERTAIN",
                    is_fake=False,
                    confidence_score=0.0,
                    fake_probability=0.5,
                    explanation=f"Error analyzing review: {str(e)}"
                ))
        
        # Calculate percentages
        total = len(request.reviews)
        real_percentage = (real_count / total * 100.0) if total > 0 else 0.0
        fake_percentage = (fake_count / total * 100.0) if total > 0 else 0.0
        
        # Generate recommendation
        if fake_percentage >= 70:
            recommendation = f"⚠️ High Risk: {fake_percentage:.1f}% of reviews appear to be fake. Strongly recommend verifying authenticity before purchase."
        elif fake_percentage >= 50:
            recommendation = f"Caution: {fake_percentage:.1f}% of reviews appear to be fake. Verify authenticity before purchase."
        elif fake_percentage >= 30:
            recommendation = f"Moderate Trust: {real_percentage:.1f}% of reviews appear authentic, but {fake_percentage:.1f}% may be fake. Exercise caution."
        elif real_percentage >= 70:
            recommendation = f"✅ High Trust: {real_percentage:.1f}% of reviews appear authentic. Product likely reliable."
        else:
            recommendation = f"Mixed Signals: {real_percentage:.1f}% real, {fake_percentage:.1f}% fake. Review individual assessments carefully."
        
        if uncertain_count > 0:
            recommendation += f" {uncertain_count} review(s) flagged for human review."
        
        return MultipleReviewsResponse(
            total_reviews=total,
            real_count=real_count,
            fake_count=fake_count,
            uncertain_count=uncertain_count,
            real_percentage=real_percentage,
            fake_percentage=fake_percentage,
            recommendation=recommendation,
            individual_results=individual_results,
            flagged_for_review=flagged_indices
        )
    
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in predict_multiple_reviews: {e}", exc_info=True)
        # Return empty result on error
        return MultipleReviewsResponse(
            total_reviews=0,
            real_count=0,
            fake_count=0,
            uncertain_count=0,
            real_percentage=0.0,
            fake_percentage=0.0,
            recommendation="Error processing batch. Please try again.",
            individual_results=[],
            flagged_for_review=[]
        )


@app.post("/predict", response_model=ReviewResponse)
async def predict_review(review: ReviewRequest):
    """
    Analyze a Hebrew product review and classify it as Real or Fake/AI-generated.
    
    Enterprise-grade fake review detection with:
    - Native Hebrew pattern detection (no translation required for patterns)
    - Multi-model ensemble classification
    - Confidence calibration to prevent false high-confidence
    - Comprehensive error handling
    - Detailed explanations
    
    Args:
        review: ReviewRequest containing the review text in Hebrew (validated by Pydantic)
        
    Returns:
        ReviewResponse with classification, confidence score (0-100%), and detailed explanation.
        All values are guaranteed to be in valid ranges.
        
    Raises:
        HTTPException: Only for invalid input (handled by Pydantic)
        Note: Model errors return UNCERTAIN classification, never crash
    """
    try:
        result = classify_review(review.text)
        
        # Extract classification
        classification = result.get('classification', 'UNCERTAIN')
        if classification == 'REAL':
            classification = 'Real'
        elif classification == 'FAKE' or classification.startswith('FAKE'):
            classification = 'Fake'
        else:
            classification = 'UNCERTAIN'
        
        # Extract scores
        fake_probability = result.get('fake_probability', 0.5)
        confidence = result.get('score', 0.5)
        
        # Convert confidence to percentage (0-100)
        confidence_score = confidence * 100.0
        
        # Get explanation
        explanation = result.get('reasoning', 'Unable to determine classification.')
        
        # Format explanation to be more concise and professional
        if not explanation or explanation == 'Unable to determine classification.':
            if classification == 'Real':
                explanation = "Natural Hebrew phrasing with authentic personal experience and contextual details."
            elif classification == 'Fake':
                explanation = "Generic, repetitive phrasing without personal detail; resembles AI-generated promotional content."
            else:
                explanation = "Ambiguous case requiring human review."
        
        is_fake = classification == 'Fake' or fake_probability >= 0.5
        
        return ReviewResponse(
            classification=classification,
            is_fake=is_fake,
            confidence_score=confidence_score,
            fake_probability=fake_probability,
            explanation=explanation
        )
    
    except ValueError as e:
        return ReviewResponse(
            classification="UNCERTAIN",
            is_fake=False,
            confidence_score=0.0,
            fake_probability=0.5,
            explanation=f"Input validation error: {str(e)}"
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in predict_review: {e}", exc_info=True)
        return ReviewResponse(
            classification="UNCERTAIN",
            is_fake=False,
            confidence_score=0.0,
            fake_probability=0.5,
            explanation="System error occurred. Please try again or flag for human review."
        )

