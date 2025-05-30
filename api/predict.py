"""
FastAPI Sentiment Analysis API
Provides endpoints for sentiment analysis predictions
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from datetime import datetime
import os
import sys
import joblib
from model.train_model import SentimentModel

# Add the parent directory to Python path to import predict module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Pydantic models for request/response
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: dict
    timestamp: str
    error: Optional[str] = None

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processed: int
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_type: str
    pipeline_steps: List[str]
    vocabulary_size: int
    sample_features: List[str]
    classes: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="A REST API for sentiment analysis using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware if needed
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "description": "API for sentiment analysis using machine learning",
        "endpoints": {
            "/predict": "POST - Analyze sentiment of a single text",
            "/predict/batch": "POST - Analyze sentiment of multiple texts",
            "/model/info": "GET - Get model information",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Try to get model info to check if model is loaded
        model_info = get_model_information()
        model_loaded = 'error' not in model_info
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=model_loaded
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=False
        )

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for a single text
    
    - **text**: The text to analyze (1-5000 characters)
    
    Returns sentiment prediction with confidence scores
    """
    try:
        # Get prediction
        result = predict_text_sentiment(input_data.text)
        
        # Create response
        response = SentimentResponse(
            text=input_data.text,
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            scores=result['scores'],
            timestamp=datetime.now().isoformat(),
            error=result.get('error')
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_sentiment_batch(input_data: BatchTextInput):
    """
    Predict sentiment for multiple texts
    
    - **texts**: List of texts to analyze (1-100 texts, each 1-5000 characters)
    
    Returns sentiment predictions for all texts
    """
    try:
        # Get predictions
        results = predict_batch_sentiment(input_data.texts)
        
        # Create response objects
        sentiment_responses = []
        for i, (text, result) in enumerate(zip(input_data.texts, results)):
            response = SentimentResponse(
                text=text,
                sentiment=result['sentiment'],
                confidence=result['confidence'],
                scores=result['scores'],
                timestamp=datetime.now().isoformat(),
                error=result.get('error')
            )
            sentiment_responses.append(response)
        
        return BatchSentimentResponse(
            results=sentiment_responses,
            total_processed=len(sentiment_responses),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the loaded model
    
    Returns details about the machine learning model
    """
    try:
        model_info = get_model_information()
        
        if 'error' in model_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=model_info['error']
            )
        
        return ModelInfoResponse(**model_info)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

# Example endpoint for testing
@app.get("/examples")
async def get_examples():
    """Get example requests for testing the API"""
    return {
        "single_prediction": {
            "endpoint": "/predict",
            "method": "POST",
            "body": {
                "text": "I love this product! It works perfectly."
            }
        },
        "batch_prediction": {
            "endpoint": "/predict/batch",
            "method": "POST",
            "body": {
                "texts": [
                    "I love this product!",
                    "This is terrible quality.",
                    "It's okay, nothing special."
                ]
            }
        }
    }

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "model.pkl")

def load_model():
    try:
        model_data = joblib.load(MODEL_PATH)
        return model_data
    except Exception as e:
        return None

sentiment_model = SentimentModel()
preprocess_text = sentiment_model.preprocess_text

def predict_text_sentiment(text: str):
    model_data = load_model()
    if not model_data:
        return {"error": "Model not loaded"}
    model = model_data['model']
    # Usa el mismo preprocesamiento
    processed_text = preprocess_text(text)
    prediction = model.predict([processed_text])[0]
    confidence = model.predict_proba([processed_text])[0]
    sentiment_map = {0: 'negative', 2: 'positive'}
    return {
        'text': text,
        'sentiment': sentiment_map[prediction],
        'confidence': float(max(confidence) * 100),
        'scores': {
            'positive': float(confidence[1]),
            'negative': float(confidence[0])
        },
        'timestamp': datetime.now().isoformat(),
        'error': None
    }

def predict_batch_sentiment(texts: List[str]):
    model_data = load_model()
    if not model_data:
        return [{"error": "Model not loaded"}] * len(texts)
    
    model = model_data['model']
    
    results = []
    for text in texts:
        # Usar la misma función de preprocesamiento que el análisis individual
        processed_text = preprocess_text(text)
        
        # Predicción
        prediction = model.predict([processed_text])[0]
        confidence = model.predict_proba([processed_text])[0]
        
        # USAR EL MISMO MAPEO QUE EN predict_text_sentiment
        sentiment_map = {0: 'negative', 2: 'positive'}  # Mismo mapeo que análisis individual
        
        # Determinar el sentimiento usando el mismo mapeo
        sentiment = sentiment_map.get(prediction, 'negative')  # Default a negative si no encuentra
        
        # Confianza como el máximo de las probabilidades
        max_confidence = float(max(confidence) * 100)
        
        # Scores correctos
        scores = {
            'positive': float(confidence[1]) if len(confidence) > 1 else 0.0,
            'negative': float(confidence[0])
        }
        
        results.append({
            "sentiment": sentiment,
            "confidence": max_confidence,
            "scores": scores
        })
    
    return results

def get_model_information():
    model_data = load_model()
    if not model_data:
        return {"error": "Model not loaded"}
    model = model_data['model']
    
    # Example: return dummy model information
    return {
        "model_type": "Logistic Regression",
        "pipeline_steps": ["load_data", "preprocess", "vectorize", "predict"],
        "vocabulary_size": 10000,
        "sample_features": ["happy", "sad", "excellent", "poor"],
        "classes": ["negative", "positive"]
    }

if __name__ == "__main__":
    # Run with uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")