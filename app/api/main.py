from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.schemas import (ClassificationResponse,RetrievalCase,RetrievalResponse,SupportRequest,SupportResponse,)
from app.api.service import agent_service
app = FastAPI(
    title="Hiver AI Support Agent",
    description="AI-powered AppleSupport customer support agent",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "hiver-ai-support-agent",
    }
@app.get("/api/v1/intents")
def intents():
    return {
        "intents": [
            "battery_charging",
            "keyboard_text_input",
            "connectivity",
            "apple_id_account",
            "messaging",
            "apple_app_service",
            "photos_media",
            "purchase_payment_order",
            "software_update",
            "device_performance",
        ]
    }
@app.post(
    "/api/v1/classify",
    response_model=ClassificationResponse,
)
def classify(request: SupportRequest):
    try:
        intent, confidence = agent_service.classify(request.message)

        return ClassificationResponse(
            message=request.message,
            intent=intent,
            confidence=confidence,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
@app.post(
    "/api/v1/retrieve",
    response_model=RetrievalResponse,
)
def retrieve(request: SupportRequest):
    try:
        cases = agent_service.retrieve(request.message)
        return RetrievalResponse(
            message=request.message,
            cases=[
                RetrievalCase(**case)
                for case in cases
            ],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
@app.post(
    "/api/v1/support",
    response_model=SupportResponse,
)
def support(request: SupportRequest):
    try:
        result = agent_service.process(request.message)
        return SupportResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )