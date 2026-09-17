from pathlib import Path
import joblib
from src.agent.escalation import decide_escalation
from src.agent.generate_response import generate_response
from src.agent.response_validator import validate_response
from src.retrieval.search import load_retrieval, search
MODEL_PATH = Path("models/apple_intent_classifier.joblib")

class SupportAgentService:
    def __init__(self):
        saved_model = joblib.load(MODEL_PATH)
        self.vectorizer = saved_model["vectorizer"]
        self.classifier = saved_model["model"]
        self.retrieval = load_retrieval()
    def classify(self, message):
        features = self.vectorizer.transform([message])
        intent = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        confidence = float(probabilities.max())
        return intent, confidence
    def retrieve(self, message):
        results = search(message, self.retrieval)
        cases = []
        for _, row in results.iterrows():
            cases.append(
                {
                    "customer_text": row["customer_text"],
                    "support_text": row["support_text"],
                    "similarity": float(row["similarity"]),
                }
            )
        return cases
    def process(self, message):
        intent, confidence = self.classify(message)
        cases = self.retrieve(message)
        escalation = decide_escalation(
            intent=intent,
            classifier_confidence=confidence,
            retrieved_cases=cases,
            customer_message=message,
        )
        result = None
        if not escalation["escalate"]:
            result = generate_response(
                customer_message=message,
                intent=intent,
                retrieved_cases=cases,
            )
            validation = validate_response(
                response=result,
                customer_message=message,
            )
            if not validation["safe"]:
                escalation = {
                    "escalate": True,
                    "reason": validation["reason"],
                }
                result = None
        return {
            "customer_message": message,
            "intent": intent,
            "classifier_confidence": confidence,
            "top_retrieval_similarity": (
                cases[0]["similarity"] if cases else None
            ),
            "decision": (
                "human_escalation"
                if escalation["escalate"]
                else "auto_handle"
            ),
            "decision_reason": escalation["reason"],
            "response": result,
        }

agent_service = SupportAgentService()