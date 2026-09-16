id="gq3m1p"
from typing import Any


RISK_PATTERNS = [
    "refund",
    "chargeback",
    "charged twice",
    "charged me twice",
    "cancel my order",
    "cancel subscription",
    "delete my account",
    "erase my iphone",
    "factory reset",
    "legal action",
    "lawsuit",
    "police",
    "stolen",
    "hacked",
    "unauthorized purchase",
]


def contains_risky_request(message: str) -> bool:
    text = message.lower()

    return any(
        pattern in text
        for pattern in RISK_PATTERNS
    )


def decide_escalation(
    intent,
    classifier_confidence,
    retrieved_cases,
    customer_message="",
    min_classifier_confidence=0.80,
    min_retrieval_similarity=0.35
) -> dict[str, Any]:

    if intent == "other_unclear":
        return {
            "escalate": True,
            "reason": "The message could not be mapped to a supported intent."
        }

    if classifier_confidence < min_classifier_confidence:
        return {
            "escalate": True,
            "reason": "Classifier confidence is below the safe handling threshold."
        }

    if not retrieved_cases:
        return {
            "escalate": True,
            "reason": "No relevant historical support cases were retrieved."
        }

    top_similarity = retrieved_cases[0]["similarity"]

    if top_similarity < min_retrieval_similarity:
        return {
            "escalate": True,
            "reason": "Retrieved historical cases are not sufficiently similar."
        }

    if contains_risky_request(customer_message):
        return {
            "escalate": True,
            "reason": "The request involves an action that requires human review."
        }

    return {
        "escalate": False,
        "reason": "The intent is confident, relevant historical evidence is available, and no human-review trigger was detected."
    }