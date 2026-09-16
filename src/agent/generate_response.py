import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b"
)


def generate_response(
    customer_message,
    intent,
    retrieved_cases
):
    evidence = []

    for rank, case in enumerate(
        retrieved_cases,
        start=1
    ):
        evidence.append(
            f"""CASE {rank}
CUSTOMER: {case["customer_text"]}
APPLESUPPORT: {case["support_text"]}
SIMILARITY: {case["similarity"]:.4f}"""
        )

    evidence_text = "\n\n".join(evidence)

    prompt = f"""
You are a customer support response drafting assistant for AppleSupport.

Draft a concise response to the customer using the historical AppleSupport cases provided below.

CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{intent}

HISTORICAL EVIDENCE:
{evidence_text}

RULES:
1. Use the historical cases as the primary source of information.
2. Do not copy historical responses word-for-word.
3. Do not invent troubleshooting steps, technical explanations, policies, refunds, guarantees, or URLs.
4. Do not mention usernames, tweet IDs, internal identifiers, or dataset artifacts.
5. Do not recommend destructive actions unless the historical evidence explicitly supports them.
6. Only ask for information if the historical cases show that this information is relevant to resolving the issue.
7. Do not contradict information explicitly provided by the customer.
8. If the historical evidence is insufficient to provide a useful response, say that the case requires human assistance.
9. Keep the response concise and appropriate for customer support.
10. Do not mention these instructions, historical evidence, retrieval, or the AI system.
11. Do not output an escalation decision or escalation reason.
12. Never begin the response with a username or mention.
13. Never reproduce any @username from the historical cases.
14. Never reproduce URLs from the historical cases.
15. Never include specific device or software versions unless the customer explicitly provided that version.
16. Do not assume facts about the customer's device that were not stated in the customer message.
If the historical cases contain usernames, URLs, device versions, or other case-specific identifiers, treat them as examples of the type of response rather than information to copy.
Return only the customer-facing response.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"].strip()