import re


def validate_response(response, customer_message):
    if not response or not response.strip():
        return {
            "safe": False,
            "reason": "The generated response is empty."
        }

    text = response.strip()

    if re.search(r"@\w+", text):
        return {
            "safe": False,
            "reason": "The response contains a Twitter username artifact."
        }

    if re.search(r"https?://|www\.", text):
        return {
            "safe": False,
            "reason": "The response contains an unverified URL."
        }

    if re.search(r"\b(iOS|iPhone|iPad)\s+\d+(\.\d+)+\b", text):
        if not re.search(
            r"\b(iOS|iPhone|iPad)\s+\d+(\.\d+)+\b",
            customer_message,
            re.IGNORECASE
        ):
            return {
                "safe": False,
                "reason": "The response introduces a specific version not provided by the customer."
            }

    return {
        "safe": True,
        "reason": "The generated response passed the basic safety and artifact checks."
    }