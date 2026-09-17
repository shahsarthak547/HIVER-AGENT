import pandas as pd

from src.data.conversation import reconstruct_conversations


def test_reconstruct_conversations():
    df = pd.DataFrame({
        "tweet_id": ["1", "2", "3"],
        "in_response_to_tweet_id": [None, "1", "2"],
        "inbound": [True, False, True]
    })

    result = reconstruct_conversations(df)

    assert "conversation_id" in result.columns
    assert len(result) == 3