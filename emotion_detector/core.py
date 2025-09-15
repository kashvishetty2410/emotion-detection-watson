"""
core.py - Contains the emotion_predictor function for analyzing text using Watson NLP.
"""

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_cloud_sdk_core.api_exception import ApiException

# Replace with your IBM Watson NLU API key and URL
API_KEY = "YOUR_API_KEY"
API_URL = "YOUR_API_URL"

def _get_nlu_client():
    """Initialize the IBM Watson NLU client."""
    try:
        authenticator = IAMAuthenticator(API_KEY)
        nlu_client = NaturalLanguageUnderstandingV1(
            version="2021-08-01",
            authenticator=authenticator
        )
        nlu_client.set_service_url(API_URL)
        return nlu_client
    except ApiException as e:
        print(f"Error initializing NLU client: {e}")
        return None

def emotion_predictor(text: str):
    """
    Analyze the input text and return emotion scores and dominant emotion.
    
    Args:
        text (str): Text to analyze.
        
    Returns:
        dict: {
            'input_text': str,
            'emotions': dict,
            'dominant_emotion': str
        }
    """
    if not text:
        return {"error": "No text provided for analysis."}

    nlu_client = _get_nlu_client()
    if not nlu_client:
        return {"error": "NLU client not initialized."}

    try:
        response = nlu_client.analyze(
            text=text,
            features=Features(emotion=EmotionOptions())
        ).get_result()

        emotions = response["emotion"]["document"]["emotion"]
        dominant = max(emotions, key=emotions.get)

        return {
            "input_text": text,
            "emotions": emotions,
            "dominant_emotion": dominant
        }
    except ApiException as e:
        return {"error": f"Watson API exception: {e}"}
