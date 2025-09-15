"""
server.py - Flask server for the Watson Emotion Detection application.
"""

from flask import Flask, request, jsonify
from emotion_detector.core import emotion_predictor

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle POST requests to /predict and return emotion analysis results.
    
    Expected JSON body:
        {
            "text": "Your text here"
        }
    
    Returns:
        JSON response with emotions or error message.
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Invalid input. Please provide text to analyze."}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be blank."}), 400

    result = emotion_predictor(text)
    if "error" in result:
        return jsonify(result), 400

    return jsonify(result), 200

if __name__ == "__main__":
    app.run(debug=True)
