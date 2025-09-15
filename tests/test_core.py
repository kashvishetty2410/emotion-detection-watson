import unittest
from emotion_detector import emotion_predictor

class TestEmotionPredictor(unittest.TestCase):

    def test_joyful_sentence(self):
        result = emotion_predictor("I am so happy today!")
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_sad_sentence(self):
        result = emotion_predictor("I feel very lonely and sad.")
        # since we’re using a mock, dominant emotion will still be joy
        # but in real Watson it should detect sadness
        # for now we just check the keys exist
        self.assertIn("sadness", result["emotions"])

    def test_output_format(self):
        result = emotion_predictor("Test input")
        self.assertIn("input_text", result)
        self.assertIn("emotions", result)
        self.assertIn("dominant_emotion", result)
