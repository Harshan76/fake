import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

SENSATIONAL_WORDS = {
    "shocking", "unbelievable", "secret", "miracle", "exposed", "banned",
    "cure", "hidden", "conspiracy", "hoax", "viral", "alert", "urgent",
    "breaking news", "forward to everyone", "share immediately", "dont let them hide",
    "they are lying", "illuminati", "coverup", "100% cure", "free money", "won lottery"
}

JOURNALISTIC_WORDS = {
    "according to", "reuters", "associated press", "spokesperson", "statement",
    "officials said", "ministry", "department of", "confirmed", "reported",
    "published in", "investigation", "study indicates", "evidence suggests",
    "in an interview", "press release", "bureau", "correspondent"
}

class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts stylometric, syntactic, and sensationalism features
    from text to distinguish clickbait/misinformation from journalistic text.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            if not isinstance(text, str):
                text = str(text)
            
            words = text.split()
            word_count = max(len(words), 1)
            char_count = max(len(text), 1)

            # 1. Capitalization ratio
            caps_count = sum(1 for c in text if c.isupper())
            caps_ratio = caps_count / char_count

            # 2. Punctuation intensity
            exclamations = text.count("!") / word_count
            questions = text.count("?") / word_count
            multiple_punct = len(re.findall(r'[!?]{2,}', text)) / word_count

            # 3. Sensationalism density
            text_lower = text.lower()
            sensational_hits = sum(1 for word in SENSATIONAL_WORDS if word in text_lower)
            sensational_density = sensational_hits / (word_count / 10 + 1)

            # 4. Journalistic attribution density
            journalistic_hits = sum(1 for word in JOURNALISTIC_WORDS if word in text_lower)
            journalistic_density = journalistic_hits / (word_count / 10 + 1)

            # 5. Numerical density
            num_tokens = len(re.findall(r'\b\d+\b', text)) / word_count

            # 6. Average word length
            avg_word_len = sum(len(w) for w in words) / word_count

            features.append([
                caps_ratio,
                exclamations,
                questions,
                multiple_punct,
                sensational_density,
                journalistic_density,
                num_tokens,
                avg_word_len
            ])
            
        return np.array(features)
