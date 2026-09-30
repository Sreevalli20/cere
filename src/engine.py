"""
Tone Analysis Engine for ToneTrace
Estimates sentiment, emotion, sarcasm, irony, passive-aggression, and hostility
Uses lightweight feature-based rules, no external models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .features import FeatureExtractor


class ToneEngine:
    """Estimate tone signals from linguistic features"""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def _get_scalar(self, value):
        """Extract scalar value from potentially Series object"""
        if isinstance(value, pd.Series):
            return value.iloc[0] if len(value) > 0 else 0
        return value

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze conversation and add tone estimates
        Returns DataFrame with tone columns added
        """
        # Extract features
        df = self.feature_extractor.extract_features(df)

        # Calculate tone signals for each message
        tone_results = []

        for idx, row in df.iterrows():
            tone_dict = {
                'sentiment': self._estimate_sentiment(row),
                'sentiment_score': self._estimate_sentiment_score(row),
                'joy': self._estimate_emotion(row, 'joy'),
                'sadness': self._estimate_emotion(row, 'sadness'),
                'anger': self._estimate_emotion(row, 'anger'),
                'fear': self._estimate_emotion(row, 'fear'),
                'frustration': self._estimate_emotion(row, 'frustration'),
                'sarcasm': self._estimate_sarcasm(row),
                'irony': self._estimate_irony(row),
                'passive_aggression': self._estimate_passive_aggression(row),
                'hostility': self._estimate_hostility(row),
            }
            tone_results.append(tone_dict)

        tone_df = pd.DataFrame(tone_results)

        # Combine with original data
        result = pd.concat([df.reset_index(drop=True), tone_df], axis=1)
        return result

    def _estimate_sentiment(self, row: pd.Series) -> str:
        """Estimate sentiment category: positive, neutral, negative"""
        score = self._estimate_sentiment_score(row)

        if score > 0.2:
            return 'positive'
        elif score < -0.2:
            return 'negative'
        else:
            return 'neutral'

    def _estimate_sentiment_score(self, row: pd.Series) -> float:
        """
        Estimate continuous sentiment score from -1 to +1
        Based on lexical balance, negation, intensity, and punctuation
        """
        # Base sentiment from lexical balance
        positive = self._get_scalar(row['positive_lexical'])
        negative = self._get_scalar(row['negative_lexical'])

        if positive + negative == 0:
            base_score = 0.0
        else:
            base_score = (positive - negative) / (positive + negative)

        # Apply negation penalty
        negation_factor = 1.0
        if self._get_scalar(row['negation_markers']) > 0:
            negation_factor = -0.6

        # Apply intensifier
        intensifier_factor = 1.0 + (self._get_scalar(row['intensifiers']) * 0.2)

        # Punctuation influence
        punctuation_factor = 1.0
        if self._get_scalar(row['exclamation_count']) > 0:
            punctuation_factor = 1.1
        if self._get_scalar(row['question_count']) > 0:
            punctuation_factor = 0.9

        # Combine
        score = base_score * negation_factor * intensifier_factor * punctuation_factor

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))

    def _estimate_emotion(self, row: pd.Series, emotion: str) -> float:
        """
        Estimate emotion signal intensity (0 to 1)
        Based on lexical cues and context
        """
        emotion_lexical_map = {
            'joy': 'joy_lexical',
            'sadness': 'sadness_lexical',
            'anger': 'anger_lexical',
            'fear': 'fear_lexical',
            'frustration': 'frustration_lexical'
        }

        lexical_col = emotion_lexical_map.get(emotion, '')
        if lexical_col not in row:
            return 0.0

        # Base from lexical cues
        base = self._get_scalar(row[lexical_col])

        # Normalize by message length (longer messages with same cue count = less intense)
        word_count = max(1, self._get_scalar(row['word_count']))
        normalized = base / word_count

        # Scale to 0-1 range
        intensity = min(1.0, normalized * 5.0)

        # Boost from punctuation
        if self._get_scalar(row['exclamation_count']) > 0:
            intensity = min(1.0, intensity * 1.3)

        # Boost from uppercase
        if self._get_scalar(row['uppercase_ratio']) > 0.3:
            intensity = min(1.0, intensity * 1.2)

        return intensity

    def _estimate_sarcasm(self, row: pd.Series) -> float:
        """
        Estimate sarcasm signal (0 to 1)
        Based on positive words in negative context, cues, and patterns
        """
        # Direct sarcasm cues
        cue_score = min(1.0, self._get_scalar(row['sarcasm_cues']) * 0.4)

        # Pattern: positive words + negative sentiment + disagreement
        sentiment_score = row.get('sentiment_score', 0)
        positive_in_negative = 0.0
        if self._get_scalar(row['positive_lexical']) > 0 and sentiment_score < -0.1:
            positive_in_negative = 0.5

        # Pattern: dismissive agreement
        dismissive_score = min(1.0, self._get_scalar(row['dismissive_markers']) * 0.3)

        # Pattern: rhetorical questions
        rhetorical_score = 0.0
        if self._get_scalar(row['question_count']) > 0 and self._get_scalar(row['disagreement_markers']) > 0:
            rhetorical_score = 0.4

        # Combine
        sarcasm_score = max(cue_score, positive_in_negative, dismissive_score, rhetorical_score)

        return min(1.0, sarcasm_score)

    def _estimate_irony(self, row: pd.Series) -> float:
        """
        Estimate irony signal (0 to 1)
        Based on semantic contrast and contradiction patterns
        """
        # Direct irony cues
        cue_score = min(1.0, self._get_scalar(row['irony_cues']) * 0.5)

        # Pattern: positive-negative mismatch
        sentiment_score = row.get('sentiment_score', 0)
        mismatch_score = 0.0
        if self._get_scalar(row['positive_lexical']) > 0 and self._get_scalar(row['negative_lexical']) > 0:
            mismatch_score = 0.3

        # Pattern: contradictory framing
        contradiction_score = 0.0
        if self._get_scalar(row['disagreement_markers']) > 0 and self._get_scalar(row['positive_lexical']) > 0:
            contradiction_score = 0.4

        # Combine
        irony_score = max(cue_score, mismatch_score, contradiction_score)

        return min(1.0, irony_score)

    def _estimate_passive_aggression(self, row: pd.Series) -> float:
        """
        Estimate passive-aggression signal (0 to 1)
        Based on dismissive agreement, indirect blame, and contextual patterns
        """
        # Direct passive-aggressive cues
        cue_score = min(1.0, self._get_scalar(row['passive_aggressive_cues']) * 0.4)

        # Dismissive agreement pattern
        dismissive_score = min(1.0, self._get_scalar(row['dismissive_markers']) * 0.3)

        # Indirect blame pattern
        blame_score = min(1.0, self._get_scalar(row['blame_indicators']) * 0.3)

        # Pattern: politeness + negative context
        politeness_negative = 0.0
        if self._get_scalar(row['positive_lexical']) > 0 and row.get('sentiment_score', 0) < -0.2:
            politeness_negative = 0.4

        # Combine
        pa_score = max(cue_score, dismissive_score, blame_score, politeness_negative)

        return min(1.0, pa_score)

    def _estimate_hostility(self, row: pd.Series) -> float:
        """
        Estimate hostility signal (0 to 1)
        Based on aggressive vocabulary, direct blame, and antagonistic patterns
        """
        # Direct hostility cues
        cue_score = min(1.0, self._get_scalar(row['hostility_cues']) * 0.4)

        # Blame indicators
        blame_score = min(1.0, self._get_scalar(row['blame_indicators']) * 0.3)

        # Anger lexical
        anger_score = min(1.0, self._get_scalar(row['anger_lexical']) * 0.2)

        # Aggressive punctuation (excessive exclamation, caps)
        aggression_punctuation = 0.0
        if self._get_scalar(row['exclamation_count']) >= 2 or self._get_scalar(row['uppercase_ratio']) > 0.5:
            aggression_punctuation = 0.3

        # Combine
        hostility_score = max(cue_score, blame_score, anger_score, aggression_punctuation)

        return min(1.0, hostility_score)

    def get_dominant_tone(self, row: pd.Series) -> str:
        """Get the dominant tone for a message"""
        tone_scores = {
            'joy': row.get('joy', 0),
            'sadness': row.get('sadness', 0),
            'anger': row.get('anger', 0),
            'fear': row.get('fear', 0),
            'frustration': row.get('frustration', 0),
        }

        max_tone = max(tone_scores.items(), key=lambda x: x[1])
        if max_tone[1] < 0.2:
            return 'neutral'
        return max_tone[0]
