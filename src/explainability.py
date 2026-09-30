"""
Explainability Module for ToneTrace
Provides evidence and explanations for tone detections
"""

import pandas as pd
from typing import Dict, List, Optional


class ExplainabilityEngine:
    """Generate explanations for tone detections"""

    def __init__(self):
        self.cue_descriptions = {
            'dismissive_agreement': 'dismissive agreement pattern',
            'negative_framing': 'negative lexical framing',
            'implied_blame': 'blame attribution pattern',
            'contextual_contrast': 'positive-negative contextual mismatch',
            'abrupt_tone_change': 'abrupt shift in tone',
            'exaggerated_praise': 'exaggerated positive phrasing',
            'rhetorical_construction': 'rhetorical question pattern',
            'contradiction': 'contradictory framing',
            'indirect_blame': 'indirect blame attribution',
            'sarcastic_politeness': 'politeness in negative context',
            'aggressive_punctuation': 'excessive exclamation or capitalization',
            'hostile_vocabulary': 'hostile lexical content',
            'direct_blame': 'direct blame attribution',
            'repeated_antagonism': 'repeated antagonistic interaction',
        }

    def get_turning_point_explanation(self, df: pd.DataFrame, message_idx: int) -> Dict:
        """
        Generate explanation for a turning point
        """
        if message_idx < 1 or message_idx >= len(df):
            return {}

        current = df.iloc[message_idx]
        previous = df.iloc[message_idx - 1]

        explanation = {
            'message_number': message_idx + 1,
            'speaker': current['speaker'],
            'message': current['text'],
            'timestamp': current.get('timestamp', 'N/A'),
            'previous_tone': self._format_tone(previous),
            'current_tone': self._format_tone(current),
            'tone_delta': self._calculate_tone_delta(previous, current),
            'supporting_signals': self._get_supporting_signals(current, previous),
            'why_detected': self._explain_turning_point(current, previous),
        }

        return explanation

    def _format_tone(self, row: pd.Series) -> Dict:
        """Format tone values for display"""
        return {
            'sentiment': row.get('sentiment', 'neutral'),
            'sentiment_score': round(row.get('sentiment_score', 0), 2),
            'frustration': round(row.get('frustration', 0), 2),
            'hostility': round(row.get('hostility', 0), 2),
            'sarcasm': round(row.get('sarcasm', 0), 2),
            'passive_aggression': round(row.get('passive_aggression', 0), 2),
        }

    def _calculate_tone_delta(self, previous: pd.Series, current: pd.Series) -> Dict:
        """Calculate tone changes"""
        return {
            'sentiment': round(current.get('sentiment_score', 0) - previous.get('sentiment_score', 0), 2),
            'frustration': round(current.get('frustration', 0) - previous.get('frustration', 0), 2),
            'hostility': round(current.get('hostility', 0) - previous.get('hostility', 0), 2),
            'sarcasm': round(current.get('sarcasm', 0) - previous.get('sarcasm', 0), 2),
            'passive_aggression': round(current.get('passive_aggression', 0) - previous.get('passive_aggression', 0), 2),
        }

    def _get_supporting_signals(self, current: pd.Series, previous: pd.Series) -> List[str]:
        """Get linguistic cues supporting the detection"""
        signals = []

        # Check for various cues
        if current.get('dismissive_markers', 0) > 0:
            signals.append(self.cue_descriptions['dismissive_agreement'])

        if current.get('negative_lexical', 0) > previous.get('negative_lexical', 0):
            signals.append(self.cue_descriptions['negative_framing'])

        if current.get('blame_indicators', 0) > 0:
            signals.append(self.cue_descriptions['implied_blame'])

        if current.get('positive_lexical', 0) > 0 and current.get('sentiment_score', 0) < -0.1:
            signals.append(self.cue_descriptions['contextual_contrast'])

        if abs(current.get('sentiment_score_delta', 0)) > 0.3:
            signals.append(self.cue_descriptions['abrupt_tone_change'])

        if current.get('sarcasm_cues', 0) > 0:
            signals.append(self.cue_descriptions['exaggerated_praise'])

        if current.get('question_count', 0) > 0 and current.get('disagreement_markers', 0) > 0:
            signals.append(self.cue_descriptions['rhetorical_construction'])

        if current.get('irony_cues', 0) > 0:
            signals.append(self.cue_descriptions['contradiction'])

        if current.get('passive_aggressive_cues', 0) > 0:
            signals.append(self.cue_descriptions['sarcastic_politeness'])

        if current.get('exclamation_count', 0) >= 2 or current.get('uppercase_ratio', 0) > 0.5:
            signals.append(self.cue_descriptions['aggressive_punctuation'])

        if current.get('hostility_cues', 0) > 0:
            signals.append(self.cue_descriptions['hostile_vocabulary'])

        return signals if signals else ['tone change detected']

    def _explain_turning_point(self, current: pd.Series, previous: pd.Series) -> str:
        """Generate explanation for why this is a turning point"""
        reasons = []

        sentiment_delta = current.get('sentiment_score_delta', 0)
        frustration_delta = current.get('frustration_delta', 0)
        hostility_delta = current.get('hostility_delta', 0)
        anger_delta = current.get('anger_delta', 0)

        if sentiment_delta < -0.3:
            reasons.append(f"sentiment decreased significantly ({sentiment_delta:.2f})")

        if frustration_delta > 0.2:
            reasons.append(f"frustration increased ({frustration_delta:.2f})")

        if hostility_delta > 0.2:
            reasons.append(f"hostility increased ({hostility_delta:.2f})")

        if anger_delta > 0.2:
            reasons.append(f"anger increased ({anger_delta:.2f})")

        if current.get('escalation', 0) > 30:
            reasons.append(f"escalation level elevated ({current.get('escalation', 0):.1f})")

        if not reasons:
            reasons.append("multiple tone dimensions changed substantially")

        return "This message is a candidate turning point because " + ", ".join(reasons) + "."

    def get_strongest_tone_change(self, df: pd.DataFrame) -> Dict:
        """Find the strongest tone change in the conversation"""
        if len(df) < 2:
            return {}

        # Calculate overall tone change magnitude
        max_change = 0
        max_idx = 0

        for idx in range(1, len(df)):
            current = df.iloc[idx]
            previous = df.iloc[idx - 1]

            # Calculate change magnitude
            change_magnitude = (
                abs(current.get('sentiment_score_delta', 0)) +
                abs(current.get('frustration_delta', 0)) +
                abs(current.get('hostility_delta', 0)) +
                abs(current.get('anger_delta', 0))
            )

            if change_magnitude > max_change:
                max_change = change_magnitude
                max_idx = idx

        if max_change == 0:
            return {}

        return self.get_turning_point_explanation(df, max_idx)

    def get_message_explanation(self, row: pd.Series) -> Dict:
        """Get explanation for a single message's tone signals"""
        explanation = {
            'estimated_signals': {
                'sentiment': row.get('sentiment', 'neutral'),
                'sentiment_score': round(row.get('sentiment_score', 0), 2),
                'joy': round(row.get('joy', 0), 2),
                'sadness': round(row.get('sadness', 0), 2),
                'anger': round(row.get('anger', 0), 2),
                'fear': round(row.get('fear', 0), 2),
                'frustration': round(row.get('frustration', 0), 2),
                'sarcasm': round(row.get('sarcasm', 0), 2),
                'irony': round(row.get('irony', 0), 2),
                'passive_aggression': round(row.get('passive_aggression', 0), 2),
                'hostility': round(row.get('hostility', 0), 2),
            },
            'supporting_cues': self._get_message_cues(row),
        }

        return explanation

    def _get_message_cues(self, row: pd.Series) -> List[str]:
        """Get linguistic cues for a message"""
        cues = []

        if row.get('positive_lexical', 0) > 0:
            cues.append(f"positive lexical indicators ({row['positive_lexical']})")

        if row.get('negative_lexical', 0) > 0:
            cues.append(f"negative lexical indicators ({row['negative_lexical']})")

        if row.get('negation_markers', 0) > 0:
            cues.append(f"negation markers ({row['negation_markers']})")

        if row.get('intensifiers', 0) > 0:
            cues.append(f"intensifiers ({row['intensifiers']})")

        if row.get('disagreement_markers', 0) > 0:
            cues.append(f"disagreement markers ({row['disagreement_markers']})")

        if row.get('sarcasm_cues', 0) > 0:
            cues.append(f"sarcasm cues ({row['sarcasm_cues']})")

        if row.get('exclamation_count', 0) > 0:
            cues.append(f"exclamation marks ({row['exclamation_count']})")

        if row.get('question_count', 0) > 0:
            cues.append(f"questions ({row['question_count']})")

        if row.get('uppercase_ratio', 0) > 0.3:
            cues.append(f"uppercase ratio ({row['uppercase_ratio']:.2f})")

        return cues if cues else ['standard message structure']

    def format_signal_strength(self, value: float) -> str:
        """Format signal strength for display"""
        if value < 0.2:
            return "Minimal"
        elif value < 0.4:
            return "Low"
        elif value < 0.6:
            return "Moderate"
        elif value < 0.8:
            return "Strong"
        else:
            return "Very Strong"
