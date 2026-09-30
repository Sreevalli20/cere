"""
Lightweight NLP Feature Extraction for ToneTrace
Uses lexical patterns, no external models
"""

import re
import pandas as pd
from typing import Dict, List
import string


class FeatureExtractor:
    """Extract lightweight linguistic features from messages"""

    def __init__(self):
        # Lexical dictionaries
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'awesome', 'fantastic',
            'love', 'happy', 'glad', 'pleased', 'thanks', 'thank', 'appreciate', 'perfect',
            'beautiful', 'nice', 'best', 'better', 'brilliant', 'super', 'cool', 'fine',
            'ok', 'okay', 'yes', 'yeah', 'sure', 'agree', 'right', 'correct', 'perfectly'
        }

        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry', 'annoyed',
            'frustrated', 'upset', 'sad', 'disappointed', 'wrong', 'stupid', 'dumb', 'idiotic',
            'ridiculous', 'absurd', 'unacceptable', 'disagree', 'no', 'never', 'worst',
            'worse', 'poor', 'fail', 'failure', 'problem', 'issue', 'error', 'mistake'
        }

        self.negation_words = {
            'not', 'no', 'never', 'none', 'neither', 'nobody', 'nothing', 'nowhere',
            'hardly', 'barely', 'scarcely', "don't", "doesn't", "didn't", "won't", "wouldn't",
            "couldn't", "shouldn't", "can't", "cannot", "isn't", "aren't", "wasn't", "weren't"
        }

        self.intensifiers = {
            'very', 'really', 'extremely', 'absolutely', 'completely', 'totally', 'utterly',
            'highly', 'incredibly', 'extraordinarily', 'so', 'too', 'quite', 'rather',
            'especially', 'particularly', 'seriously', 'definitely', 'certainly'
        }

        self.disagreement_markers = {
            'disagree', 'disagreement', 'wrong', 'incorrect', 'not true', 'false', 'no',
            'actually', 'in fact', 'however', 'but', 'contrary', 'dispute', 'reject', 'deny'
        }

        self.blame_indicators = {
            'your fault', 'blame', 'responsible', 'cause', 'because of you', 'you made',
            'you did', 'you should have', 'you need to', 'you always', 'you never'
        }

        self.dismissive_markers = {
            'whatever', 'fine', 'whatever you say', 'as you wish', 'if you say so',
            'sure', 'okay then', 'alright', 'fine then', 'whatever works'
        }

        self.sarcasm_cues = {
            'yeah right', 'sure thing', 'of course', 'great', 'amazing', 'brilliant',
            'wow', 'impressive', 'clearly', 'obviously', 'totally', 'exactly',
            'perfect', 'just what we needed', 'thanks a lot'
        }

        self.irony_cues = {
            'ironically', 'interestingly', 'funny enough', 'strangely enough',
            'contrary to', 'opposite of', 'supposed to', 'meant to'
        }

        self.passive_aggressive_cues = {
            'fine', 'whatever', 'as you wish', 'if you say so', 'apparently',
            'no problem', 'thanks for nothing', 'glad i could help', 'sure whatever',
            'okay then', 'fine by me', 'whatever you want'
        }

        self.hostility_cues = {
            'stupid', 'idiot', 'dumb', 'moron', 'fool', 'ridiculous', 'absurd',
            'shut up', 'get lost', 'go away', 'hate', 'deserve', 'pathetic',
            'useless', 'worthless', 'terrible', 'awful', 'horrible'
        }

        # Emotion keywords
        self.joy_words = {
            'happy', 'joy', 'excited', 'thrilled', 'delighted', 'cheerful', 'glad',
            'love', 'enjoy', 'fun', 'great', 'wonderful', 'awesome', 'fantastic'
        }

        self.sadness_words = {
            'sad', 'unhappy', 'depressed', 'down', 'upset', 'disappointed', 'hurt',
            'sorry', 'regret', 'miss', 'lost', 'grief', 'sorrow', 'unfortunate'
        }

        self.anger_words = {
            'angry', 'furious', 'mad', 'rage', 'irritated', 'annoyed', 'frustrated',
            'upset', 'livid', 'outraged', 'fuming', 'hostile', 'aggressive'
        }

        self.fear_words = {
            'afraid', 'scared', 'fear', 'worried', 'anxious', 'nervous', 'concerned',
            'terrified', 'panic', 'dread', 'apprehensive', 'uneasy'
        }

        self.frustration_words = {
            'frustrated', 'stuck', 'blocked', 'impossible', 'can\'t', 'won\'t',
            'useless', 'pointless', 'waste', 'annoying', 'difficult', 'hard',
            'struggling', 'failing', 'failed', 'wrong', 'error', 'problem'
        }

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from conversation DataFrame"""
        features = []

        for idx, row in df.iterrows():
            text = str(row['text']).lower()
            words = text.split()

            feature_dict = {
                'message_length': len(row['text']),
                'word_count': len(words),
                'sentence_count': self._count_sentences(row['text']),
                'uppercase_ratio': self._uppercase_ratio(row['text']),
                'uppercase_word_count': self._uppercase_word_count(row['text']),
                'question_count': text.count('?'),
                'exclamation_count': text.count('!'),
                'repeated_punctuation': self._repeated_punctuation(text),
                'emoji_count': self._emoji_count(text),
                'positive_lexical': self._word_match_count(text, self.positive_words),
                'negative_lexical': self._word_match_count(text, self.negative_words),
                'negation_markers': self._word_match_count(text, self.negation_words),
                'intensifiers': self._word_match_count(text, self.intensifiers),
                'disagreement_markers': self._word_match_count(text, self.disagreement_markers),
                'blame_indicators': self._word_match_count(text, self.blame_indicators),
                'dismissive_markers': self._word_match_count(text, self.dismissive_markers),
                'sarcasm_cues': self._word_match_count(text, self.sarcasm_cues),
                'irony_cues': self._word_match_count(text, self.irony_cues),
                'passive_aggressive_cues': self._word_match_count(text, self.passive_aggressive_cues),
                'hostility_cues': self._word_match_count(text, self.hostility_cues),
                'joy_lexical': self._word_match_count(text, self.joy_words),
                'sadness_lexical': self._word_match_count(text, self.sadness_words),
                'anger_lexical': self._word_match_count(text, self.anger_words),
                'fear_lexical': self._word_match_count(text, self.fear_words),
                'frustration_lexical': self._word_match_count(text, self.frustration_words),
            }

            features.append(feature_dict)

        feature_df = pd.DataFrame(features)

        # Combine with original data
        result = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
        return result

    def _count_sentences(self, text: str) -> int:
        """Count sentences based on punctuation"""
        # Simple heuristic: split by . ! ? followed by space or end
        pattern = r'[.!?]+(?:\s|$)'
        return len(re.findall(pattern, text))

    def _uppercase_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase letters"""
        if not text:
            return 0.0
        uppercase = sum(1 for c in text if c.isupper())
        return uppercase / len(text) if len(text) > 0 else 0.0

    def _uppercase_word_count(self, text: str) -> int:
        """Count words that are entirely uppercase"""
        words = text.split()
        return sum(1 for word in words if word.isupper() and len(word) > 1)

    def _repeated_punctuation(self, text: str) -> int:
        """Count repeated punctuation marks"""
        patterns = [r'\?\?+', r'!!+', r'\.\.\.+']
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text))
        return count

    def _emoji_count(self, text: str) -> int:
        """Count emoji-like patterns"""
        # Simple emoji detection (not exhaustive)
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
        return len(re.findall(emoji_pattern, text))

    def _word_match_count(self, text: str, word_set: set) -> int:
        """Count occurrences of words from a set"""
        words = text.split()
        count = 0
        for word in words:
            # Remove punctuation
            clean_word = word.strip(string.punctuation)
            if clean_word in word_set:
                count += 1
        return count
