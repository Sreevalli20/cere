"""
Tests for Tone Engine
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import ToneEngine
from src.parser import ConversationParser


def test_sentiment_estimation():
    """Test sentiment estimation"""
    engine = ToneEngine()

    df = pd.DataFrame({
        'text': ['This is great!', 'This is terrible', 'This is neutral'],
        'positive_lexical': [1, 0, 0],
        'negative_lexical': [0, 1, 0],
        'negation_markers': [0, 0, 0],
        'intensifiers': [0, 0, 0],
        'exclamation_count': [1, 0, 0],
        'question_count': [0, 0, 0],
        'word_count': [4, 4, 4],
        'sentence_count': [1, 1, 1],
        'uppercase_ratio': [0.0, 0.0, 0.0],
        'uppercase_word_count': [0, 0, 0],
        'repeated_punctuation': [0, 0, 0],
        'emoji_count': [0, 0, 0],
        'disagreement_markers': [0, 0, 0],
        'blame_indicators': [0, 0, 0],
        'dismissive_markers': [0, 0, 0],
        'sarcasm_cues': [0, 0, 0],
        'irony_cues': [0, 0, 0],
        'passive_aggressive_cues': [0, 0, 0],
        'hostility_cues': [0, 0, 0],
        'joy_lexical': [1, 0, 0],
        'sadness_lexical': [0, 0, 0],
        'anger_lexical': [0, 0, 0],
        'fear_lexical': [0, 0, 0],
        'frustration_lexical': [0, 0, 0],
    })

    df = engine.analyze(df)

    assert df.iloc[0]['sentiment'] == 'positive'
    assert df.iloc[1]['sentiment'] == 'negative'
    assert df.iloc[2]['sentiment'] == 'neutral'


def test_sentiment_score_range():
    """Test that sentiment scores are in [-1, 1] range"""
    engine = ToneEngine()

    df = pd.DataFrame({
        'text': ['Test message'],
        'positive_lexical': [0],
        'negative_lexical': [0],
        'negation_markers': [0],
        'intensifiers': [0],
        'exclamation_count': [0],
        'question_count': [0],
        'word_count': [2],
        'sentence_count': [1],
        'uppercase_ratio': [0.0],
        'uppercase_word_count': [0],
        'repeated_punctuation': [0],
        'emoji_count': [0],
        'disagreement_markers': [0],
        'blame_indicators': [0],
        'dismissive_markers': [0],
        'sarcasm_cues': [0],
        'irony_cues': [0],
        'passive_aggressive_cues': [0],
        'hostility_cues': [0],
        'joy_lexical': [0],
        'sadness_lexical': [0],
        'anger_lexical': [0],
        'fear_lexical': [0],
        'frustration_lexical': [0],
    })

    df = engine.analyze(df)

    assert -1.0 <= df.iloc[0]['sentiment_score'] <= 1.0


def test_emotion_estimation():
    """Test emotion estimation"""
    engine = ToneEngine()

    df = pd.DataFrame({
        'text': ['I am happy', 'I am sad', 'I am angry'],
        'positive_lexical': [0, 0, 0],
        'negative_lexical': [0, 0, 0],
        'negation_markers': [0, 0, 0],
        'intensifiers': [0, 0, 0],
        'exclamation_count': [0, 0, 0],
        'question_count': [0, 0, 0],
        'word_count': [3, 3, 3],
        'sentence_count': [1, 1, 1],
        'uppercase_ratio': [0.0, 0.0, 0.0],
        'uppercase_word_count': [0, 0, 0],
        'repeated_punctuation': [0, 0, 0],
        'emoji_count': [0, 0, 0],
        'disagreement_markers': [0, 0, 0],
        'blame_indicators': [0, 0, 0],
        'dismissive_markers': [0, 0, 0],
        'sarcasm_cues': [0, 0, 0],
        'irony_cues': [0, 0, 0],
        'passive_aggressive_cues': [0, 0, 0],
        'hostility_cues': [0, 0, 0],
        'joy_lexical': [1, 0, 0],
        'sadness_lexical': [0, 1, 0],
        'anger_lexical': [0, 0, 1],
        'fear_lexical': [0, 0, 0],
        'frustration_lexical': [0, 0, 0],
    })

    df = engine.analyze(df)

    assert df.iloc[0]['joy'] > 0
    assert df.iloc[1]['sadness'] > 0
    assert df.iloc[2]['anger'] > 0


def test_sarcasm_detection():
    """Test sarcasm signal detection"""
    engine = ToneEngine()

    df = pd.DataFrame({
        'text': ['Yeah right', 'Great job'],
        'positive_lexical': [1, 1],
        'negative_lexical': [0, 0],
        'negation_markers': [0, 0],
        'intensifiers': [0, 0],
        'exclamation_count': [0, 0],
        'question_count': [0, 0],
        'word_count': [2, 2],
        'sentence_count': [1, 1],
        'uppercase_ratio': [0.0, 0.0],
        'uppercase_word_count': [0, 0],
        'repeated_punctuation': [0, 0],
        'emoji_count': [0, 0],
        'disagreement_markers': [0, 0],
        'blame_indicators': [0, 0],
        'dismissive_markers': [0, 0],
        'sarcasm_cues': [1, 1],
        'irony_cues': [0, 0],
        'passive_aggressive_cues': [0, 0],
        'hostility_cues': [0, 0],
        'joy_lexical': [0, 0],
        'sadness_lexical': [0, 0],
        'anger_lexical': [0, 0],
        'fear_lexical': [0, 0],
        'frustration_lexical': [0, 0],
    })

    df = engine.analyze(df)

    # Should detect sarcasm from cues
    assert df.iloc[0]['sarcasm'] > 0 or df.iloc[1]['sarcasm'] > 0


def test_hostility_detection():
    """Test hostility signal detection"""
    engine = ToneEngine()

    df = pd.DataFrame({
        'text': ['This is stupid', 'You are an idiot'],
        'positive_lexical': [0, 0],
        'negative_lexical': [0, 0],
        'negation_markers': [0, 0],
        'intensifiers': [0, 0],
        'exclamation_count': [0, 0],
        'question_count': [0, 0],
        'word_count': [3, 4],
        'sentence_count': [1, 1],
        'uppercase_ratio': [0.0, 0.0],
        'uppercase_word_count': [0, 0],
        'repeated_punctuation': [0, 0],
        'emoji_count': [0, 0],
        'disagreement_markers': [0, 0],
        'blame_indicators': [0, 0],
        'dismissive_markers': [0, 0],
        'sarcasm_cues': [0, 0],
        'irony_cues': [0, 0],
        'passive_aggressive_cues': [0, 0],
        'hostility_cues': [1, 1],
        'joy_lexical': [0, 0],
        'sadness_lexical': [0, 0],
        'anger_lexical': [0, 0],
        'fear_lexical': [0, 0],
        'frustration_lexical': [0, 0],
    })

    df = engine.analyze(df)

    assert df.iloc[0]['hostility'] > 0
    assert df.iloc[1]['hostility'] > 0


def test_signal_ranges():
    """Test that all signals are in [0, 1] range"""
    engine = ToneEngine()

    df = pd.DataFrame({
        'text': ['Test'],
        'positive_lexical': [0],
        'negative_lexical': [0],
        'negation_markers': [0],
        'intensifiers': [0],
        'exclamation_count': [0],
        'question_count': [0],
        'word_count': [1],
        'sentence_count': [1],
        'uppercase_ratio': [0.0],
        'uppercase_word_count': [0],
        'repeated_punctuation': [0],
        'emoji_count': [0],
        'disagreement_markers': [0],
        'blame_indicators': [0],
        'dismissive_markers': [0],
        'sarcasm_cues': [0],
        'irony_cues': [0],
        'passive_aggressive_cues': [0],
        'hostility_cues': [0],
        'joy_lexical': [0],
        'sadness_lexical': [0],
        'anger_lexical': [0],
        'fear_lexical': [0],
        'frustration_lexical': [0],
    })

    df = engine.analyze(df)

    signals = ['joy', 'sadness', 'anger', 'fear', 'frustration',
               'sarcasm', 'irony', 'passive_aggression', 'hostility']

    for signal in signals:
        assert 0.0 <= df.iloc[0][signal] <= 1.0


def test_dominant_tone():
    """Test dominant tone extraction"""
    engine = ToneEngine()

    df = pd.DataFrame({
        'text': ['Test'],
        'positive_lexical': [0],
        'negative_lexical': [0],
        'negation_markers': [0],
        'intensifiers': [0],
        'exclamation_count': [0],
        'question_count': [0],
        'word_count': [1],
        'sentence_count': [1],
        'uppercase_ratio': [0.0],
        'uppercase_word_count': [0],
        'repeated_punctuation': [0],
        'emoji_count': [0],
        'disagreement_markers': [0],
        'blame_indicators': [0],
        'dismissive_markers': [0],
        'sarcasm_cues': [0],
        'irony_cues': [0],
        'passive_aggressive_cues': [0],
        'hostility_cues': [0],
        'joy_lexical': [1],
        'sadness_lexical': [0],
        'anger_lexical': [0],
        'fear_lexical': [0],
        'frustration_lexical': [0],
    })

    df = engine.analyze(df)

    dominant = engine.get_dominant_tone(df.iloc[0])
    assert dominant == 'joy'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
