"""
Tests for Temporal Analysis
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.temporal import TemporalAnalyzer
from src.engine import ToneEngine


def test_temporal_analysis():
    """Test temporal analysis on conversation"""
    analyzer = TemporalAnalyzer()

    # Create mock analyzed data
    df = pd.DataFrame({
        'conversation_id': ['test'] * 3,
        'message_id': [0, 1, 2],
        'timestamp': pd.date_range('2026-09-30', periods=3, freq='h'),
        'speaker': ['Alice', 'Bob', 'Alice'],
        'text': ['Hello', 'Hi', 'Bye'],
        'turn_index': [0, 1, 2],
        'sentiment': ['positive', 'neutral', 'negative'],
        'sentiment_score': [0.5, 0.0, -0.5],
        'joy': [0.5, 0.0, 0.0],
        'sadness': [0.0, 0.0, 0.3],
        'anger': [0.0, 0.0, 0.4],
        'fear': [0.0, 0.0, 0.0],
        'frustration': [0.0, 0.0, 0.3],
        'sarcasm': [0.0, 0.0, 0.2],
        'irony': [0.0, 0.0, 0.0],
        'passive_aggression': [0.0, 0.0, 0.2],
        'hostility': [0.0, 0.0, 0.3],
    })

    df = analyzer.analyze(df)

    # Check that delta columns were added
    assert 'sentiment_score_delta' in df.columns
    assert 'anger_delta' in df.columns
    assert 'frustration_delta' in df.columns

    # Check that escalation was calculated
    assert 'escalation' in df.columns
    assert df['escalation'].iloc[0] == 0.0  # First message has no escalation

    # Check that turning points were detected
    assert 'is_turning_point' in df.columns


def test_delta_calculation():
    """Test that deltas are calculated correctly"""
    analyzer = TemporalAnalyzer()

    df = pd.DataFrame({
        'sentiment_score': [0.5, -0.3, 0.0],
        'anger': [0.0, 0.4, 0.2],
        'frustration': [0.0, 0.3, 0.1],
        'hostility': [0.0, 0.3, 0.1],
        'sarcasm': [0.0, 0.2, 0.0],
        'passive_aggression': [0.0, 0.2, 0.0],
        'joy': [0.5, 0.0, 0.0],
        'sadness': [0.0, 0.0, 0.0],
        'fear': [0.0, 0.0, 0.0],
    })

    df = analyzer._calculate_deltas(df)

    # First row should have delta 0
    assert df['sentiment_score_delta'].iloc[0] == 0.0

    # Second row should have negative delta
    assert df['sentiment_score_delta'].iloc[1] == pytest.approx(-0.8, abs=0.01)

    # Anger should increase
    assert df['anger_delta'].iloc[1] == pytest.approx(0.4, abs=0.01)


def test_escalation_range():
    """Test that escalation scores are in [0, 100] range"""
    analyzer = TemporalAnalyzer()

    df = pd.DataFrame({
        'sentiment_score': [0.5, -0.5, 0.0],
        'anger': [0.0, 0.5, 0.2],
        'frustration': [0.0, 0.5, 0.1],
        'hostility': [0.0, 0.5, 0.1],
        'sarcasm': [0.0, 0.5, 0.0],
        'passive_aggression': [0.0, 0.5, 0.0],
        'joy': [0.5, 0.0, 0.0],
        'sadness': [0.0, 0.0, 0.0],
        'fear': [0.0, 0.0, 0.0],
    })

    df = analyzer._calculate_deltas(df)
    df = analyzer._calculate_rolling_metrics(df)
    df = analyzer._calculate_escalation(df)

    for idx in range(len(df)):
        assert 0.0 <= df['escalation'].iloc[idx] <= 100.0


def test_turning_point_detection():
    """Test turning point detection"""
    analyzer = TemporalAnalyzer()

    df = pd.DataFrame({
        'sentiment_score': [0.5, -0.5, 0.0],
        'anger': [0.0, 0.5, 0.2],
        'frustration': [0.0, 0.5, 0.1],
        'hostility': [0.0, 0.5, 0.1],
        'sarcasm': [0.0, 0.5, 0.0],
        'passive_aggression': [0.0, 0.5, 0.0],
        'joy': [0.5, 0.0, 0.0],
        'sadness': [0.0, 0.0, 0.0],
        'fear': [0.0, 0.0, 0.0],
    })

    df = analyzer._calculate_deltas(df)
    df = analyzer._calculate_rolling_metrics(df)
    df = analyzer._calculate_escalation(df)
    df = analyzer._detect_turning_points(df)

    # First message should not be a turning point
    assert df['is_turning_point'].iloc[0] == False

    # Second message with large negative change might be a turning point
    # (depending on escalation level)


def test_conversation_summary():
    """Test conversation summary generation"""
    analyzer = TemporalAnalyzer()

    df = pd.DataFrame({
        'conversation_id': ['test'] * 3,
        'message_id': [0, 1, 2],
        'timestamp': pd.date_range('2026-09-30', periods=3, freq='h'),
        'speaker': ['Alice', 'Bob', 'Alice'],
        'text': ['Hello', 'Hi', 'Bye'],
        'turn_index': [0, 1, 2],
        'sentiment': ['positive', 'neutral', 'negative'],
        'sentiment_score': [0.5, 0.0, -0.5],
        'joy': [0.5, 0.0, 0.0],
        'sadness': [0.0, 0.0, 0.3],
        'anger': [0.0, 0.0, 0.4],
        'fear': [0.0, 0.0, 0.0],
        'frustration': [0.0, 0.0, 0.3],
        'sarcasm': [0.0, 0.0, 0.2],
        'irony': [0.0, 0.0, 0.0],
        'passive_aggression': [0.0, 0.0, 0.2],
        'hostility': [0.0, 0.0, 0.3],
    })

    df = analyzer.analyze(df)
    summary = analyzer.get_conversation_summary(df)

    assert summary['total_messages'] == 3
    assert summary['speakers'] == 2
    assert 'estimated_escalation' in summary
    assert 'peak_escalation' in summary
    assert 'turning_points' in summary


def test_speaker_analysis():
    """Test per-speaker analysis"""
    analyzer = TemporalAnalyzer()

    df = pd.DataFrame({
        'conversation_id': ['test'] * 4,
        'message_id': [0, 1, 2, 3],
        'timestamp': pd.date_range('2026-09-30', periods=4, freq='h'),
        'speaker': ['Alice', 'Bob', 'Alice', 'Bob'],
        'text': ['Hello', 'Hi', 'Bye', 'Ok'],
        'turn_index': [0, 1, 2, 3],
        'sentiment': ['positive', 'neutral', 'negative', 'neutral'],
        'sentiment_score': [0.5, 0.0, -0.5, 0.0],
        'joy': [0.5, 0.0, 0.0, 0.0],
        'sadness': [0.0, 0.0, 0.3, 0.0],
        'anger': [0.0, 0.0, 0.4, 0.0],
        'fear': [0.0, 0.0, 0.0, 0.0],
        'frustration': [0.0, 0.0, 0.3, 0.0],
        'sarcasm': [0.0, 0.0, 0.2, 0.0],
        'irony': [0.0, 0.0, 0.0, 0.0],
        'passive_aggression': [0.0, 0.0, 0.2, 0.0],
        'hostility': [0.0, 0.0, 0.3, 0.0],
    })

    df = analyzer.analyze(df)
    speaker_df = analyzer.get_speaker_analysis(df)

    assert len(speaker_df) == 2
    assert 'Alice' in speaker_df['speaker'].values
    assert 'Bob' in speaker_df['speaker'].values
    assert 'message_count' in speaker_df.columns
    assert 'avg_sentiment' in speaker_df.columns


def test_escalation_timeline():
    """Test escalation timeline generation"""
    analyzer = TemporalAnalyzer()

    df = pd.DataFrame({
        'conversation_id': ['test'] * 3,
        'message_id': [0, 1, 2],
        'timestamp': pd.date_range('2026-09-30', periods=3, freq='h'),
        'speaker': ['Alice', 'Bob', 'Alice'],
        'text': ['Hello', 'Hi', 'Bye'],
        'turn_index': [0, 1, 2],
        'sentiment': ['positive', 'neutral', 'negative'],
        'sentiment_score': [0.5, 0.0, -0.5],
        'joy': [0.5, 0.0, 0.0],
        'sadness': [0.0, 0.0, 0.3],
        'anger': [0.0, 0.0, 0.4],
        'fear': [0.0, 0.0, 0.0],
        'frustration': [0.0, 0.0, 0.3],
        'sarcasm': [0.0, 0.0, 0.2],
        'irony': [0.0, 0.0, 0.0],
        'passive_aggression': [0.0, 0.0, 0.2],
        'hostility': [0.0, 0.0, 0.3],
    })

    df = analyzer.analyze(df)
    timeline = analyzer.get_escalation_timeline(df)

    assert len(timeline) == 3
    assert timeline[0]['message_id'] == 0
    assert 'escalation' in timeline[0]
    assert 'sentiment' in timeline[0]
    assert 'is_turning_point' in timeline[0]


def test_single_message_conversation():
    """Test handling of single-message conversation"""
    analyzer = TemporalAnalyzer()

    df = pd.DataFrame({
        'conversation_id': ['test'],
        'message_id': [0],
        'timestamp': pd.date_range('2026-09-30', periods=1, freq='h'),
        'speaker': ['Alice'],
        'text': ['Hello'],
        'turn_index': [0],
        'sentiment': ['positive'],
        'sentiment_score': [0.5],
        'joy': [0.5],
        'sadness': [0.0],
        'anger': [0.0],
        'fear': [0.0],
        'frustration': [0.0],
        'sarcasm': [0.0],
        'irony': [0.0],
        'passive_aggression': [0.0],
        'hostility': [0.0],
    })

    df = analyzer.analyze(df)

    # Should not crash
    assert len(df) == 1
    assert df['escalation'].iloc[0] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
