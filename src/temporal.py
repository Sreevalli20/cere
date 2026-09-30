"""
Temporal Analysis for ToneTrace
Analyzes tone evolution, escalation, and turning points across conversation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class TemporalAnalyzer:
    """Analyze temporal patterns in conversation tone"""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform temporal analysis on conversation
        Returns DataFrame with temporal columns added
        """
        # Initialize columns for single-message case
        if len(df) < 2:
            df['escalation'] = 0.0
            df['escalation_rolling'] = 0.0
            df['is_turning_point'] = False
            tone_columns = [
                'sentiment_score', 'joy', 'sadness', 'anger', 'fear', 'frustration',
                'sarcasm', 'irony', 'passive_aggression', 'hostility'
            ]
            for col in tone_columns:
                if col in df.columns:
                    df[f'{col}_delta'] = 0.0
                    df[f'{col}_rolling'] = df[col]
            return df

        # Calculate deltas (change from previous message)
        df = self._calculate_deltas(df)

        # Calculate rolling metrics
        df = self._calculate_rolling_metrics(df)

        # Calculate escalation score
        df = self._calculate_escalation(df)

        # Detect turning points
        df = self._detect_turning_points(df)

        return df

    def _calculate_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate change in tone from previous message"""
        tone_columns = [
            'sentiment_score', 'joy', 'sadness', 'anger', 'fear', 'frustration',
            'sarcasm', 'irony', 'passive_aggression', 'hostility'
        ]

        for col in tone_columns:
            if col in df.columns:
                df[f'{col}_delta'] = df[col].diff().fillna(0)

        return df

    def _calculate_rolling_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling averages for temporal smoothing"""
        tone_columns = [
            'sentiment_score', 'anger', 'frustration', 'hostility',
            'sarcasm', 'passive_aggression'
        ]

        for col in tone_columns:
            if col in df.columns:
                df[f'{col}_rolling'] = df[col].rolling(
                    window=min(self.window_size, len(df)),
                    min_periods=1
                ).mean()

        return df

    def _calculate_escalation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Estimated Conversation Escalation (0-100)
        Formula combines multiple signals:
        - Negative sentiment change
        - Anger increase
        - Frustration increase
        - Hostility increase
        - Passive-aggression signal
        - Sarcasm signal
        - Tone volatility
        """
        escalation_scores = []

        for idx, row in df.iterrows():
            if idx == 0:
                escalation_scores.append(0.0)
                continue

            # Sentiment decrease
            sentiment_decrease = max(0, -row.get('sentiment_score_delta', 0))

            # Negative sentiment rolling
            negative_sentiment = max(0, -row.get('sentiment_score_rolling', 0))

            # Anger increase
            anger_increase = max(0, row.get('anger_delta', 0))
            anger_level = row.get('anger_rolling', 0)

            # Frustration increase
            frustration_increase = max(0, row.get('frustration_delta', 0))
            frustration_level = row.get('frustration_rolling', 0)

            # Hostility increase
            hostility_increase = max(0, row.get('hostility_delta', 0))
            hostility_level = row.get('hostility_rolling', 0)

            # Sarcasm and passive-aggression
            sarcasm_signal = row.get('sarcasm', 0)
            pa_signal = row.get('passive_aggression', 0)

            # Tone volatility (standard deviation of recent tones)
            volatility = self._calculate_volatility(df, idx)

            # Weighted combination
            escalation = (
                sentiment_decrease * 15 +
                negative_sentiment * 10 +
                anger_increase * 20 +
                anger_level * 15 +
                frustration_increase * 15 +
                frustration_level * 10 +
                hostility_increase * 15 +
                hostility_level * 15 +
                sarcasm_signal * 10 +
                pa_signal * 10 +
                volatility * 20
            )

            # Clamp to 0-100
            escalation = max(0, min(100, escalation))

            escalation_scores.append(escalation)

        df['escalation'] = escalation_scores

        # Calculate rolling escalation
        df['escalation_rolling'] = df['escalation'].rolling(
            window=min(self.window_size, len(df)),
            min_periods=1
        ).mean()

        return df

    def _calculate_volatility(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate tone volatility in recent window"""
        if idx < 2:
            return 0.0

        start_idx = max(0, idx - self.window_size)
        window = df.iloc[start_idx:idx+1]

        if len(window) < 2:
            return 0.0

        # Calculate volatility based on sentiment and anger changes
        sentiment_changes = window['sentiment_score_delta'].abs().sum()
        anger_changes = window['anger_delta'].abs().sum()

        return (sentiment_changes + anger_changes) / len(window)

    def _detect_turning_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect candidate turning points where tone changes substantially
        A turning point requires:
        - Large sentiment decrease OR
        - Large frustration increase OR
        - Large hostility increase
        - Above threshold for recent context
        """
        turning_points = [False] * len(df)

        for idx in range(1, len(df)):
            row = df.iloc[idx]

            # Get change magnitudes
            sentiment_change = abs(row.get('sentiment_score_delta', 0))
            frustration_change = row.get('frustration_delta', 0)
            hostility_change = row.get('hostility_delta', 0)
            anger_change = row.get('anger_delta', 0)

            # Thresholds
            sentiment_threshold = 0.3
            frustration_threshold = 0.2
            hostility_threshold = 0.2
            anger_threshold = 0.2

            # Check if significant change
            significant_change = (
                sentiment_change > sentiment_threshold or
                frustration_change > frustration_threshold or
                hostility_change > hostility_threshold or
                anger_change > anger_threshold
            )

            # Check if change is in negative direction
            negative_direction = (
                row.get('sentiment_score_delta', 0) < -0.1 or
                frustration_change > 0 or
                hostility_change > 0 or
                anger_change > 0
            )

            # Check if escalation is elevated
            elevated_escalation = row.get('escalation', 0) > 30

            # Combine conditions
            is_turning_point = significant_change and negative_direction and elevated_escalation

            turning_points[idx] = is_turning_point

        df['is_turning_point'] = turning_points

        return df

    def get_conversation_summary(self, df: pd.DataFrame) -> Dict:
        """Get overall conversation summary statistics"""
        if len(df) == 0:
            return {}

        summary = {
            'total_messages': len(df),
            'speakers': df['speaker'].nunique(),
            'estimated_escalation': df['escalation'].mean(),
            'peak_escalation': df['escalation'].max(),
            'turning_points': df['is_turning_point'].sum(),
            'dominant_sentiment': df['sentiment'].mode()[0] if not df['sentiment'].mode().empty else 'neutral',
            'sarcasm_signals': (df['sarcasm'] > 0.5).sum(),
            'passive_aggression_signals': (df['passive_aggression'] > 0.5).sum(),
            'hostility_signals': (df['hostility'] > 0.5).sum(),
        }

        return summary

    def get_speaker_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get per-speaker analysis statistics"""
        speaker_stats = []

        for speaker in df['speaker'].unique():
            speaker_df = df[df['speaker'] == speaker]

            stats = {
                'speaker': speaker,
                'message_count': len(speaker_df),
                'avg_sentiment': speaker_df['sentiment_score'].mean(),
                'tone_volatility': speaker_df['sentiment_score_delta'].abs().mean(),
                'avg_frustration': speaker_df['frustration'].mean(),
                'avg_hostility': speaker_df['hostility'].mean(),
                'sarcasm_signals': (speaker_df['sarcasm'] > 0.5).sum(),
                'passive_aggression_signals': (speaker_df['passive_aggression'] > 0.5).sum(),
            }

            speaker_stats.append(stats)

        return pd.DataFrame(speaker_stats)

    def get_escalation_timeline(self, df: pd.DataFrame) -> List[Dict]:
        """Get escalation data for timeline visualization"""
        timeline = []

        for idx, row in df.iterrows():
            timeline.append({
                'message_id': idx,
                'turn_index': row['turn_index'],
                'escalation': row['escalation'],
                'sentiment': row['sentiment_score'],
                'is_turning_point': row['is_turning_point'],
            })

        return timeline
