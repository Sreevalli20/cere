"""
Conversation Parser for ToneTrace
Supports CSV, JSON, and WhatsApp TXT formats
"""

import pandas as pd
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import io


class ConversationParser:
    """Parse conversation data from multiple formats"""

    def __init__(self, max_messages: int = 5000, max_message_length: int = 4000):
        self.max_messages = max_messages
        self.max_message_length = max_message_length

    def parse(self, file_content: str, filename: str) -> pd.DataFrame:
        """
        Auto-detect format and parse conversation
        Returns normalized DataFrame with columns:
        conversation_id, message_id, timestamp, speaker, text, turn_index
        """
        try:
            if filename.endswith('.csv'):
                return self._parse_csv(file_content)
            elif filename.endswith('.json'):
                return self._parse_json(file_content)
            elif filename.endswith('.txt'):
                return self._parse_whatsapp(file_content)
            else:
                # Try to detect based on content
                if file_content.strip().startswith('{'):
                    return self._parse_json(file_content)
                elif self._is_whatsapp_format(file_content):
                    return self._parse_whatsapp(file_content)
                else:
                    return self._parse_csv(file_content)
        except Exception as e:
            raise ValueError(f"Failed to parse conversation: {str(e)}")

    def _is_whatsapp_format(self, content: str) -> bool:
        """Check if content matches WhatsApp export format"""
        lines = content.split('\n')[:10]
        whatsapp_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\s*-\s*[^:]+:'
        return any(re.match(whatsapp_pattern, line.strip()) for line in lines if line.strip())

    def _parse_csv(self, content: str) -> pd.DataFrame:
        """Parse CSV with flexible column name recognition"""
        df = pd.read_csv(io.StringIO(content))

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Map common column names
        speaker_col = self._find_column(df, ['speaker', 'sender', 'author', 'user', 'from', 'name'])
        text_col = self._find_column(df, ['text', 'message', 'content', 'body'])
        timestamp_col = self._find_column(df, ['timestamp', 'time', 'date', 'datetime'])

        if not speaker_col or not text_col:
            raise ValueError("CSV must contain speaker/text columns")

        # Create normalized DataFrame
        normalized = pd.DataFrame({
            'conversation_id': 'uploaded',
            'message_id': range(len(df)),
            'timestamp': df[timestamp_col] if timestamp_col else None,
            'speaker': df[speaker_col].astype(str),
            'text': df[text_col].astype(str),
            'turn_index': range(len(df))
        })

        return self._validate_and_normalize(normalized)

    def _parse_json(self, content: str) -> pd.DataFrame:
        """Parse JSON with flexible structure recognition"""
        data = json.loads(content)

        # Handle different JSON structures
        if isinstance(data, dict):
            if 'messages' in data:
                messages = data['messages']
            elif 'conversations' in data:
                messages = data['conversations'][0].get('messages', [])
            else:
                messages = [data]
        elif isinstance(data, list):
            messages = data
        else:
            raise ValueError("Unsupported JSON structure")

        # Extract messages
        rows = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            speaker = self._extract_field(msg, ['speaker', 'sender', 'author', 'user', 'from', 'name'])
            text = self._extract_field(msg, ['text', 'message', 'content', 'body'])
            timestamp = self._extract_field(msg, ['timestamp', 'time', 'date', 'datetime', 'ts'])

            if speaker and text:
                rows.append({
                    'conversation_id': 'uploaded',
                    'message_id': i,
                    'timestamp': timestamp,
                    'speaker': str(speaker),
                    'text': str(text),
                    'turn_index': i
                })

        if not rows:
            raise ValueError("No valid messages found in JSON")

        df = pd.DataFrame(rows)
        return self._validate_and_normalize(df)

    def _parse_whatsapp(self, content: str) -> pd.DataFrame:
        """Parse WhatsApp TXT export format"""
        lines = content.split('\n')
        messages = []
        turn_index = 0

        # Pattern: DD/MM/YY, HH:MM PM - Speaker: Message
        # Supports variations in date format and separators
        pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s*(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?)\s*-\s*([^:]+):\s*(.*)'

        current_message = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip system messages
            if 'created group' in line.lower() or 'added' in line.lower() or 'left' in line.lower():
                continue

            match = re.match(pattern, line)
            if match:
                # Save previous message if exists
                if current_message:
                    messages.append(current_message)
                    turn_index += 1

                date_str, time_str, speaker, text = match.groups()
                timestamp = f"{date_str} {time_str}"

                current_message = {
                    'conversation_id': 'uploaded',
                    'message_id': turn_index,
                    'timestamp': timestamp,
                    'speaker': speaker.strip(),
                    'text': text.strip(),
                    'turn_index': turn_index
                }
            elif current_message:
                # Continuation of multiline message
                current_message['text'] += ' ' + line

        # Add last message
        if current_message:
            messages.append(current_message)

        if not messages:
            raise ValueError("No valid WhatsApp messages found")

        df = pd.DataFrame(messages)
        return self._validate_and_normalize(df)

    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column from list of possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    def _extract_field(self, obj: Dict, possible_keys: List[str]) -> Optional[str]:
        """Extract field from dict using possible keys"""
        for key in possible_keys:
            if key in obj:
                return obj[key]
        return None

    def _validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the parsed conversation"""
        # Enforce limits
        if len(df) > self.max_messages:
            raise ValueError(
                f"Conversation exceeds maximum of {self.max_messages} messages. "
                f"Please provide a smaller conversation."
            )

        # Truncate message length
        df['text'] = df['text'].str.slice(0, self.max_message_length)

        # Remove empty messages
        df = df[df['text'].str.strip() != '']

        # Ensure required columns
        required = ['conversation_id', 'message_id', 'timestamp', 'speaker', 'text', 'turn_index']
        for col in required:
            if col not in df.columns:
                df[col] = None

        # Convert timestamp if possible
        if df['timestamp'].notna().any():
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            except:
                pass

        return df.reset_index(drop=True)
