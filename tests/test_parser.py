"""
Tests for Conversation Parser
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import ConversationParser


def test_csv_parser():
    """Test CSV parsing with flexible column names"""
    parser = ConversationParser()

    csv_content = """speaker,text,timestamp
Alice,Hello world,2026-09-30 10:00:00
Bob,Hi there,2026-09-30 10:01:00
"""

    df = parser._parse_csv(csv_content)

    assert len(df) == 2
    assert 'speaker' in df.columns
    assert 'text' in df.columns
    assert df.iloc[0]['speaker'] == 'Alice'
    assert df.iloc[0]['text'] == 'Hello world'


def test_csv_alternative_columns():
    """Test CSV with alternative column names"""
    parser = ConversationParser()

    csv_content = """author,message,date
Alice,Hello world,2026-09-30 10:00:00
Bob,Hi there,2026-09-30 10:01:00
"""

    df = parser._parse_csv(csv_content)

    assert len(df) == 2
    assert df.iloc[0]['speaker'] == 'Alice'


def test_json_parser():
    """Test JSON parsing"""
    parser = ConversationParser()

    json_content = """[
        {"speaker": "Alice", "text": "Hello", "timestamp": "2026-09-30 10:00:00"},
        {"speaker": "Bob", "text": "Hi", "timestamp": "2026-09-30 10:01:00"}
    ]"""

    df = parser._parse_json(json_content)

    assert len(df) == 2
    assert df.iloc[0]['speaker'] == 'Alice'
    assert df.iloc[0]['text'] == 'Hello'


def test_json_alternative_fields():
    """Test JSON with alternative field names"""
    parser = ConversationParser()

    json_content = """[
        {"author": "Alice", "content": "Hello"},
        {"author": "Bob", "content": "Hi"}
    ]"""

    df = parser._parse_json(json_content)

    assert len(df) == 2
    assert df.iloc[0]['speaker'] == 'Alice'


def test_whatsapp_parser():
    """Test WhatsApp TXT parsing"""
    parser = ConversationParser()

    whatsapp_content = """12/09/26, 10:42 pm - Alice: Hello world
12/09/26, 10:43 pm - Bob: Hi there
"""

    df = parser._parse_whatsapp(whatsapp_content)

    assert len(df) == 2
    assert df.iloc[0]['speaker'] == 'Alice'
    assert df.iloc[0]['text'] == 'Hello world'


def test_whatsapp_multiline():
    """Test WhatsApp with multiline messages"""
    parser = ConversationParser()

    whatsapp_content = """12/09/26, 10:42 pm - Alice: Hello world
This is a continuation
12/09/26, 10:43 pm - Bob: Hi there
"""

    df = parser._parse_whatsapp(whatsapp_content)

    assert len(df) == 2
    assert 'continuation' in df.iloc[0]['text']


def test_whatsapp_system_messages():
    """Test WhatsApp with system messages (should be skipped)"""
    parser = ConversationParser()

    whatsapp_content = """12/09/26, 10:42 pm - Alice: Hello
12/09/26, 10:43 pm - Messages to this chat are now secured with end-to-end encryption.
12/09/26, 10:44 pm - Bob: Hi
"""

    df = parser._parse_whatsapp(whatsapp_content)

    assert len(df) == 2
    assert df.iloc[0]['speaker'] == 'Alice'
    assert df.iloc[1]['speaker'] == 'Bob'


def test_max_messages_limit():
    """Test that max messages limit is enforced"""
    parser = ConversationParser(max_messages=5)

    csv_content = "\n".join([f"speaker{i},text{i},2026-09-30 10:0{i}:00" for i in range(10)])

    with pytest.raises(ValueError, match="exceeds maximum"):
        parser._parse_csv(csv_content)


def test_max_message_length():
    """Test that message length is truncated"""
    parser = ConversationParser(max_message_length=10)

    csv_content = """speaker,text,timestamp
Alice,This is a very long message that should be truncated,2026-09-30 10:00:00
"""

    df = parser._parse_csv(csv_content)

    assert len(df.iloc[0]['text']) <= 10


def test_empty_messages_removed():
    """Test that empty messages are removed"""
    parser = ConversationParser()

    csv_content = """speaker,text,timestamp
Alice,Hello,2026-09-30 10:00:00
Bob,   ,2026-09-30 10:01:00
Charlie,World,2026-09-30 10:02:00
"""

    df = parser._parse_csv(csv_content)

    assert len(df) == 2


def test_auto_detect_format():
    """Test auto-detection of format"""
    parser = ConversationParser()

    # Should detect as JSON
    json_content = '[{"speaker": "Alice", "text": "Hello"}]'
    df = parser.parse(json_content, 'test.json')
    assert len(df) == 1

    # Should detect as WhatsApp
    whatsapp_content = '12/09/26, 10:42 pm - Alice: Hello'
    df = parser.parse(whatsapp_content, 'test.txt')
    assert len(df) == 1

    # Should detect as CSV
    csv_content = 'speaker,text\nAlice,Hello'
    df = parser.parse(csv_content, 'test.csv')
    assert len(df) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
