"""
Demo Data Generator for ToneTrace
Creates synthetic demonstration conversation
"""

import pandas as pd
from pathlib import Path


def get_demo_conversation() -> pd.DataFrame:
    """
    Return synthetic demonstration conversation
    Progression: calm → disagreement → frustration → sarcasm → passive aggression → escalation → de-escalation
    """
    demo_data = [
        {
            'conversation_id': 'demo',
            'message_id': 0,
            'timestamp': '2026-09-30 10:00:00',
            'speaker': 'Alice',
            'text': 'Hey! I wanted to discuss the project timeline. Do you have a moment?',
            'turn_index': 0
        },
        {
            'conversation_id': 'demo',
            'message_id': 1,
            'timestamp': '2026-09-30 10:01:00',
            'speaker': 'Bob',
            'text': 'Sure, I have some time. What\'s on your mind?',
            'turn_index': 1
        },
        {
            'conversation_id': 'demo',
            'message_id': 2,
            'timestamp': '2026-09-30 10:02:00',
            'speaker': 'Alice',
            'text': 'I think we need to adjust the deadline. The current timeline seems too tight given the scope.',
            'turn_index': 2
        },
        {
            'conversation_id': 'demo',
            'message_id': 3,
            'timestamp': '2026-09-30 10:03:00',
            'speaker': 'Bob',
            'text': 'I disagree. We\'ve already committed to this timeline with the client.',
            'turn_index': 3
        },
        {
            'conversation_id': 'demo',
            'message_id': 4,
            'timestamp': '2026-09-30 10:04:00',
            'speaker': 'Alice',
            'text': 'But the requirements changed since then. We can\'t realistically deliver everything by then.',
            'turn_index': 4
        },
        {
            'conversation_id': 'demo',
            'message_id': 5,
            'timestamp': '2026-09-30 10:05:00',
            'speaker': 'Bob',
            'text': 'That\'s not my problem. You should have raised this earlier.',
            'turn_index': 5
        },
        {
            'conversation_id': 'demo',
            'message_id': 6,
            'timestamp': '2026-09-30 10:06:00',
            'speaker': 'Alice',
            'text': 'I did raise it, but nobody listened. This is frustrating.',
            'turn_index': 6
        },
        {
            'conversation_id': 'demo',
            'message_id': 7,
            'timestamp': '2026-09-30 10:07:00',
            'speaker': 'Bob',
            'text': 'Yeah, sure. Apparently nobody reads what I say either.',
            'turn_index': 7
        },
        {
            'conversation_id': 'demo',
            'message_id': 8,
            'timestamp': '2026-09-30 10:08:00',
            'speaker': 'Alice',
            'text': 'Of course. Great communication we have here.',
            'turn_index': 8
        },
        {
            'conversation_id': 'demo',
            'message_id': 9,
            'timestamp': '2026-09-30 10:09:00',
            'speaker': 'Bob',
            'text': 'Fine. Whatever you want. Do whatever you think is best.',
            'turn_index': 9
        },
        {
            'conversation_id': 'demo',
            'message_id': 10,
            'timestamp': '2026-09-30 10:10:00',
            'speaker': 'Alice',
            'text': 'If you say so. I\'ll just work with what we have.',
            'turn_index': 10
        },
        {
            'conversation_id': 'demo',
            'message_id': 11,
            'timestamp': '2026-09-30 10:11:00',
            'speaker': 'Bob',
            'text': 'No problem. Thanks for nothing.',
            'turn_index': 11
        },
        {
            'conversation_id': 'demo',
            'message_id': 12,
            'timestamp': '2026-09-30 10:12:00',
            'speaker': 'Alice',
            'text': 'This is ridiculous. We need to have a proper conversation about this.',
            'turn_index': 12
        },
        {
            'conversation_id': 'demo',
            'message_id': 13,
            'timestamp': '2026-09-30 10:13:00',
            'speaker': 'Bob',
            'text': 'You\'re the one making it difficult!',
            'turn_index': 13
        },
        {
            'conversation_id': 'demo',
            'message_id': 14,
            'timestamp': '2026-09-30 10:14:00',
            'speaker': 'Alice',
            'text': 'Wait, let\'s take a step back. I don\'t want this to turn into an argument.',
            'turn_index': 14
        },
        {
            'conversation_id': 'demo',
            'message_id': 15,
            'timestamp': '2026-09-30 10:15:00',
            'speaker': 'Bob',
            'text': 'You\'re right. I apologize for getting heated. Let\'s discuss this constructively.',
            'turn_index': 15
        },
        {
            'conversation_id': 'demo',
            'message_id': 16,
            'timestamp': '2026-09-30 10:16:00',
            'speaker': 'Alice',
            'text': 'Thank you. I appreciate that. Let\'s figure out a realistic solution together.',
            'turn_index': 16
        },
        {
            'conversation_id': 'demo',
            'message_id': 17,
            'timestamp': '2026-09-30 10:17:00',
            'speaker': 'Bob',
            'text': 'Agreed. Let me review the requirements again and we can discuss options.',
            'turn_index': 17
        },
    ]

    df = pd.DataFrame(demo_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def save_demo_conversation(output_path: str = 'data/demo_conversation.csv'):
    """Save demo conversation to CSV file"""
    df = get_demo_conversation()
    df.to_csv(output_path, index=False)
    return df


if __name__ == '__main__':
    # Save demo conversation when run directly
    save_demo_conversation()
    print("Demo conversation saved to data/demo_conversation.csv")
