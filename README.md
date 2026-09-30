# ToneTrace

**Conversation Intelligence**

*"See the conversation before it becomes a conflict."*

---

## CEREBRO 2026 - PS-01: Tone Intelligence in Conversations

ToneTrace analyzes how tone evolves across conversations rather than independently analyzing isolated messages. It tracks sentiment progression, detects escalation, identifies turning points, and provides explainable evidence for tone shifts.

**Competition reference/submission page:** https://bagchibijoyini-a11y.github.io/prith123/

---

## Problem

Traditional sentiment analysis tools analyze messages in isolation, missing the critical context of how tone progresses across a conversation. This makes it difficult to detect escalation, identify turning points, or understand the arc of a dialogue.

## Solution

ToneTrace implements **temporal conversation intelligence**:

- **Sequential Analysis**: Compares each message with preceding context
- **Escalation Detection**: Identifies when conversations become heated
- **Turning-Point Detection**: Finds where tone shifts substantially
- **Evidence-Based**: Every detection includes supporting linguistic cues
- **Multi-Format Support**: WhatsApp, CSV, JSON, Slack-style, Discord-style

## Why Sequential Analysis Matters

A single message might appear neutral in isolation but represent escalation in context:

```
Message 1: "Sure, I can help with that." (Neutral)
Message 2: "Yeah, right. Apparently nobody reads what I say." (Escalation - sarcasm)
Message 3: "Fine. Whatever you want." (Escalation - passive-aggression)
```

ToneTrace detects this progression and shows where it occurs.

---

## Features

### Core Capabilities

- **Multi-turn chat sentiment and tone tracking**
- **Sarcasm detection** (signal-based, not diagnostic)
- **Passive-aggression detection** (signal-based, not diagnostic)
- **Irony detection** (signal-based, not diagnostic)
- **Emotion-arc visualization** (joy, sadness, anger, fear, frustration)
- **WhatsApp-style conversation support**
- **Slack-style conversation support**
- **Discord-style conversation support**
- **User-provided data upload**
- **Detection of conversational tone changes**
- **Identification of escalation**
- **Identification of significant turning points**
- **Interpretable output** showing which message contributed to tone shift

### Dashboard Pages

1. **Overview**: Conversation health, escalation metrics, emotion arc
2. **Conversation**: Chronological message view with tone analysis
3. **Conflict Signals**: Turning points, escalation timeline, largest changes
4. **Speaker Analysis**: Per-speaker statistics (message count, sentiment, volatility)
5. **Evidence**: Strongest tone change with detailed explanation
6. **Methodology**: Pipeline documentation and responsible AI notes

---

## Architecture

```
User Upload
    ↓
Conversation Parser (CSV/JSON/WhatsApp)
    ↓
Normalization
    ↓
Lightweight NLP Feature Extraction
    ↓
Message Tone Estimation
    ↓
Sequential / Temporal Analysis
    ↓
Escalation Detection
    ↓
Turning-Point Detection
    ↓
Evidence / Explainability
    ↓
Streamlit Dashboard
```

### Core Differentiator

**TEMPORAL CONVERSATION INTELLIGENCE** — not merely sentiment classification.

---

## Technology

### Stack

- **Python 3.14.6**
- **Streamlit** (UI)
- **Pandas** (data processing)
- **NumPy** (numerical operations)
- **Plotly** (visualization)

### No-API Architecture

ToneTrace is intentionally designed to run without external dependencies:

- ❌ No OpenAI API
- ❌ No Gemini API
- ❌ No Claude API
- ❌ No Anthropic API
- ❌ No API keys
- ❌ No paid services
- ❌ No external inference APIs
- ❌ No GPU
- ❌ No PyTorch
- ❌ No TensorFlow
- ❌ No Transformers
- ❌ No sentence-transformers
- ❌ No large language models
- ❌ No large embedding models
- ❌ No FAISS
- ❌ No vector database
- ❌ No Redis
- ❌ No database
- ❌ No background workers
- ❌ No model downloads at startup
- ❌ No model downloads from Hugging Face at runtime
- ❌ No model training during Render startup
- ❌ No external AI calls

### Memory-Efficient Design

Designed specifically for **free Render Web Service** (~512 MB RAM):

- Maximum upload: 5 MB
- Maximum messages: 5,000
- Maximum message length: 4,000 characters
- Small data structures
- Lazy processing
- Minimal dependencies
- No large models
- No database
- No external APIs

---

## Supported Formats

### CSV
Flexible column name recognition:
- Speaker: `speaker`, `sender`, `author`, `user`, `from`, `name`
- Text: `text`, `message`, `content`, `body`
- Timestamp: `timestamp`, `time`, `date`, `datetime`

### JSON
Supports common structures with fields:
- Speaker: `speaker`, `sender`, `author`, `user`
- Text: `text`, `message`, `content`
- Timestamp: `timestamp`, `time`, `date` (optional)

### WhatsApp TXT
Common export format:
```
12/09/26, 10:42 pm - Rahul: Hello
```
Handles multiline messages and system messages.

### Slack/Discord
CSV/JSON with flexible field recognition.

---

## Demo

A synthetic demonstration conversation is included in `data/demo_conversation.csv`.

**Important:** This is **SYNTHETIC DEMONSTRATION DATA** provided only to demonstrate the application pipeline. No claim is made that it represents a benchmark dataset or real conversation.

The demo shows a realistic progression:
- Calm → Disagreement → Frustration → Sarcasm → Passive Aggression → Escalation → De-escalation

To try the demo:
1. Run the application
2. Click "Load Demo Conversation" on the landing page

---

## Data Provenance

See `data/DATA_SOURCES.md` for detailed information about data sources.

**External datasets:** None currently used.

**Quantitative benchmark metrics:** Not reported because the included demonstration data is synthetic and is not a valid benchmark.

---

## Methodology

### Feature Extraction

Lightweight NLP features:
- Lexical features (positive/negative words)
- Structural features (punctuation, capitalization, emojis)
- Pattern features (disagreement markers, blame indicators)
- Cue features (sarcasm, irony, passive-aggression cues)

### Tone Estimation

All signals are **estimated linguistic signals** (0 to 1 intensity):
- **Sentiment**: Positive/neutral/negative (-1 to +1 score)
- **Emotions**: Joy, sadness, anger, fear, frustration
- **Sarcasm**: Based on positive wording in negative context, cues
- **Irony**: Based on semantic contrast, contradiction patterns
- **Passive-Aggression**: Based on dismissive agreement, indirect blame
- **Hostility**: Based on aggressive vocabulary, direct blame

### Temporal Analysis

- **Deltas**: Change from previous message
- **Rolling averages**: Temporal smoothing (5-message window)
- **Escalation score**: Weighted combination of signals (0-100)
- **Turning points**: Significant tone changes relative to recent context

### Escalation Formula

```
Escalation = (
    sentiment_decrease * 15 +
    negative_sentiment * 10 +
    anger_increase * 20 +
    anger_level * 15 +
    frustration_increase * 15 +
    frustration_level * 10 +
    hostility_increase * 15 +
    hostility_level * 15 +
    sarcasm_signal * 10 +
    passive_aggression_signal * 10 +
    volatility * 20
)
```

Clamped to 0-100. This formula is **not clinically validated**.

---

## Responsible AI

**Important Limitations:**

- ToneTrace estimates **linguistic signals**, not a person's true internal emotional state
- Sarcasm and passive-aggression detection are **uncertain and context-dependent**
- Language, culture, dialect, and code-switching can affect results
- Results should **not** be used as psychological diagnosis
- Results should **not** be used for employment, disciplinary, legal, or other high-stakes decisions
- The system is intentionally lightweight and API-free
- The escalation formula is not clinically validated

ToneTrace is designed for **exploratory conversation analysis**, not for high-stakes decision-making.

---

## Limitations

- **Context window**: Limited to recent 5 messages for temporal smoothing
- **Language**: Primarily designed for English; other languages may have reduced accuracy
- **Cultural differences**: Sarcasm and indirect communication patterns vary across cultures
- **Ambiguity**: Some messages are genuinely ambiguous; the system makes best-effort estimates
- **No training**: The system uses rule-based features, not trained models
- **Memory constraint**: Limited to 5,000 messages for free hosting
- **File size**: Limited to 5 MB uploads

---

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/Sreevalli20/cere.git
cd cere

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Requirements

- Python 3.14.6
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.17.0

---

## Render Deployment

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

### Configuration

See `render.yaml` for full Render configuration.

**Important:** The application requires **ZERO secrets** and runs entirely on free tier.

---

## Testing

Run the test suite:

```bash
# Install pytest (if not already installed)
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_parser.py -v
```

### Test Coverage

- `test_parser.py`: CSV, JSON, WhatsApp parsing, format detection, limits
- `test_engine.py`: Sentiment, emotion, sarcasm, hostility, signal ranges
- `test_temporal.py`: Temporal analysis, escalation, turning points, speaker analysis

---

## Project Structure

```
cere/
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── .python-version             # Python version specification
├── render.yaml                 # Render deployment config
├── README.md                   # This file
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── parser.py               # Conversation parsers
│   ├── features.py             # Feature extraction
│   ├── engine.py               # Tone analysis engine
│   ├── temporal.py             # Temporal analysis
│   ├── explainability.py       # Evidence extraction
│   └── demo.py                 # Demo data generator
├── data/
│   ├── demo_conversation.csv   # Synthetic demo data
│   └── DATA_SOURCES.md         # Data provenance
├── tests/
│   ├── test_parser.py
│   ├── test_engine.py
│   └── test_temporal.py
└── .streamlit/
    └── config.toml             # Streamlit configuration
```

---

## Future Improvements

Potential enhancements (not implemented in current version):

- **Multi-language support**: Expand beyond English
- **Larger context windows**: Configurable temporal windows
- **Custom lexicons**: User-defined word lists
- **Export functionality**: Download analysis reports
- **Conversation comparison**: Compare multiple conversations
- **Real-time analysis**: WebSocket support for live conversations
- **Advanced visualization**: Network graphs, speaker interaction maps

---

## Team

**CEREBRO 2026 Submission**

**Repository:** https://github.com/Sreevalli20/cere

**Competition reference:** https://bagchibijoyini-a11y.github.io/prith123/

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- CEREBRO 2026 organizers for the competition
- Streamlit for the excellent UI framework
- The open-source community for NLP tools and resources
