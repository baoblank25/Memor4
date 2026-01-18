# Momento

A memory companion for caregiving. Helps people with memory difficulties recognize family members by detecting who's in front of the camera and whispering helpful reminders about them.

## What it does

Point the camera at someone wearing a specific color, and Momento will:
- Recognize them based on their clothing color (red, blue, green)
- Show their name and relationship
- Read out memories about them ("That's Liam, your grandson. He's building an AI drone project.")

The dashboard lets caregivers manage family members and add notes/memories that get spoken when that person is recognized.

## Quick start

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up config
cp .env.example .env

# Seed demo data (optional but recommended)
python seed_db.py

# Run the server
python server.py
```

Open http://localhost:8000

## Demo mode

The seeder creates 3 demo people:
- **Red** → Liam (grandson)
- **Blue** → Sarah (daughter)
- **Green** → Dr. Chen (doctor)

Click "Start Session", allow camera access, and hold up something red/blue/green to trigger recognition.

## API Keys

For full functionality, add these to your `.env`:

```
OVERSHOOT_API_KEY=     # Vision API (optional - falls back to local color detection)
PERPLEXITY_API_KEY=    # AI fact extraction from conversations
ELEVENLABS_API_KEY=    # Voice synthesis
WISPR_API_KEY=         # Audio transcription
```

Works without them using local fallbacks (OpenCV color detection + browser TTS).

## Tech

- **Backend**: FastAPI + SQLite
- **Frontend**: Vanilla JS single-page app
- **Vision**: Overshoot API with OpenCV fallback
- **Audio**: Web Speech API for recognition, ElevenLabs/pyttsx3 for TTS

---

Built at Overshoot hackathon.
