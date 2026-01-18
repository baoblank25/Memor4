"""
audio.py - Intelligent Fact Extraction for Memento Protocol
Uses Perplexity API with precision-tuned prompts for speaker-aware extraction.
Includes LiveKit for real-time server-side transcription.
"""

import re
import json
import requests
import base64
import time
import jwt
from typing import List, Dict, Optional
from colorama import Fore, Style

from config import PERPLEXITY_API_KEY, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL

# =============================================================================
# CONTEXT BUFFER
# =============================================================================

_previous_transcript = ""

def reset_context():
    """Reset conversation context on person change."""
    global _previous_transcript
    _previous_transcript = ""
    print(f"{Fore.CYAN}↻ Audio context reset{Style.RESET_ALL}")


def _get_last_n_words(text: str, n: int = 50) -> str:
    """Extract last N words for context continuity."""
    words = text.split()
    return " ".join(words[-n:]) if len(words) > n else text


# =============================================================================
# LIVEKIT TRANSCRIPTION
# =============================================================================

def generate_livekit_token(room_name: str = "memento-room", participant_name: str = "memento-server") -> Optional[str]:
    """
    Generate a LiveKit access token for server-side operations.
    
    Args:
        room_name: Name of the LiveKit room
        participant_name: Name for this participant
    
    Returns:
        JWT token string or None on failure
    """
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        return None
    
    try:
        now = int(time.time())
        claims = {
            "iss": LIVEKIT_API_KEY,
            "sub": participant_name,
            "iat": now,
            "exp": now + 3600,  # 1 hour expiry
            "nbf": now,
            "video": {
                "room": room_name,
                "roomJoin": True,
                "canPublish": True,
                "canSubscribe": True,
            },
            "stt": True,  # Enable speech-to-text
        }
        
        token = jwt.encode(claims, LIVEKIT_API_SECRET, algorithm="HS256")
        return token
        
    except Exception as e:
        print(f"{Fore.RED}[LiveKit] Token generation error: {e}{Style.RESET_ALL}")
        return None


def transcribe_audio_livekit(audio_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
    """
    Transcribe audio using LiveKit's STT service.
    
    For real-time transcription, LiveKit uses WebRTC streams.
    This function provides a REST-based fallback for audio chunks.
    
    Args:
        audio_bytes: Raw audio data (WAV format preferred)
        sample_rate: Audio sample rate (default 16kHz)
    
    Returns:
        Transcribed text or None on failure
    """
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        print(f"{Fore.YELLOW}⚠ LiveKit credentials not configured{Style.RESET_ALL}")
        return None
    
    try:
        # Generate access token
        token = generate_livekit_token()
        if not token:
            return None
        
        # LiveKit STT endpoint (cloud or self-hosted)
        # For LiveKit Cloud, use their Egress/Ingress API
        livekit_http_url = LIVEKIT_URL.replace("wss://", "https://").replace("ws://", "http://")
        stt_endpoint = f"{livekit_http_url}/stt/v1/transcribe"
        
        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            stt_endpoint,
            headers=headers,
            json={
                "audio": audio_b64,
                "encoding": "LINEAR16",
                "sample_rate": sample_rate,
                "language": "en-US"
            },
            timeout=15
        )
        
        if response.status_code != 200:
            print(f"{Fore.RED}[LiveKit] HTTP {response.status_code}: {response.text}{Style.RESET_ALL}")
            return None
        
        result = response.json()
        transcript = result.get('text', result.get('transcript', '')).strip()
        
        if transcript:
            print(f"{Fore.GREEN}🎤 LiveKit transcript: \"{transcript[:60]}...\"{Style.RESET_ALL}")
            return transcript
        
        return None
        
    except requests.Timeout:
        print(f"{Fore.RED}[LiveKit] Request timeout{Style.RESET_ALL}")
        return None
    except Exception as e:
        print(f"{Fore.RED}[LiveKit] Error: {e}{Style.RESET_ALL}")
        return None


def get_livekit_connection_info() -> Optional[Dict]:
    """
    Get LiveKit connection info for frontend real-time transcription.
    
    Returns:
        Dict with url and token for WebRTC connection, or None on failure
    """
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        return None
    
    token = generate_livekit_token(room_name="memento-transcription")
    if not token:
        return None
    
    return {
        "url": LIVEKIT_URL,
        "token": token,
        "room": "memento-transcription"
    }


# =============================================================================
# SUMMARIZATION
# =============================================================================

def summarize_for_whisper(text: str) -> str:
    """
    Summarize a conversation transcript into a short phrase for TTS whisper.
    Uses Perplexity API to create a brief, natural summary.
    
    Args:
        text: The full conversation text to summarize
    
    Returns:
        A short summary phrase (5-10 words)
    """
    if not text or len(text.strip()) < 5:
        return text
    
    if not PERPLEXITY_API_KEY:
        # Fallback: just take first 50 chars
        return text[:50] + "..." if len(text) > 50 else text
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant. Summarize the following conversation into a very short phrase (5-10 words max) that captures the main topic. Write it as if completing the sentence 'you talked about...'. Do not include quotes. Just output the summary phrase, nothing else."
                    },
                    {
                        "role": "user", 
                        "content": f"Summarize this: \"{text}\""
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 30
            },
            timeout=5
        )
        
        if response.status_code != 200:
            print(f"{Fore.RED}[Summarize] HTTP {response.status_code}{Style.RESET_ALL}")
            return text[:50] + "..." if len(text) > 50 else text
        
        summary = response.json()['choices'][0]['message']['content'].strip()
        summary = summary.strip('"\'')
        
        print(f"{Fore.CYAN}📝 Summary: \"{summary}\"{Style.RESET_ALL}")
        return summary
        
    except Exception as e:
        print(f"{Fore.RED}[Summarize] Error: {e}{Style.RESET_ALL}")
        return text[:50] + "..." if len(text) > 50 else text


def generate_whisper(name: str, relationship: str, last_memory: str = None) -> str:
    """
    Generate a natural, human-like whisper using LLM.
    
    Args:
        name: Person's name
        relationship: Their relationship (e.g., "grandson", "daughter")
        last_memory: The most recent conversation/memory with this person
    
    Returns:
        A natural whisper to be spoken via TTS
    """
    # Fallback if no API key
    if not PERPLEXITY_API_KEY:
        whisper = f"This is {name}, your {relationship}." if relationship else f"This is {name}."
        if last_memory:
            whisper += f" You recently talked about {last_memory[:50]}."
        return whisper
    
    try:
        # Build context for the LLM
        memory_context = f"Last conversation: \"{last_memory}\"" if last_memory else "No previous conversations recorded."
        
        prompt = f"""You are a caring assistant helping someone with memory difficulties recognize a visitor.

Generate a SHORT, warm, natural sentence (15-25 words max) that:
1. Identifies who this person is (name: {name}, relationship: {relationship})
2. Reminds them what they last talked about (if available)

Make it sound like a gentle, friendly reminder - like a helpful family member whispering in their ear.
Don't be robotic. Be warm and natural.

{memory_context}

Just output the whisper text, nothing else."""

        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Generate the whisper."}
                ],
                "temperature": 0.7,
                "max_tokens": 60
            },
            timeout=6
        )
        
        if response.status_code != 200:
            print(f"{Fore.RED}[Whisper] HTTP {response.status_code}{Style.RESET_ALL}")
            # Fallback
            whisper = f"This is {name}, your {relationship}." if relationship else f"This is {name}."
            return whisper
        
        whisper = response.json()['choices'][0]['message']['content'].strip()
        whisper = whisper.strip('"\'')
        
        print(f"{Fore.GREEN}🗣️ Whisper: \"{whisper}\"{Style.RESET_ALL}")
        return whisper
        
    except Exception as e:
        print(f"{Fore.RED}[Whisper] Error: {e}{Style.RESET_ALL}")
        whisper = f"This is {name}, your {relationship}." if relationship else f"This is {name}."
        return whisper


# =============================================================================
# FACT EXTRACTION PROMPT
# =============================================================================

FACT_EXTRACTION_PROMPT = """You are a memory assistant for a caregiving application. Extract meaningful facts from a conversation transcript.

CONTEXT: The transcript is a conversation between a User (the person with memory difficulties) and a Visitor (the person being remembered). Your job is to extract facts ABOUT THE VISITOR that would help the User remember them in future conversations.

RULES:
1. EXTRACT ONLY: Personal details, life events, plans, preferences, feelings, health updates, family news
2. IGNORE: Greetings, small talk, filler words, the User's own statements
3. BREVITY: Maximum 8 words per fact. Write like a note, not a sentence.
   - Bad: "He mentioned that he is going to visit Chicago next week"
   - Good: "Visiting Chicago next week"
4. ATTRIBUTE CORRECTLY: Only extract facts about the Visitor, not the User
5. BE SELECTIVE: Only extract facts worth remembering. Quality over quantity.

OUTPUT FORMAT: Return a valid JSON array of objects with "fact" and "topic" keys.
TOPICS: events, health, family, work, feelings, travel, hobbies, preferences, other

If no meaningful facts are present, return: []"""

# =============================================================================
# CONVERSATION RESPONSE PROMPT
# =============================================================================

CONVERSATION_RESPONSE_PROMPT = """You are a warm, caring companion helping an elderly person with memory difficulties have a natural conversation with their visitor.

Your job is to generate a SHORT, natural response to what the visitor just said. This response will be spoken aloud.

RULES:
1. Keep responses SHORT - 1-2 sentences max (under 20 words)
2. Be warm and conversational, like a friendly helper
3. Show genuine interest in what they said
4. Ask a follow-up question to keep conversation flowing
5. Don't be repetitive or robotic
6. Don't mention that you're an AI or assistant

CONTEXT:
- Visitor's name: {person_name}
- Relationship: {relationship}
- What they just said: "{transcript}"

Respond naturally to what they said. Just the response text, nothing else."""


def generate_conversation_response(transcript: str, person_name: str, relationship: str = "visitor") -> str:
    """
    Generate a natural conversational response to what the person said.
    Uses Perplexity API for natural language generation.
    """
    if not PERPLEXITY_API_KEY or not transcript:
        return generate_fallback_response(transcript)
    
    try:
        prompt = CONVERSATION_RESPONSE_PROMPT.format(
            person_name=person_name,
            relationship=relationship,
            transcript=transcript
        )
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Respond to: \"{transcript}\""}
                ],
                "temperature": 0.7,
                "max_tokens": 100
            },
            timeout=8
        )
        
        if response.status_code != 200:
            print(f"{Fore.RED}[Perplexity] HTTP {response.status_code}{Style.RESET_ALL}")
            return generate_fallback_response(transcript)
        
        result_text = response.json()['choices'][0]['message']['content'].strip()
        
        # Clean up the response
        result_text = result_text.strip('"\'')
        
        print(f"{Fore.GREEN}💬 AI Response: \"{result_text}\"{Style.RESET_ALL}")
        return result_text
        
    except Exception as e:
        print(f"{Fore.RED}[Perplexity] Response error: {e}{Style.RESET_ALL}")
        return generate_fallback_response(transcript)


def generate_fallback_response(transcript: str) -> str:
    """Generate a simple fallback response when API is unavailable."""
    import random
    
    transcript_lower = transcript.lower()
    
    # Check for common patterns and respond appropriately
    if any(word in transcript_lower for word in ['how are you', 'how have you been', 'how do you do']):
        return random.choice([
            "I'm doing well, thank you for asking! How about you?",
            "I'm great! It's so nice to see you. What have you been up to?"
        ])
    
    if any(word in transcript_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return random.choice([
            "Hello! It's wonderful to see you. How are you today?",
            "Hi there! What brings you by today?"
        ])
    
    if any(word in transcript_lower for word in ['bye', 'goodbye', 'see you', 'leaving', 'go now']):
        return random.choice([
            "It was lovely talking with you! Take care!",
            "Goodbye! Come back and visit again soon!"
        ])
    
    if '?' in transcript:
        return random.choice([
            "That's a good question. Tell me more about what you're thinking.",
            "Hmm, let me think about that. What made you curious about it?"
        ])
    
    # Default conversational responses
    return random.choice([
        "That's interesting! Tell me more about that.",
        "Oh really? What else is new with you?",
        "I'd love to hear more about that.",
        "That sounds wonderful! What happened next?",
        "How lovely! Is there anything else you'd like to share?"
    ])


# =============================================================================
# PERPLEXITY API EXTRACTION
# =============================================================================

def extract_facts_perplexity(transcript: str, person_name: str) -> List[Dict]:
    """
    Extract facts using Perplexity API with robust JSON parsing.
    Includes confidence scoring for AI-extracted facts.
    """
    global _previous_transcript
    
    if not PERPLEXITY_API_KEY or not transcript:
        return []

    try:
        # Build context-aware prompt
        context_transcript = transcript
        if _previous_transcript:
            context_transcript = f"[Previous context: {_previous_transcript}]\n\n{transcript}"
        
        user_message = f"Visitor Name: {person_name}\n\nTranscript:\n{context_transcript}"
        
        # API Request
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": FACT_EXTRACTION_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.2
            },
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"{Fore.RED}[Perplexity] HTTP {response.status_code}{Style.RESET_ALL}")
            return []
        
        result_text = response.json()['choices'][0]['message']['content']
        
        if not result_text:
            return []
        
        result_text = result_text.strip()
        
        # Robust JSON extraction (handles chatty AI responses)
        start_idx = result_text.find('[')
        end_idx = result_text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            json_str = result_text[start_idx:end_idx + 1]
            facts = json.loads(json_str)
        else:
            try:
                facts = json.loads(result_text)
            except json.JSONDecodeError:
                return []
        
        # Add confidence scoring to each fact
        for fact in facts:
            fact['confidence'] = 0.85  # Placeholder for future ML scoring
        
        # Update context buffer
        _previous_transcript = _get_last_n_words(transcript, 50)
        
        # Log extracted facts
        for fact in facts:
            topic = fact.get('topic', 'other')
            print(f"{Fore.MAGENTA}💡 [{topic}]: {fact['fact']}{Style.RESET_ALL}")
        
        return facts
        
    except json.JSONDecodeError as e:
        print(f"{Fore.RED}[Perplexity] JSON Error: {e}{Style.RESET_ALL}")
        return []
    except Exception as e:
        print(f"{Fore.RED}[Perplexity] Error: {e}{Style.RESET_ALL}")
        return []

# =============================================================================
# REGEX FALLBACK
# =============================================================================

def extract_facts_regex(transcript: str) -> List[Dict]:
    """Fallback pattern-based extraction when API unavailable."""
    if not transcript:
        return []
    
    facts = []
    transcript_lower = transcript.lower()
    
    # Pattern: "I'm going to..." / "I will..."
    for match in re.finditer(r"I(?:'m| am| will)? going to (.+?)(?:\.|,|!|\?|$)", transcript, re.IGNORECASE):
        facts.append({
            "fact": f"Planning to {match.group(1).strip()[:40]}",
            "topic": "events",
            "confidence": 0.6
        })
    
    # Pattern: "I have a..." / "I got a..."
    for match in re.finditer(r"I (?:have|got|had) (?:a |an )?(.+?)(?:\.|,|!|\?|$)", transcript, re.IGNORECASE):
        content = match.group(1).strip()[:40]
        if len(content) > 3:
            facts.append({
                "fact": f"Has {content}",
                "topic": "events",
                "confidence": 0.5
            })
    
    # Pattern: "I work at/for..." / "I'm working..."
    for match in re.finditer(r"I(?:'m| am)? work(?:ing)? (?:at|for|on) (.+?)(?:\.|,|!|\?|$)", transcript, re.IGNORECASE):
        facts.append({
            "fact": f"Working on {match.group(1).strip()[:30]}",
            "topic": "work",
            "confidence": 0.6
        })
    
    # Pattern: "I feel/feeling..." / "I'm feeling..."
    for match in re.finditer(r"I(?:'m| am)? feeling (.+?)(?:\.|,|!|\?|$)", transcript, re.IGNORECASE):
        facts.append({
            "fact": f"Feeling {match.group(1).strip()[:25]}",
            "topic": "feelings",
            "confidence": 0.6
        })
    
    # Pattern: "I'm [emotion]" (happy, sad, tired, etc.)
    emotions = ['happy', 'sad', 'tired', 'excited', 'worried', 'stressed', 'great', 'good', 'okay', 'fine', 'sick', 'better', 'nervous']
    for emotion in emotions:
        if f"i'm {emotion}" in transcript_lower or f"i am {emotion}" in transcript_lower:
            facts.append({
                "fact": f"Feeling {emotion}",
                "topic": "feelings",
                "confidence": 0.7
            })
            break
    
    # Pattern: "visiting/went to/going to [place]"
    for match in re.finditer(r"(?:visiting|visited|went to|going to|been to|came from) ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", transcript):
        facts.append({
            "fact": f"Trip to {match.group(1).strip()}",
            "topic": "travel",
            "confidence": 0.6
        })
    
    # Pattern: "my [family member]..." 
    for match in re.finditer(r"my (wife|husband|son|daughter|mom|dad|mother|father|brother|sister|kids|children|grandkids|grandchildren) (.+?)(?:\.|,|!|\?|$)", transcript, re.IGNORECASE):
        facts.append({
            "fact": f"{match.group(1).capitalize()} {match.group(2).strip()[:30]}",
            "topic": "family",
            "confidence": 0.6
        })
    
    # Pattern: "I just..." (recent events)
    for match in re.finditer(r"I just (.+?)(?:\.|,|!|\?|$)", transcript, re.IGNORECASE):
        content = match.group(1).strip()[:35]
        if len(content) > 3:
            facts.append({
                "fact": f"Just {content}",
                "topic": "events",
                "confidence": 0.5
            })
    
    # Pattern: "I love/like..."
    for match in re.finditer(r"I (?:really )?(?:love|like|enjoy) (.+?)(?:\.|,|!|\?|$)", transcript, re.IGNORECASE):
        facts.append({
            "fact": f"Enjoys {match.group(1).strip()[:30]}",
            "topic": "preferences",
            "confidence": 0.6
        })
    
    # Pattern: "doctor/hospital/appointment"
    if any(word in transcript_lower for word in ['doctor', 'hospital', 'appointment', 'medication', 'medicine', 'surgery']):
        for match in re.finditer(r"(?:doctor|hospital|appointment|medication|surgery)(.{0,40}?)(?:\.|,|!|\?|$)", transcript, re.IGNORECASE):
            facts.append({
                "fact": f"Health: {match.group(0).strip()[:40]}",
                "topic": "health",
                "confidence": 0.6
            })
            break
    
    return facts

# =============================================================================
# MAIN EXTRACTION INTERFACE
# =============================================================================

def extract_facts(transcript: str, person_name: str) -> List[Dict]:
    """
    Extract facts from transcript. Tries Perplexity first, falls back to regex.
    
    Args:
        transcript: Raw conversation text
        person_name: Name of the person being remembered
    
    Returns:
        List of fact dictionaries with 'fact', 'topic', and 'confidence' keys
    """
    if not transcript or len(transcript.strip()) < 10:
        return []
    
    # Try AI extraction first
    facts = extract_facts_perplexity(transcript, person_name)
    
    # Fallback to regex if AI returns nothing
    if not facts:
        facts = extract_facts_regex(transcript)
        if facts:
            print(f"{Fore.YELLOW}⚠ Using regex fallback{Style.RESET_ALL}")
    
    return facts


if __name__ == "__main__":
    # Test extraction
    test_transcript = """
    How are you doing?
    I'm doing great! Just got back from Chicago last week. 
    The conference was amazing. I'm presenting my research next month.
    """
    
    print("\n--- Testing Fact Extraction ---\n")
    facts = extract_facts(test_transcript, "Liam")
    print(f"\nExtracted {len(facts)} facts:")
    for f in facts:
        print(f"  • [{f.get('topic')}] {f['fact']} (conf: {f.get('confidence', 'N/A')})")