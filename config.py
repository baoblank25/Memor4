"""
config.py - Configuration loader with validation
Loads API keys, defines timing constants, prints colored status
"""

import os
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# API KEYS (default to None if not present)
# =============================================================================
OVERSHOOT_API_KEY = os.getenv("OVERSHOOT_API_KEY")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://your-project.livekit.cloud")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# =============================================================================
# API ENDPOINTS
# =============================================================================
OVERSHOOT_ENDPOINT = "https://api.overshoot.ai/v1/detect"
ELEVENLABS_VOICE_ID = "pFZP5JQG7iQjIQuC4Bku"  # Lily - gentle girl
ELEVENLABS_ENDPOINT = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

# =============================================================================
# TIMING CONSTANTS
# =============================================================================
PERSON_COOLDOWN_SECONDS = 10        # Don't re-announce same person within this window
DETECTION_CONFIDENCE_FRAMES = 2     # Must detect person N times before announcing
ABSENCE_RESET_SECONDS = 5           # Person must be absent this long to be re-announced
RECORDING_DURATION_SECONDS = 10     # Audio recording duration
SAMPLE_RATE = 16000                 # Audio sample rate

# =============================================================================
# DATABASE
# =============================================================================
DB_PATH = "memories.db"

# =============================================================================
# DEMO MODE
# =============================================================================
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# =============================================================================
# COLOR MAPPING (for vision fallback)
# =============================================================================
COLOR_TO_PERSON = {
    "red": "liam",
    "blue": "sarah",
    "green": "doctor",
    "black": "black_person"  # TODO: Update this to the actual person's ID
}

# =============================================================================
# FACE RECOGNITION
# =============================================================================
FACE_RECOGNITION_ENABLED = True
FACE_MODEL_NAME = "Facenet"           # 128-d embeddings, fast and accurate
FACE_DETECTOR_BACKEND = "retinaface"  # More accurate detection (opencv can miss faces)
FACE_SIMILARITY_THRESHOLD = 0.6       # Cosine similarity threshold for match
FACE_IMAGES_DIR = "static/faces"      # Directory to store enrolled face images


def _check_key(key_value: str, key_name: str) -> bool:
    """Check if API key is configured (not None and not placeholder)."""
    if key_value is None:
        return False
    if "your_" in key_value.lower() or key_value.strip() == "":
        return False
    return True


def validate_config() -> dict:
    """
    Validate configuration and print colored status.
    Returns dict of {service: bool} indicating availability.
    """
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== MEMENTO PROTOCOL CONFIG ==={Style.RESET_ALL}\n")
    
    status = {}
    
    # Vision (Overshoot)
    if _check_key(OVERSHOOT_API_KEY, "OVERSHOOT_API_KEY"):
        print(f"  {Fore.GREEN}✓{Style.RESET_ALL} Vision (Overshoot): {Fore.GREEN}Configured{Style.RESET_ALL}")
        status["vision"] = True
    else:
        print(f"  {Fore.YELLOW}○{Style.RESET_ALL} Vision (Overshoot): {Fore.YELLOW}Missing (will use color fallback){Style.RESET_ALL}")
        status["vision"] = False
    
    # Transcription (LiveKit)
    if _check_key(LIVEKIT_API_KEY, "LIVEKIT_API_KEY") and _check_key(LIVEKIT_API_SECRET, "LIVEKIT_API_SECRET"):
        print(f"  {Fore.GREEN}✓{Style.RESET_ALL} Transcription (LiveKit): {Fore.GREEN}Configured{Style.RESET_ALL}")
        status["transcription_livekit"] = True
    else:
        print(f"  {Fore.YELLOW}○{Style.RESET_ALL} Transcription (LiveKit): {Fore.YELLOW}Missing (will use browser speech){Style.RESET_ALL}")
        status["transcription_livekit"] = False
    
    # Voice (ElevenLabs)
    if _check_key(ELEVENLABS_API_KEY, "ELEVENLABS_API_KEY"):
        print(f"  {Fore.GREEN}✓{Style.RESET_ALL} Voice (ElevenLabs): {Fore.GREEN}Configured{Style.RESET_ALL}")
        status["voice"] = True
    else:
        print(f"  {Fore.YELLOW}○{Style.RESET_ALL} Voice (ElevenLabs): {Fore.YELLOW}Missing (will use pyttsx3 fallback){Style.RESET_ALL}")
        status["voice"] = False
    
    # LLM (Perplexity) - Required for transcription fallback and fact extraction
    if _check_key(PERPLEXITY_API_KEY, "PERPLEXITY_API_KEY"):
        print(f"  {Fore.GREEN}✓{Style.RESET_ALL} LLM (Perplexity): {Fore.GREEN}Configured{Style.RESET_ALL}")
        # Show partial key for debugging
        masked = PERPLEXITY_API_KEY[:8] + "..." + PERPLEXITY_API_KEY[-4:] if len(PERPLEXITY_API_KEY) > 12 else "***"
        print(f"    {Fore.CYAN}Key: {masked}{Style.RESET_ALL}")
        status["llm"] = True
    else:
        print(f"  {Fore.RED}✗{Style.RESET_ALL} LLM (Perplexity): {Fore.RED}Missing (regex fallback only){Style.RESET_ALL}")
        if PERPLEXITY_API_KEY:
            print(f"    {Fore.YELLOW}(Key value exists but appears invalid){Style.RESET_ALL}")
        status["llm"] = False
    
    # Demo mode indicator
    print()
    if DEMO_MODE:
        print(f"  {Fore.MAGENTA}◆{Style.RESET_ALL} Demo Mode: {Fore.MAGENTA}ENABLED{Style.RESET_ALL}")
    else:
        print(f"  {Fore.WHITE}◇{Style.RESET_ALL} Demo Mode: {Fore.WHITE}Disabled{Style.RESET_ALL}")
    
    print()
    print(f"{Fore.CYAN}{'='*32}{Style.RESET_ALL}\n")
    
    return status


if __name__ == "__main__":
    # Test configuration when run directly
    validate_config()