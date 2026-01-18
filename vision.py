"""
vision.py - Person detection logic
Decoupled from hardware: Processes image bytes provided by the server.
Uses face recognition (primary), Overshoot API for detection, falls back to OpenCV color detection.
"""

import base64
import json
import time
from collections import deque
from typing import Optional, Tuple, List, Dict
import requests
import numpy as np
import cv2

from colorama import Fore, Style

from config import (
    OVERSHOOT_API_KEY,
    OVERSHOOT_ENDPOINT,
    DETECTION_CONFIDENCE_FRAMES,
    COLOR_TO_PERSON,
    FACE_RECOGNITION_ENABLED,
    FACE_MODEL_NAME,
    FACE_DETECTOR_BACKEND,
    FACE_SIMILARITY_THRESHOLD
)
from memory import get_person_by_marker, get_face_embeddings

# =============================================================================
# MODULE-LEVEL STATE
# =============================================================================

# Detection buffer for hysteresis - stores recent person detections
_detection_buffer = deque(maxlen=DETECTION_CONFIDENCE_FRAMES)

# Face recognition state
_face_embeddings_cache: List[Dict] = []
_face_cache_timestamp: float = 0
_FACE_CACHE_REFRESH_INTERVAL = 30  # Refresh embeddings from DB every 30 seconds
_deepface_available = False
_face_model_loaded = False

# Frame skipping for real-time detection (face detection is expensive)
_frame_counter = 0
_FACE_DETECTION_INTERVAL = 5  # Only run face detection every N frames
_last_face_result: Optional[str] = None  # Cache last face detection result

# Try to import DeepFace
try:
    from deepface import DeepFace
    _deepface_available = True
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} DeepFace loaded successfully")
except ImportError as e:
    print(f"{Fore.YELLOW}⚠ DeepFace not available: {e}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}  Face recognition disabled, using color detection only{Style.RESET_ALL}")


def preload_face_model():
    """Pre-load the face recognition model to avoid first-call delay."""
    global _face_model_loaded
    if not _deepface_available or _face_model_loaded:
        return

    try:
        print(f"{Fore.CYAN}⏳ Pre-loading face recognition model...{Style.RESET_ALL}")
        # Create a small dummy image to trigger model loading
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy[30:70, 30:70] = 128  # Add some content
        try:
            DeepFace.represent(
                img_path=dummy,
                model_name=FACE_MODEL_NAME,
                detector_backend="opencv",
                enforce_detection=False
            )
        except:
            pass  # Expected to fail, but model is now loaded
        _face_model_loaded = True
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Face recognition model pre-loaded")
    except Exception as e:
        print(f"{Fore.YELLOW}⚠ Could not pre-load face model: {e}{Style.RESET_ALL}")


# =============================================================================
# FACE RECOGNITION
# =============================================================================

def _refresh_face_embeddings_cache():
    """Refresh the face embeddings cache from the database."""
    global _face_embeddings_cache, _face_cache_timestamp

    current_time = time.time()
    if current_time - _face_cache_timestamp < _FACE_CACHE_REFRESH_INTERVAL:
        return  # Cache is still fresh

    try:
        raw_embeddings = get_face_embeddings()
        _face_embeddings_cache = []

        for emb_record in raw_embeddings:
            try:
                # Deserialize the embedding from JSON bytes
                embedding_list = json.loads(emb_record['embedding'])
                _face_embeddings_cache.append({
                    'person_id': emb_record['person_id'],
                    'embedding': np.array(embedding_list, dtype=np.float32),
                    'model_name': emb_record['model_name']
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"{Fore.YELLOW}⚠ Invalid embedding for {emb_record.get('person_id')}: {e}{Style.RESET_ALL}")

        _face_cache_timestamp = current_time
        if _face_embeddings_cache:
            print(f"{Fore.CYAN}↻ Refreshed face embeddings cache: {len(_face_embeddings_cache)} faces{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error refreshing face embeddings cache: {e}{Style.RESET_ALL}")


def extract_face_embedding(frame: np.ndarray, debug: bool = False, thorough: bool = False) -> Optional[np.ndarray]:
    """
    Extract face embedding from a frame using DeepFace.

    Args:
        frame: OpenCV BGR image
        debug: If True, print detailed debug info
        thorough: If True, try multiple backends (slower but more reliable, use for enrollment)
                  If False, use only fast opencv backend (for real-time detection)

    Returns:
        128-dimensional embedding vector or None if no face detected
    """
    if not _deepface_available or not FACE_RECOGNITION_ENABLED:
        if debug:
            print(f"{Fore.YELLOW}⚠ DeepFace not available or disabled{Style.RESET_ALL}")
        return None

    if debug:
        print(f"{Fore.CYAN}📷 Image shape: {frame.shape}, dtype: {frame.dtype}{Style.RESET_ALL}")

    # For real-time: use only fast opencv backend
    # For enrollment: try multiple backends for better accuracy
    if thorough:
        backends_to_try = [FACE_DETECTOR_BACKEND, "retinaface", "mtcnn", "opencv", "ssd"]
        backends_to_try = list(dict.fromkeys(backends_to_try))  # Remove duplicates
    else:
        backends_to_try = ["opencv"]  # Fast only for real-time

    for backend in backends_to_try:
        try:
            if debug:
                print(f"{Fore.CYAN}🔍 Trying detector backend: {backend}{Style.RESET_ALL}")

            results = DeepFace.represent(
                img_path=frame,
                model_name=FACE_MODEL_NAME,
                detector_backend=backend,
                enforce_detection=True
            )

            if results and len(results) > 0:
                result = results[0]
                embedding = result.get('embedding')

                if debug:
                    print(f"{Fore.CYAN}📊 DeepFace results: {len(results)} face(s) found with {backend}{Style.RESET_ALL}")
                    face_confidence = result.get('face_confidence', 'N/A')
                    facial_area = result.get('facial_area', {})
                    print(f"{Fore.CYAN}   Face confidence: {face_confidence}{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}   Facial area: {facial_area}{Style.RESET_ALL}")

                if embedding:
                    if debug:
                        print(f"{Fore.GREEN}✓ Embedding extracted: {len(embedding)} dimensions (backend: {backend}){Style.RESET_ALL}")
                    return np.array(embedding, dtype=np.float32)

        except ValueError as e:
            error_msg = str(e).lower()
            if "face could not be detected" in error_msg or "no face" in error_msg:
                if debug:
                    print(f"{Fore.YELLOW}   No face detected with {backend}{Style.RESET_ALL}")
                continue
            else:
                if debug:
                    print(f"{Fore.RED}Face embedding ValueError ({backend}): {e}{Style.RESET_ALL}")
                continue

        except Exception as e:
            if debug:
                print(f"{Fore.RED}   Error with {backend}: {type(e).__name__}: {e}{Style.RESET_ALL}")
            continue

    if debug:
        print(f"{Fore.YELLOW}⚠ No face detected with any backend{Style.RESET_ALL}")
    return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def match_face_embedding(embedding: np.ndarray) -> Optional[str]:
    """
    Match an embedding against stored face embeddings.

    Args:
        embedding: 128-dimensional face embedding

    Returns:
        person_id of best match if similarity exceeds threshold, None otherwise
    """
    if embedding is None or len(_face_embeddings_cache) == 0:
        return None

    best_match = None
    best_similarity = -1

    for stored in _face_embeddings_cache:
        similarity = _cosine_similarity(embedding, stored['embedding'])

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = stored['person_id']

    if best_similarity >= FACE_SIMILARITY_THRESHOLD:
        return best_match

    return None


def detect_person_face(frame: np.ndarray) -> Optional[str]:
    """
    Detect and identify a person by their face.
    Uses frame skipping to reduce CPU load during real-time detection.

    Args:
        frame: OpenCV BGR image

    Returns:
        person_id if face matched, None otherwise
    """
    global _frame_counter, _last_face_result

    if not _deepface_available or not FACE_RECOGNITION_ENABLED:
        return None

    # Refresh embeddings cache if needed
    _refresh_face_embeddings_cache()

    # No enrolled faces to match against
    if len(_face_embeddings_cache) == 0:
        return None

    # Frame skipping: only run expensive face detection every N frames
    _frame_counter += 1
    if _frame_counter < _FACE_DETECTION_INTERVAL:
        return _last_face_result  # Return cached result

    _frame_counter = 0  # Reset counter

    # Extract embedding from current frame (fast mode for real-time)
    embedding = extract_face_embedding(frame, debug=False, thorough=False)
    if embedding is None:
        _last_face_result = None
        return None

    # Match against stored embeddings
    person_id = match_face_embedding(embedding)

    if person_id and person_id != _last_face_result:
        print(f"{Fore.GREEN}👤 Face recognized: {person_id}{Style.RESET_ALL}")

    _last_face_result = person_id
    return person_id


def invalidate_face_cache():
    """Force refresh of face embeddings cache on next detection."""
    global _face_cache_timestamp, _last_face_result, _frame_counter
    _face_cache_timestamp = 0
    _last_face_result = None
    _frame_counter = 0


# =============================================================================
# IMAGE PROCESSING (SERVER-SIDE)
# =============================================================================

def process_image_bytes(image_bytes: bytes) -> Optional[str]:
    """
    Decode raw JPEG bytes into an OpenCV frame and run detection.
    Returns the person_id if a stable detection occurs, else None.
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
            
        return detect_person(frame)
    except Exception as e:
        print(f"{Fore.RED}Error processing frame bytes: {e}{Style.RESET_ALL}")
        return None


# =============================================================================
# IMAGE ENCODING (HELPER)
# =============================================================================

def frame_to_base64(frame: np.ndarray) -> str:
    """
    Encode frame as JPEG and return base64 string.
    Required for sending the image to Overshoot API.
    """
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


# =============================================================================
# COLOR DETECTION (FALLBACK)
# =============================================================================

def detect_dominant_color(frame: np.ndarray) -> Optional[str]:
    """
    Detect dominant clothing color in center region of frame.
    This is the FALLBACK when Overshoot API fails.
    """
    try:
        # Focus on center region (where person's torso would be)
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Extract center region (roughly torso area)
        roi_width, roi_height = width // 3, height // 3
        x1 = center_x - roi_width // 2
        y1 = center_y - roi_height // 2
        x2 = center_x + roi_width // 2
        y2 = center_y + roi_height // 2
        
        roi = frame[y1:y2, x1:x2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Color ranges (H, S, V)
        color_ranges = {
            'red': [
                ((0, 100, 100), (10, 255, 255)),
                ((160, 100, 100), (180, 255, 255))
            ],
            'blue': [
                ((100, 100, 100), (130, 255, 255))
            ],
            'green': [
                ((40, 100, 100), (80, 255, 255))
            ],
            'black': [
                ((0, 0, 0), (180, 255, 50))  # Low value (V) for black
            ]
        }
        
        # Count pixels for each color
        color_counts = {}
        total_pixels = roi.shape[0] * roi.shape[1]
        
        for color_name, ranges in color_ranges.items():
            count = 0
            for (lower, upper) in ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                count += cv2.countNonZero(mask)
            color_counts[color_name] = count
        
        # Find dominant color (must be at least 10% of pixels)
        min_threshold = total_pixels * 0.10
        dominant = max(color_counts, key=color_counts.get)
        
        if color_counts[dominant] > min_threshold:
            return dominant
        
        return None
        
    except Exception as e:
        print(f"{Fore.RED}Color detection error: {e}{Style.RESET_ALL}")
        return None


# =============================================================================
# OVERSHOOT API DETECTION
# =============================================================================

def detect_person_overshoot(frame: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Overshoot API for person detection.
    Returns: (person_label, clothing_color) or (None, None) on ANY error
    """
    if not OVERSHOOT_API_KEY:
        return (None, None)
    
    try:
        # Encode frame
        image_b64 = frame_to_base64(frame)
        
        # Make API request
        headers = {
            "Authorization": f"Bearer {OVERSHOOT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            OVERSHOOT_ENDPOINT,
            headers=headers,
            json={"image": image_b64},
            timeout=5
        )
        
        if response.status_code != 200:
            print(f"{Fore.RED}[Overshoot] Error {response.status_code}{Style.RESET_ALL}")
            return (None, None)
        
        data = response.json()
        
        if "detections" not in data:
            return (None, None)
        
        for detection in data["detections"]:
            if detection.get("label") == "person":
                confidence = detection.get("confidence", 0)
                if confidence > 0.5:
                    attributes = detection.get("attributes", {})
                    clothing_color = attributes.get("clothing_color")
                    return ("person", clothing_color)
        
        return (None, None)
        
    except Exception as e:
        print(f"{Fore.RED}[Overshoot] Error: {e}{Style.RESET_ALL}")
        return (None, None)


# =============================================================================
# MAIN DETECTION (WITH HYSTERESIS)
# =============================================================================

def detect_person(frame: np.ndarray) -> Optional[str]:
    """
    Detect person in frame with hysteresis for stability.
    Detection priority:
    1. Face recognition (primary)
    2. Overshoot API color detection (secondary)
    3. OpenCV color detection (fallback)

    Returns: person_id if stable detection, None otherwise
    """
    global _detection_buffer

    person_id = None

    # 1. Try face recognition first (primary method)
    if FACE_RECOGNITION_ENABLED and _deepface_available:
        person_id = detect_person_face(frame)

    # 2. If face not recognized, try color-based detection
    if person_id is None:
        clothing_color = None

        # Try Overshoot API
        person_label, api_color = detect_person_overshoot(frame)

        if person_label and api_color:
            clothing_color = api_color.lower()
        else:
            # Fallback to local color detection
            clothing_color = detect_dominant_color(frame)

        # Map color to person_id
        if clothing_color and clothing_color in COLOR_TO_PERSON:
            person_id = COLOR_TO_PERSON[clothing_color]
        elif clothing_color:
            # Check database for custom markers
            person_info = get_person_by_marker(clothing_color)
            if person_info:
                person_id = person_info['person_id']

    # Add to detection buffer (hysteresis)
    _detection_buffer.append(person_id)

    # Only return person_id if ALL recent detections are the SAME (consistent)
    if len(_detection_buffer) >= DETECTION_CONFIDENCE_FRAMES:
        buffer_set = set(_detection_buffer)
        if len(buffer_set) == 1 and None not in buffer_set:
            return _detection_buffer[0]

    return None

def clear_detection_buffer():
    global _detection_buffer
    _detection_buffer.clear()


# =============================================================================
# OVERWATCH / SENTINEL MODE - HAZARD DETECTION
# =============================================================================

def check_hazard(image_bytes: bytes, profile: dict) -> Optional[dict]:
    """
    Analyze FPV image for safety hazards using Overshoot API.
    
    Args:
        image_bytes: Raw JPEG bytes from WebSocket
        profile: Safety profile dict with keys:
            - allergy: str (e.g., "peanuts")
            - med_name: str (e.g., "Amoxicillin")
            - meds_taken: bool
    
    Returns:
        dict: {'type': 'hazard'|'log'|'none', 'text': str, 'icon': str} or None on error
    """
    if not OVERSHOOT_API_KEY:
        print(f"{Fore.YELLOW}⚠ Overshoot API key not configured for hazard detection{Style.RESET_ALL}")
        return {"type": "none"}
    
    try:
        # Encode image to base64
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Build dynamic context-aware prompt
        allergy = profile.get('allergy', 'peanuts')
        med_name = profile.get('med_name', 'Amoxicillin')
        meds_taken = profile.get('meds_taken', False)
        status_str = "ALREADY TAKEN TODAY" if meds_taken else "NOT YET TAKEN TODAY"
        
        prompt = f"""You are a safety monitoring system analyzing a first-person view image.

PATIENT PROFILE:
- ALLERGIC TO: {allergy}
- Medication: {med_name} (Status: {status_str})

TASK: Look at this image carefully. Identify any objects, drinks, food, or items visible.

CHECK FOR HAZARDS:
1. Is there a Monster Energy drink can visible? (green claw logo, black can)
2. Is there any item related to "{allergy}"?
3. Is medication visible while status is ALREADY TAKEN TODAY?

IMPORTANT: A Monster Energy drink/can IS a hazard for this patient.

RESPOND WITH ONLY ONE JSON:

If Monster Energy drink OR "{allergy}" related item found:
{{"type": "hazard", "text": "Warning: {allergy} detected!", "icon": "⚠️"}}

If medication found AND status is ALREADY TAKEN TODAY:
{{"type": "hazard", "text": "Stop: Dose already taken today", "icon": "🚫"}}

If medication found AND status is NOT YET TAKEN TODAY:
{{"type": "log", "text": "Medication logged successfully"}}

If nothing hazardous:
{{"type": "none"}}

Output ONLY the JSON."""

        # Call Overshoot API
        headers = {
            "Authorization": f"Bearer {OVERSHOOT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "prompt": prompt,
            "images": [img_base64],
            "max_tokens": 100
        }
        
        print(f"{Fore.CYAN}🔍 [Hazard] Analyzing frame for: {allergy} | Med: {med_name} | Taken: {meds_taken}{Style.RESET_ALL}")
        
        response = requests.post(
            OVERSHOOT_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=8
        )
        
        if response.status_code != 200:
            print(f"{Fore.RED}[Hazard] Overshoot API error: {response.status_code} - {response.text[:500]}{Style.RESET_ALL}")
            return {"type": "none"}
        
        result = response.json()
        print(f"{Fore.YELLOW}📡 [Hazard] Raw API Response: {result}{Style.RESET_ALL}")
        
        # Try different response formats
        response_text = ""
        if "choices" in result:
            response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        elif "result" in result:
            response_text = result.get("result", "")
        elif "output" in result:
            response_text = result.get("output", "")
        elif "text" in result:
            response_text = result.get("text", "")
        else:
            response_text = str(result)
        
        print(f"{Fore.YELLOW}📡 [Hazard] Parsed Response: {response_text}{Style.RESET_ALL}")
        
        # Parse JSON response
        try:
            # Try direct JSON parse
            hazard_result = json.loads(response_text.strip())
            
            # Log detection
            if hazard_result.get("type") == "hazard":
                print(f"{Fore.RED}🚨 HAZARD DETECTED: {hazard_result.get('text')}{Style.RESET_ALL}")
            elif hazard_result.get("type") == "log":
                print(f"{Fore.GREEN}✅ MEDICATION LOGGED: {hazard_result.get('text')}{Style.RESET_ALL}")
            
            return hazard_result
            
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                try:
                    hazard_result = json.loads(json_match.group())
                    return hazard_result
                except:
                    pass
            
            print(f"{Fore.YELLOW}⚠ Could not parse hazard response: {response_text[:100]}{Style.RESET_ALL}")
            return {"type": "none"}
            
    except requests.Timeout:
        print(f"{Fore.RED}[Hazard] API timeout{Style.RESET_ALL}")
        return {"type": "none"}
    except Exception as e:
        print(f"{Fore.RED}[Hazard] Error: {e}{Style.RESET_ALL}")
        return {"type": "none"}