"""
server.py - FastAPI Backend for Memento Protocol
Full-featured REST API + WebSocket for caregiver dashboard and HUD.
"""

import sys
import json
import asyncio
import logging
import re
import random
import base64
import requests
from typing import Dict, Optional, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from colorama import init, Fore, Style

# Core modules
from memory import (
    init_db,
    save_memory,
    get_person_by_id,
    get_all_people,
    get_all_memories,
    get_recent_memories,
    get_memory_stats,
    register_person,
    delete_person,
    add_caregiver_note,
    save_last_conversation,
    get_last_conversation,
    clear_last_conversation,
    get_recent_topics,
    get_important_personal_info,
    save_face_embedding,
    get_face_embeddings,
    delete_face_embeddings,
    update_person_face_image,
    has_face_enrolled
)
import vision
import audio
from config import (
    PERSON_COOLDOWN_SECONDS,
    validate_config,
    FACE_RECOGNITION_ENABLED,
    FACE_MODEL_NAME,
    FACE_IMAGES_DIR
)

# Initialize colorama
init(autoreset=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memento-server")

# =============================================================================
# SAFETY STATE (OVERWATCH MODE)
# =============================================================================

# Global Safety State - Reset on server restart or at midnight
SAFETY_STATE = {
    "allergy": "Monster energy drinks",
    "med_name": "Amoxicillin",
    "meds_taken": False,
    "last_reset": datetime.now().date()
}

def check_midnight_reset():
    """Reset meds_taken at midnight."""
    global SAFETY_STATE
    today = datetime.now().date()
    if SAFETY_STATE["last_reset"] != today:
        SAFETY_STATE["meds_taken"] = False
        SAFETY_STATE["last_reset"] = today
        print(f"{Fore.CYAN}↻ Midnight reset: meds_taken = False{Style.RESET_ALL}")
        return True
    return False

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class TranscriptRequest(BaseModel):
    transcript: str
    person_id: Optional[str] = None

class AudioTranscribeRequest(BaseModel):
    audio: str  # base64 encoded audio
    person_id: Optional[str] = None
    sample_rate: int = 16000

class PersonCreate(BaseModel):
    name: str
    relationship: str
    marker: str
    bio: Optional[str] = ""
    face_image: Optional[str] = None  # Base64 encoded face image

class FaceEnrollRequest(BaseModel):
    image: str  # Base64 encoded face image

class CaregiverNote(BaseModel):
    person_id: str
    note: str
    topic: Optional[str] = "caregiver"

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="Memento Protocol",
    description="AI Memory Companion for Caregiving",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# LIFECYCLE
# =============================================================================

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*50)
    print("   MEMENTO PROTOCOL v2.0")
    print("   AI Memory Companion for Caregiving")
    print("="*50 + "\n")
    init_db()
    # validate_config()  # Show API key status - Temporarily disabled due to Unicode issues

    # Pre-load face recognition model to avoid first-call delay
    # Temporarily disabled due to Unicode output issues on Windows
    # if FACE_RECOGNITION_ENABLED:
    #     vision.preload_face_model()

    print("\n[OK] Server ready at http://localhost:8000\n")

# =============================================================================
# STATIC FILES & SPA
# =============================================================================

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_spa():
    """Serve the Single Page Application."""
    return FileResponse("static/index.html")

# =============================================================================
# REST API: PEOPLE
# =============================================================================

@app.get("/api/people")
async def api_get_people():
    """Get all registered people with memory counts."""
    import os
    try:
        people = get_all_people()
        # Convert face_image_path to web-accessible URL
        for person in people:
            if person.get('face_image_path'):
                # Convert absolute path to relative URL
                face_path = person['face_image_path']
                if os.path.exists(face_path):
                    # Extract the relative path from static/faces/...
                    if 'static' in face_path:
                        relative_path = face_path.split('static', 1)[1].replace('\\', '/')
                        person['face_image_path'] = '/static' + relative_path
                    else:
                        person['face_image_path'] = ''
                else:
                    person['face_image_path'] = ''
        return {"status": "ok", "data": people}
    except Exception as e:
        logger.error(f"Error fetching people: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/people")
async def api_create_person(person: PersonCreate):
    """Register a new person. Optionally enroll face if face_image provided."""
    import base64
    import os
    import json
    import cv2
    import numpy as np

    try:
        # Generate person_id from name (lowercase, underscores)
        person_id = re.sub(r'[^a-z0-9]+', '_', person.name.lower()).strip('_')

        # Check for duplicate
        existing = get_person_by_id(person_id)
        if existing:
            raise HTTPException(status_code=409, detail=f"Person '{person_id}' already exists")

        register_person(
            person_id=person_id,
            display_name=person.name,
            relationship=person.relationship,
            visual_marker=person.marker.lower(),
            bio=person.bio or ""
        )

        # Save bio as an event/memory if provided
        if person.bio and person.bio.strip():
            save_memory(
                person_id=person_id,
                fact=person.bio.strip(),
                topic='about',
                source='caregiver',
                confidence=1.0,
                check_duplicates=False
            )
            print(f"{Fore.GREEN}✓ Bio saved as memory for {person_id}{Style.RESET_ALL}")

        face_enrolled = False

        # If face image provided, enroll it
        if person.face_image and FACE_RECOGNITION_ENABLED:
            try:
                # Decode base64 image
                image_data = base64.b64decode(person.face_image)
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Extract face embedding (thorough mode for enrollment)
                    print(f"{Fore.CYAN}🔍 Attempting face extraction during person creation...{Style.RESET_ALL}")
                    embedding = vision.extract_face_embedding(frame, debug=True, thorough=True)
                    if embedding is not None:
                        # Save face image to disk
                        os.makedirs(FACE_IMAGES_DIR, exist_ok=True)
                        person_face_dir = os.path.join(FACE_IMAGES_DIR, person_id)
                        os.makedirs(person_face_dir, exist_ok=True)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        face_filename = f"face_{timestamp}.jpg"
                        face_path = os.path.join(person_face_dir, face_filename)
                        cv2.imwrite(face_path, frame)

                        # Serialize and save embedding
                        embedding_json = json.dumps(embedding.tolist()).encode('utf-8')
                        save_face_embedding(person_id, embedding_json, FACE_MODEL_NAME)
                        update_person_face_image(person_id, face_path)

                        # Invalidate cache
                        vision.invalidate_face_cache()

                        face_enrolled = True
                        print(f"{Fore.GREEN}✓ Face enrolled during person creation: {person_id}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.YELLOW}⚠ No face detected in provided image for {person_id}{Style.RESET_ALL}")
            except Exception as e:
                logger.warning(f"Failed to enroll face for {person_id}: {e}")
                # Don't fail person creation if face enrollment fails

        return {"status": "ok", "person_id": person_id, "face_enrolled": face_enrolled}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating person: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/people/{person_id}")
async def api_delete_person(person_id: str):
    """Delete a person and their memories."""
    try:
        deleted = delete_person(person_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found")
        return {"status": "ok", "deleted": person_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting person: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/people/{person_id}/memories")
async def api_get_person_memories(person_id: str, limit: int = 20):
    """Get all memories for a specific person."""
    try:
        person = get_person_by_id(person_id)
        if not person:
            raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found")

        memories = get_all_memories(person_id, limit=limit)
        return {"status": "ok", "person": person, "memories": memories}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# REST API: FACE ENROLLMENT
# =============================================================================

@app.post("/api/people/{person_id}/face")
async def api_enroll_face(person_id: str, request: FaceEnrollRequest):
    """Enroll a face image for a person. Extracts and stores face embedding."""
    import base64
    import os
    import json
    import cv2
    import numpy as np

    try:
        person = get_person_by_id(person_id)
        if not person:
            raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found")

        if not FACE_RECOGNITION_ENABLED:
            raise HTTPException(status_code=503, detail="Face recognition is disabled")

        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Could not decode image")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

        # Extract face embedding (thorough mode for enrollment - tries multiple backends)
        print(f"{Fore.CYAN}🔍 Attempting face extraction for {person_id}...{Style.RESET_ALL}")
        embedding = vision.extract_face_embedding(frame, debug=True, thorough=True)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in image. Please provide a clear, front-facing photo with good lighting.")

        # Save face image to disk
        os.makedirs(FACE_IMAGES_DIR, exist_ok=True)
        person_face_dir = os.path.join(FACE_IMAGES_DIR, person_id)
        os.makedirs(person_face_dir, exist_ok=True)

        # Save with timestamp to allow multiple images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_filename = f"face_{timestamp}.jpg"
        face_path = os.path.join(person_face_dir, face_filename)
        cv2.imwrite(face_path, frame)

        # Serialize embedding to JSON bytes
        embedding_json = json.dumps(embedding.tolist()).encode('utf-8')

        # Save embedding to database
        embedding_id = save_face_embedding(person_id, embedding_json, FACE_MODEL_NAME)
        if embedding_id is None:
            raise HTTPException(status_code=500, detail="Failed to save face embedding")

        # Update person's face image path
        update_person_face_image(person_id, face_path)

        # Invalidate cache so new face is recognized immediately
        vision.invalidate_face_cache()

        print(f"{Fore.GREEN}✓ Face enrolled for {person_id}{Style.RESET_ALL}")

        return {
            "status": "ok",
            "message": f"Face enrolled for {person['display_name']}",
            "embedding_id": embedding_id,
            "face_image_path": face_path
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enrolling face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/people/{person_id}/face")
async def api_delete_face(person_id: str):
    """Delete all face data for a person."""
    import os
    import shutil

    try:
        person = get_person_by_id(person_id)
        if not person:
            raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found")

        # Delete embeddings from database
        deleted_count = delete_face_embeddings(person_id)

        # Delete face images from disk
        person_face_dir = os.path.join(FACE_IMAGES_DIR, person_id)
        if os.path.exists(person_face_dir):
            shutil.rmtree(person_face_dir)

        # Clear face image path in database
        update_person_face_image(person_id, "")

        # Invalidate cache
        vision.invalidate_face_cache()

        return {
            "status": "ok",
            "message": f"Face data deleted for {person['display_name']}",
            "embeddings_deleted": deleted_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/people/{person_id}/face")
async def api_get_face_status(person_id: str):
    """Check if a person has a face enrolled."""
    try:
        person = get_person_by_id(person_id)
        if not person:
            raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found")

        enrolled = has_face_enrolled(person_id)
        embeddings = get_face_embeddings(person_id)

        return {
            "status": "ok",
            "enrolled": enrolled,
            "embedding_count": len(embeddings),
            "face_image_path": person.get('face_image_path', ''),
            "face_recognition_enabled": FACE_RECOGNITION_ENABLED
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking face status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# REST API: MEMORIES
# =============================================================================

@app.get("/api/memories/recent")
async def api_get_recent_memories(limit: int = 20):
    """Get recent memories across all people (dashboard feed)."""
    try:
        memories = get_recent_memories(limit=limit)
        return {"status": "ok", "data": memories}
    except Exception as e:
        logger.error(f"Error fetching recent memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memories/note")
async def api_add_caregiver_note(note: CaregiverNote):
    """Add a caregiver-authored memory note."""
    try:
        person = get_person_by_id(note.person_id)
        if not person:
            raise HTTPException(status_code=404, detail=f"Person '{note.person_id}' not found")
        
        memory_id = add_caregiver_note(
            person_id=note.person_id,
            note=note.note,
            topic=note.topic or "caregiver"
        )
        
        if memory_id is None:
            return {"status": "duplicate", "message": "Similar note already exists"}
        
        return {"status": "ok", "memory_id": memory_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# REST API: STATS
# =============================================================================

@app.get("/api/stats")
async def api_get_stats():
    """Get aggregate statistics for dashboard."""
    try:
        stats = get_memory_stats()
        return {"status": "ok", "data": stats}
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# REST API: AUDIO TRANSCRIPT
# =============================================================================

@app.post("/listen")
async def listen_endpoint(request: TranscriptRequest):
    """Process audio transcript and save the full message to memory."""
    if not request.transcript or len(request.transcript.strip()) < 3:
        return {"status": "ignored", "reason": "empty_transcript", "facts_extracted": 0, "facts_saved": 0}
    
    transcript = request.transcript.strip()
    print(f"{Fore.WHITE}🎤 Heard: \"{transcript}\"{Style.RESET_ALL}")
    
    person_name = "Friend"
    
    if request.person_id:
        person_info = get_person_by_id(request.person_id)
        if person_info:
            person_name = person_info['display_name']
            print(f"{Fore.CYAN}   └─ Speaking with: {person_name}{Style.RESET_ALL}")
    
    saved_count = 0
    
    # Save the full transcript directly to memory (no compression)
    if request.person_id:
        result = save_memory(
            person_id=request.person_id,
            fact=transcript,  # Save the full message
            topic='conversation',
            source='conversation',
            confidence=1.0
        )
        if result:
            saved_count = 1
            print(f"{Fore.GREEN}   └─ 💾 Saved full transcript{Style.RESET_ALL}")
        
        # Save the last conversation for reminders
        save_last_conversation(request.person_id, transcript, 'conversation')
    
    return {
        "status": "ok",
        "transcript": transcript,
        "facts_extracted": 1 if saved_count else 0,
        "facts_saved": saved_count,
        "facts": [transcript] if saved_count else []
    }


@app.post("/transcribe")
async def transcribe_endpoint(request: AudioTranscribeRequest):
    """
    Transcribe audio using LiveKit STT (server-side alternative to browser speech).
    Accepts base64 encoded audio data.
    """
    try:
        import base64
        
        if not request.audio:
            return {"status": "error", "reason": "no_audio"}
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio)
        
        # Transcribe using LiveKit
        transcript = audio.transcribe_audio_livekit(audio_bytes, request.sample_rate)
        
        if not transcript:
            return {"status": "ignored", "reason": "transcription_failed"}
        
        # Also extract facts if person is known
        facts = []
        saved_count = 0
        
        if request.person_id:
            person_info = get_person_by_id(request.person_id)
            if person_info:
                facts = audio.extract_facts(transcript, person_info['display_name'])
                
                for fact in facts:
                    result = save_memory(
                        person_id=request.person_id,
                        fact=fact['fact'],
                        topic=fact.get('topic'),
                        source='conversation',
                        confidence=fact.get('confidence', 0.85)
                    )
                    if result:
                        saved_count += 1
        
        return {
            "status": "ok",
            "transcript": transcript,
            "facts_extracted": len(facts),
            "facts_saved": saved_count
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"status": "error", "reason": str(e)}


@app.get("/livekit/token")
async def get_livekit_token():
    """
    Get LiveKit connection info for frontend real-time transcription.
    Returns URL and token for WebRTC connection.
    """
    connection_info = audio.get_livekit_connection_info()
    
    if not connection_info:
        raise HTTPException(status_code=503, detail="LiveKit not configured")
    
    return {
        "status": "ok",
        "data": connection_info
    }


# =============================================================================
# WEBSOCKET: VIDEO STREAM
# =============================================================================

def build_whisper(person_info: dict, last_conversation: dict = None, person_id: str = None) -> str:
    """Use LLM to generate a natural, human whisper about the person."""
    name = person_info.get('display_name', 'someone')
    relationship = person_info.get('relationship', '')
    
    # Get most recent memory
    most_recent_memory = None
    if person_id:
        recent_topics = get_recent_topics(person_id, limit=1)
        if recent_topics:
            most_recent_memory = recent_topics[0]['fact']
    
    # Use LLM to generate natural whisper
    whisper = audio.generate_whisper(name, relationship, most_recent_memory)
    return whisper


# How long person must be absent before greeting resets (60 seconds)
ABSENCE_RESET_SECONDS = 60

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming and person recognition.
    Supports two modes:
    - standard: Face/memory recognition (default)
    - overwatch: Safety hazard detection (Sentinel Mode)
    """
    await websocket.accept()
    
    # Connection state
    last_announced: Dict[str, datetime] = {}
    person_left_at: Dict[str, datetime] = {}
    current_person: Optional[str] = None
    last_heartbeat = datetime.now()
    mode = "standard"  # "standard" or "overwatch"
    frame_count = 0
    
    # Check for midnight reset
    check_midnight_reset()
    
    print(f"{Fore.GREEN}✓ Client connected to /stream{Style.RESET_ALL}")
    
    try:
        while True:
            try:
                # Non-blocking receive with timeout for heartbeat
                data = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=3.0
                )
                
                now = datetime.now()
                
                # Handle text messages (mode switch, etc.)
                if "text" in data:
                    try:
                        msg = json.loads(data["text"])
                        if msg.get("type") == "mode_switch":
                            new_mode = msg.get("mode", "standard")
                            if new_mode in ["standard", "overwatch"]:
                                mode = new_mode
                                frame_count = 0
                                print(f"{Fore.YELLOW}🔄 Mode switched to: {mode.upper()}{Style.RESET_ALL}")
                                
                                # Send confirmation (convert date to string for JSON)
                                safe_state = None
                                if mode == "overwatch":
                                    safe_state = {
                                        "allergy": SAFETY_STATE["allergy"],
                                        "med_name": SAFETY_STATE["med_name"],
                                        "meds_taken": SAFETY_STATE["meds_taken"],
                                        "last_reset": SAFETY_STATE["last_reset"].isoformat()
                                    }
                                await websocket.send_text(json.dumps({
                                    "type": "mode_changed",
                                    "mode": mode,
                                    "safety_state": safe_state
                                }))
                                
                                # Reset state when switching modes
                                if mode == "overwatch":
                                    current_person = None
                                continue
                    except json.JSONDecodeError:
                        pass
                
                # Handle binary data (video frames)
                if "bytes" in data:
                    frame_bytes = data["bytes"]
                    frame_count += 1
                    
                    # Check for midnight reset
                    check_midnight_reset()
                    
                    # ========================================
                    # OVERWATCH MODE (Sentinel/Safety)
                    # ========================================
                    if mode == "overwatch":
                        # Run hazard check every ~2 frames (~1 second at 2fps streaming rate)
                        if frame_count % 2 == 0:
                            print(f"{Fore.CYAN}🔎 Running hazard check (frame {frame_count})...{Style.RESET_ALL}")
                            sentinel_result = vision.check_hazard(frame_bytes, SAFETY_STATE)
                            
                            if sentinel_result and sentinel_result.get("type") != "none":
                                if sentinel_result["type"] == "hazard":
                                    # THREAT DETECTED
                                    await websocket.send_text(json.dumps({
                                        "type": "hazard_alert",
                                        "text": sentinel_result["text"],
                                        "icon": sentinel_result.get("icon", "⚠️")
                                    }))
                                    print(f"{Fore.RED}⚠️ HAZARD: {sentinel_result['text']}{Style.RESET_ALL}")
                                    
                                elif sentinel_result["type"] == "log":
                                    # MEDICATION LOGGED
                                    SAFETY_STATE["meds_taken"] = True
                                    print(f"{Fore.GREEN}✅ State Updated: meds_taken = True{Style.RESET_ALL}")
                                    
                                    await websocket.send_text(json.dumps({
                                        "type": "hazard_log",
                                        "text": "Medication successfully logged",
                                        "icon": "✅"
                                    }))
                        continue
                    
                    # ========================================
                    # STANDARD MODE (Face/Memory Recognition)
                    # ========================================
                    detected_person_id = vision.process_image_bytes(frame_bytes)
                    
                    if detected_person_id:
                        # Person detected on screen
                        was_absent = detected_person_id != current_person
                        
                        if was_absent:
                            # Check how long they were gone
                            absence_duration = 0
                            if detected_person_id in person_left_at:
                                absence_duration = (now - person_left_at[detected_person_id]).total_seconds()
                                del person_left_at[detected_person_id]
                            
                            # Only reset context if they were gone for 60+ seconds
                            if absence_duration >= ABSENCE_RESET_SECONDS or detected_person_id not in last_announced:
                                audio.reset_context()
                                print(f"{Fore.CYAN}↻ Context reset (absent for {absence_duration:.0f}s){Style.RESET_ALL}")
                            else:
                                print(f"{Fore.GREEN}👋 Person returned after {absence_duration:.0f}s - continuing conversation{Style.RESET_ALL}")
                            
                            current_person = detected_person_id
                        
                        # Check if we should send initial greeting
                        should_announce = False
                        
                        if detected_person_id not in last_announced:
                            should_announce = True
                        elif detected_person_id in person_left_at:
                            pass
                        else:
                            if was_absent:
                                elapsed_since_announce = (now - last_announced[detected_person_id]).total_seconds()
                                if elapsed_since_announce >= ABSENCE_RESET_SECONDS:
                                    should_announce = True
                        
                        # Send initial greeting when person first appears or returns after long absence
                        if should_announce:
                            person_info = get_person_by_id(detected_person_id)
                            
                            if person_info:
                                last_conversation = get_last_conversation(detected_person_id)
                                whisper_text = build_whisper(person_info, last_conversation=last_conversation, person_id=detected_person_id)
                                
                                response = {
                                    "type": "recognition",
                                    "person_id": detected_person_id,
                                    "name": person_info['display_name'],
                                    "relationship": person_info['relationship'],
                                    "marker": person_info.get('visual_marker', ''),
                                    "bio": person_info.get('bio', ''),
                                    "whisper": whisper_text,
                                    "last_conversation": last_conversation
                                }
                                
                                await websocket.send_text(json.dumps(response))
                                last_announced[detected_person_id] = now
                                print(f"{Fore.CYAN}🗣️  Announced: {person_info['display_name']}{Style.RESET_ALL}")
                                
                                if last_conversation:
                                    print(f"{Fore.GREEN}   └─ 💭 Reminder: {last_conversation['summary'][:60]}...{Style.RESET_ALL}")
                        
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "person_visible",
                                "person_id": detected_person_id
                            }))
                    
                    else:
                        # No person detected
                        if current_person:
                            person_left_at[current_person] = now
                            print(f"{Fore.YELLOW}👀 Person stepped away (will reset after {ABSENCE_RESET_SECONDS}s){Style.RESET_ALL}")
                            
                            await websocket.send_text(json.dumps({
                                "type": "person_away",
                                "person_id": current_person
                            }))
                            current_person = None
                
            except asyncio.TimeoutError:
                # Send heartbeat on timeout
                now = datetime.now()
                if (now - last_heartbeat).total_seconds() >= 5:
                    await websocket.send_text(json.dumps({"type": "heartbeat", "timestamp": now.isoformat()}))
                    last_heartbeat = now
                
                # Check if any person has been gone for 60+ seconds (standard mode only)
                if mode == "standard":
                    for pid, left_time in list(person_left_at.items()):
                        if (now - left_time).total_seconds() >= ABSENCE_RESET_SECONDS:
                            print(f"{Fore.YELLOW}👋 Person gone for {ABSENCE_RESET_SECONDS}s - conversation ended{Style.RESET_ALL}")
                            await websocket.send_text(json.dumps({
                                "type": "person_left",
                                "person_id": pid
                            }))
                            del person_left_at[pid]
                    
    except WebSocketDisconnect:
        print(f"{Fore.YELLOW}⚠ Client disconnected{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}WebSocket Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Connection error occurred"
            }))
        except:
            pass


# =============================================================================
# REST API: SAFETY STATE
# =============================================================================

@app.get("/api/safety-state")
async def get_safety_state():
    """Get current safety state for Overwatch mode."""
    check_midnight_reset()
    # Convert date to string for JSON
    safe_data = {
        "allergy": SAFETY_STATE["allergy"],
        "med_name": SAFETY_STATE["med_name"],
        "meds_taken": SAFETY_STATE["meds_taken"],
        "last_reset": SAFETY_STATE["last_reset"].isoformat()
    }
    return {"status": "ok", "data": safe_data}

@app.post("/api/safety-state/reset")
async def reset_safety_state():
    """Reset meds_taken to False (for demo purposes)."""
    global SAFETY_STATE
    SAFETY_STATE["meds_taken"] = False
    print(f"{Fore.CYAN}↻ Manual reset: meds_taken = False{Style.RESET_ALL}")
    safe_data = {
        "allergy": SAFETY_STATE["allergy"],
        "med_name": SAFETY_STATE["med_name"],
        "meds_taken": SAFETY_STATE["meds_taken"],
        "last_reset": SAFETY_STATE["last_reset"].isoformat()
    }
    return {"status": "ok", "data": safe_data}

@app.post("/api/safety-state/meds-taken")
async def set_meds_taken():
    """Mark medication as taken."""
    global SAFETY_STATE
    SAFETY_STATE["meds_taken"] = True
    print(f"{Fore.GREEN}✅ Medication logged: meds_taken = True{Style.RESET_ALL}")
    return {"status": "ok"}

@app.get("/api/overshoot-key")
async def get_overshoot_key():
    """Get Overshoot API key for frontend Sentinel mode."""
    from config import OVERSHOOT_API_KEY
    return {"key": OVERSHOOT_API_KEY}


# =============================================================================
# REST API: TEXT-TO-SPEECH (ELEVENLABS)
# =============================================================================

class TTSRequest(BaseModel):
    text: str

@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using ElevenLabs API.
    Returns audio data as base64 encoded MP3.
    """
    from config import ELEVENLABS_API_KEY, ELEVENLABS_ENDPOINT

    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="ElevenLabs not configured")

    try:
        print(f"{Fore.CYAN}🔊 Generating TTS for: \"{request.text[:50]}...\"{Style.RESET_ALL}")

        response = requests.post(
            ELEVENLABS_ENDPOINT,
            headers={
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            },
            json={
                "text": request.text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            },
            timeout=15
        )

        if response.status_code != 200:
            logger.error(f"ElevenLabs error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail="TTS generation failed")

        # Return audio as base64
        audio_base64 = base64.b64encode(response.content).decode('utf-8')
        print(f"{Fore.GREEN}✓ TTS generated successfully ({len(response.content)} bytes){Style.RESET_ALL}")

        return {
            "status": "ok",
            "audio": audio_base64,
            "format": "mp3"
        }

    except requests.Timeout:
        logger.error("ElevenLabs request timeout")
        raise HTTPException(status_code=504, detail="TTS request timeout")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)