"""
memory.py - Thread-safe SQLite database layer for Memento Protocol
Enhanced with caregiver dashboard support, memory statistics, and cascade operations.
"""

import sqlite3
import re
from contextlib import contextmanager
from typing import Optional, List, Dict
from datetime import datetime

from colorama import Fore, Style

from config import DB_PATH


@contextmanager
def get_connection():
    """
    Context manager for thread-safe database connections.
    Each function gets its own connection - critical for async/WebSocket safety.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """
    Initialize database schema with enhanced columns for caregiver platform.
    Safe to call multiple times - uses IF NOT EXISTS.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Enhanced memories table with source tracking and confidence scoring
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                topic TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'unasked',
                source TEXT DEFAULT 'conversation',
                confidence REAL DEFAULT 1.0
            )
        """)
        
        # Enhanced known_people table with bio and photo support
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS known_people (
                person_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                relationship TEXT,
                visual_marker TEXT,
                bio TEXT DEFAULT '',
                photo_url TEXT DEFAULT '',
                face_image_path TEXT DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Face embeddings table for face recognition
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                model_name TEXT DEFAULT 'Facenet',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES known_people(person_id)
            )
        """)
        
        # Last conversation tracking - stores the last conversation summary for reminders
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS last_conversations (
                person_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                topics TEXT DEFAULT '',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES known_people(person_id)
            )
        """)
        
        # Performance indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_person_status 
            ON memories(person_id, status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp 
            ON memories(timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_people_marker
            ON known_people(visual_marker)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_embeddings_person
            ON face_embeddings(person_id)
        """)

        # Migration: Add face_image_path column if it doesn't exist
        cursor.execute("PRAGMA table_info(known_people)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'face_image_path' not in columns:
            cursor.execute("ALTER TABLE known_people ADD COLUMN face_image_path TEXT DEFAULT ''")
        
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} Database initialized ({DB_PATH})")


def _normalize_fact(fact: str) -> str:
    """Normalize fact string for duplicate comparison."""
    normalized = fact.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def _is_duplicate_fact(conn, person_id: str, fact: str, similarity_threshold: float = 0.8) -> bool:
    """Check if a similar fact already exists using Jaccard similarity."""
    cursor = conn.cursor()
    cursor.execute("SELECT fact FROM memories WHERE person_id = ?", (person_id,))
    existing_facts = [row['fact'] for row in cursor.fetchall()]
    
    if not existing_facts:
        return False
    
    new_normalized = _normalize_fact(fact)
    new_words = set(new_normalized.split())
    
    for existing in existing_facts:
        existing_normalized = _normalize_fact(existing)
        existing_words = set(existing_normalized.split())
        
        if not new_words or not existing_words:
            continue
        
        intersection = len(new_words & existing_words)
        union = len(new_words | existing_words)
        similarity = intersection / union if union > 0 else 0
        
        if similarity >= similarity_threshold:
            return True
    
    return False


def save_memory(
    person_id: str, 
    fact: str, 
    topic: str = None, 
    source: str = 'conversation',
    confidence: float = 1.0,
    check_duplicates: bool = True
) -> Optional[int]:
    """
    Save a new memory/fact about a person.
    
    Args:
        person_id: ID of the person this memory is about
        fact: The fact/memory text
        topic: Optional topic category
        source: Origin of memory ('conversation', 'caregiver', 'system')
        confidence: AI confidence score (0.0-1.0)
        check_duplicates: Skip if similar fact exists
    
    Returns:
        Memory ID or None if duplicate
    """
    with get_connection() as conn:
        if check_duplicates and _is_duplicate_fact(conn, person_id, fact):
            print(f"{Fore.YELLOW}⚠ Duplicate skipped:{Style.RESET_ALL} {fact[:40]}...")
            return None
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memories (person_id, fact, topic, timestamp, status, source, confidence)
            VALUES (?, ?, ?, ?, 'unasked', ?, ?)
        """, (person_id, fact, topic, datetime.now(), source, confidence))
        
        memory_id = cursor.lastrowid
    
    source_icon = "💬" if source == 'conversation' else "📝" if source == 'caregiver' else "🔔"
    topic_str = f" [{topic}]" if topic else ""
    print(f"{Fore.CYAN}{source_icon} Memory{topic_str}:{Style.RESET_ALL} {fact[:60]}...")
    
    return memory_id


def get_unasked_memories(person_id: str, limit: int = 3) -> List[Dict]:
    """
    Get unasked memories and atomically mark them as 'asked'.
    Prevents race conditions in concurrent access scenarios.
    """
    memories = []
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, fact, topic, timestamp, source, confidence
            FROM memories
            WHERE person_id = ? AND status = 'unasked'
            ORDER BY timestamp DESC
            LIMIT ?
        """, (person_id, limit))
        
        rows = cursor.fetchall()
        
        if rows:
            memory_ids = [row['id'] for row in rows]
            placeholders = ','.join(['?' for _ in memory_ids])
            cursor.execute(f"""
                UPDATE memories SET status = 'asked'
                WHERE id IN ({placeholders})
            """, memory_ids)
            
            memories = [
                {
                    'id': row['id'],
                    'fact': row['fact'],
                    'topic': row['topic'],
                    'timestamp': row['timestamp'],
                    'source': row['source'],
                    'confidence': row['confidence']
                }
                for row in rows
            ]
    
    return memories


def get_all_memories(person_id: str, limit: int = 20) -> List[Dict]:
    """Get all memories for a specific person."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, fact, topic, timestamp, status, source, confidence
            FROM memories
            WHERE person_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (person_id, limit))
        
        return [dict(row) for row in cursor.fetchall()]


def get_recent_memories(limit: int = 20) -> List[Dict]:
    """
    Get recent memories across all people with person details.
    Powers the dashboard memory stream.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                m.id,
                m.person_id,
                m.fact,
                m.topic,
                m.timestamp,
                m.status,
                m.source,
                m.confidence,
                p.display_name,
                p.visual_marker
            FROM memories m
            JOIN known_people p ON m.person_id = p.person_id
            ORDER BY m.timestamp DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]


def add_caregiver_note(person_id: str, note: str, topic: str = 'caregiver') -> Optional[int]:
    """
    Add a caregiver-authored memory note.
    Returns memory ID or None if person doesn't exist.
    """
    # Verify person exists
    if not get_person_by_id(person_id):
        print(f"{Fore.RED}✗ Person not found: {person_id}{Style.RESET_ALL}")
        return None
    
    return save_memory(
        person_id=person_id,
        fact=note,
        topic=topic,
        source='caregiver',
        confidence=1.0,
        check_duplicates=True
    )


def get_memory_stats() -> Dict:
    """
    Get aggregate statistics for dashboard header.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Total counts
        cursor.execute("SELECT COUNT(*) as count FROM memories")
        total_memories = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM known_people")
        total_people = cursor.fetchone()['count']
        
        # Memories by topic
        cursor.execute("""
            SELECT topic, COUNT(*) as count 
            FROM memories 
            WHERE topic IS NOT NULL
            GROUP BY topic
        """)
        by_topic = {row['topic']: row['count'] for row in cursor.fetchall()}
        
        # Memories by source
        cursor.execute("""
            SELECT source, COUNT(*) as count 
            FROM memories 
            GROUP BY source
        """)
        by_source = {row['source']: row['count'] for row in cursor.fetchall()}
        
        return {
            'total_memories': total_memories,
            'total_people': total_people,
            'by_topic': by_topic,
            'by_source': by_source
        }


def register_person(
    person_id: str,
    display_name: str,
    relationship: str,
    bio: str = "",
    photo_url: str = "",
    visual_marker: str = ""
):
    """
    Register or update a known person.
    Uses INSERT OR REPLACE for idempotent operations.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO known_people
            (person_id, display_name, relationship, bio, photo_url, visual_marker, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (person_id, display_name, relationship, bio, photo_url, visual_marker, datetime.now()))

    print(f"{Fore.GREEN}✓{Style.RESET_ALL} Registered: {display_name} ({relationship})")


def delete_person(person_id: str) -> bool:
    """
    Delete a person and cascade delete all their memories and face embeddings.
    Returns True if person existed and was deleted.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        # Check if person exists
        cursor.execute("SELECT person_id FROM known_people WHERE person_id = ?", (person_id,))
        if not cursor.fetchone():
            return False

        # Cascade delete memories first
        cursor.execute("DELETE FROM memories WHERE person_id = ?", (person_id,))
        deleted_memories = cursor.rowcount

        # Cascade delete face embeddings
        cursor.execute("DELETE FROM face_embeddings WHERE person_id = ?", (person_id,))
        deleted_embeddings = cursor.rowcount

        # Delete person
        cursor.execute("DELETE FROM known_people WHERE person_id = ?", (person_id,))

    print(f"{Fore.YELLOW}🗑 Deleted: {person_id} ({deleted_memories} memories, {deleted_embeddings} face embeddings){Style.RESET_ALL}")
    return True


def get_person_by_marker(marker: str) -> Optional[Dict]:
    """Look up a person by their visual marker (clothing color)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT person_id, display_name, relationship, visual_marker, bio, photo_url, face_image_path, created_at
            FROM known_people
            WHERE visual_marker = ?
        """, (marker,))

        row = cursor.fetchone()
        return dict(row) if row else None


def get_person_by_id(person_id: str) -> Optional[Dict]:
    """Look up a person by their ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT person_id, display_name, relationship, visual_marker, bio, photo_url, face_image_path, created_at
            FROM known_people
            WHERE person_id = ?
        """, (person_id,))

        row = cursor.fetchone()
        return dict(row) if row else None


def get_all_people() -> List[Dict]:
    """Get all registered people with memory counts."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                p.person_id,
                p.display_name,
                p.relationship,
                p.visual_marker,
                p.bio,
                p.photo_url,
                p.face_image_path,
                p.created_at,
                COUNT(m.id) as memory_count
            FROM known_people p
            LEFT JOIN memories m ON p.person_id = m.person_id
            GROUP BY p.person_id
            ORDER BY p.created_at DESC
        """)

        return [dict(row) for row in cursor.fetchall()]


def clear_all_data():
    """Clear all data from database. Use for testing/reset."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories")
        cursor.execute("DELETE FROM known_people")
    
    print(f"{Fore.YELLOW}⚠{Style.RESET_ALL} Database cleared")


def reset_memory_status(person_id: str = None):
    """Reset memories to 'unasked' status for re-announcement."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if person_id:
            cursor.execute("UPDATE memories SET status = 'unasked' WHERE person_id = ?", (person_id,))
        else:
            cursor.execute("UPDATE memories SET status = 'unasked'")
    
    print(f"{Fore.CYAN}↻{Style.RESET_ALL} Memory status reset")


def save_last_conversation(person_id: str, summary: str, topics: str = "") -> bool:
    """
    Save or update the last conversation summary for a person.
    This is used for reminders when the person returns after 60+ seconds or after session restart.
    
    Args:
        person_id: ID of the person
        summary: Summary of what was discussed
        topics: Comma-separated list of topics discussed
    
    Returns:
        True if saved successfully
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO last_conversations (person_id, summary, topics, timestamp)
            VALUES (?, ?, ?, ?)
        """, (person_id, summary, topics, datetime.now()))
    
    print(f"{Fore.CYAN}💾 Last conversation saved for {person_id}: {summary[:50]}...{Style.RESET_ALL}")
    return True


def get_last_conversation(person_id: str) -> Optional[Dict]:
    """
    Get the last conversation summary for a person.
    Used for showing reminders when they return.
    
    Returns:
        Dict with 'summary', 'topics', 'timestamp' or None if no conversation recorded
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT summary, topics, timestamp
            FROM last_conversations
            WHERE person_id = ?
        """, (person_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'summary': row['summary'],
                'topics': row['topics'],
                'timestamp': row['timestamp']
            }
        return None


def clear_last_conversation(person_id: str) -> bool:
    """Clear the last conversation for a person (after it's been shown as a reminder)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM last_conversations WHERE person_id = ?", (person_id,))
    return True


def get_recent_topics(person_id: str, limit: int = 5) -> List[Dict]:
    """
    Get the 5 most recent memories/topics discussed with a person.
    Used to remind about recent conversations.
    
    Args:
        person_id: ID of the person
        limit: Number of recent memories to fetch (default 5)
    
    Returns:
        List of recent memories with fact, topic, and timestamp
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT fact, topic, timestamp
            FROM memories
            WHERE person_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (person_id, limit))
        
        return [dict(row) for row in cursor.fetchall()]


def get_important_personal_info(person_id: str) -> Dict:
    """
    Extract important personal information from memories:
    - Age
    - Profession/Job
    - School/Education
    
    Searches through all memories looking for keywords related to these topics.
    
    Args:
        person_id: ID of the person
    
    Returns:
        Dict with 'age', 'profession', 'school' keys (values may be None if not found)
    """
    result = {
        'age': None,
        'profession': None,
        'school': None
    }
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Get all memories for this person
        cursor.execute("""
            SELECT fact, topic
            FROM memories
            WHERE person_id = ?
            ORDER BY timestamp DESC
        """, (person_id,))
        
        memories = cursor.fetchall()
        
        for row in memories:
            fact = row['fact'].lower()
            topic = row['topic'].lower() if row['topic'] else ''
            
            # Look for age information
            if result['age'] is None:
                if 'age' in topic or 'age' in fact:
                    # Try to extract age-related info
                    result['age'] = row['fact']
                elif any(word in fact for word in ['years old', 'year old', 'born in', 'birthday']):
                    result['age'] = row['fact']
            
            # Look for profession/job information
            if result['profession'] is None:
                if topic in ['profession', 'job', 'work', 'career', 'employment']:
                    result['profession'] = row['fact']
                elif any(word in fact for word in ['works as', 'work as', 'job is', 'profession is', 
                                                    'employed as', 'working at', 'works at', 
                                                    'career', 'occupation', 'employed at']):
                    result['profession'] = row['fact']
            
            # Look for school/education information
            if result['school'] is None:
                if topic in ['school', 'education', 'university', 'college', 'study', 'student']:
                    result['school'] = row['fact']
                elif any(word in fact for word in ['studies at', 'goes to school', 'attends', 
                                                    'student at', 'studying', 'enrolled at',
                                                    'university', 'college', 'school', 'graduate',
                                                    'major in', 'majoring in', 'degree']):
                    result['school'] = row['fact']
            
            # Early exit if all found
            if all(result.values()):
                break
    
    return result


def save_face_embedding(person_id: str, embedding: bytes, model_name: str = "Facenet") -> Optional[int]:
    """
    Save a face embedding for a person.

    Args:
        person_id: ID of the person
        embedding: Serialized embedding as bytes (JSON-encoded list)
        model_name: Name of the model used to generate the embedding

    Returns:
        Embedding ID or None if person doesn't exist
    """
    if not get_person_by_id(person_id):
        print(f"{Fore.RED}✗ Person not found: {person_id}{Style.RESET_ALL}")
        return None

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO face_embeddings (person_id, embedding, model_name, created_at)
            VALUES (?, ?, ?, ?)
        """, (person_id, embedding, model_name, datetime.now()))

        embedding_id = cursor.lastrowid

    print(f"{Fore.GREEN}✓{Style.RESET_ALL} Face embedding saved for {person_id}")
    return embedding_id


def get_face_embeddings(person_id: str = None) -> List[Dict]:
    """
    Get face embeddings, optionally filtered by person_id.

    Args:
        person_id: Optional person ID to filter by. If None, returns all embeddings.

    Returns:
        List of dicts with person_id, embedding (bytes), model_name, created_at
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        if person_id:
            cursor.execute("""
                SELECT id, person_id, embedding, model_name, created_at
                FROM face_embeddings
                WHERE person_id = ?
                ORDER BY created_at DESC
            """, (person_id,))
        else:
            cursor.execute("""
                SELECT id, person_id, embedding, model_name, created_at
                FROM face_embeddings
                ORDER BY created_at DESC
            """)

        return [dict(row) for row in cursor.fetchall()]


def delete_face_embeddings(person_id: str) -> int:
    """
    Delete all face embeddings for a person.

    Args:
        person_id: ID of the person

    Returns:
        Number of embeddings deleted
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM face_embeddings WHERE person_id = ?", (person_id,))
        deleted_count = cursor.rowcount

    if deleted_count > 0:
        print(f"{Fore.YELLOW}🗑 Deleted {deleted_count} face embedding(s) for {person_id}{Style.RESET_ALL}")

    return deleted_count


def update_person_face_image(person_id: str, face_image_path: str) -> bool:
    """
    Update the face_image_path for a person.

    Args:
        person_id: ID of the person
        face_image_path: Path to the stored face image

    Returns:
        True if updated successfully
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE known_people
            SET face_image_path = ?
            WHERE person_id = ?
        """, (face_image_path, person_id))

        return cursor.rowcount > 0


def has_face_enrolled(person_id: str) -> bool:
    """
    Check if a person has a face enrolled.

    Args:
        person_id: ID of the person

    Returns:
        True if person has at least one face embedding
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM face_embeddings
            WHERE person_id = ?
        """, (person_id,))

        return cursor.fetchone()['count'] > 0


if __name__ == "__main__":
    init_db()
    print("\n✓ Database layer ready")