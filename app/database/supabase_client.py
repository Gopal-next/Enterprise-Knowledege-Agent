import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

# Lazy init: only create client if credentials exist
_supabase_client = None

def _get_client():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return _supabase_client
    except Exception as e:
        print(f"Warning: Supabase init failed: {e}")
        return None

def save_chat(question, answer, tool_used, response_time):
    client = _get_client()
    if client is None:
        print("Supabase not configured – skipping chat log.")
        return None

    data = {
        "question": question,
        "answer": answer,
        "tool_used": tool_used,
        "response_time": response_time,
        "created_at": datetime.now().isoformat()
    }

    try:
        response = (
            client
            .table("chat_history")
            .insert(data)
            .execute()
        )
        return response
    except Exception as e:
        print(f"Warning: Failed to save chat to Supabase: {e}")
        return None