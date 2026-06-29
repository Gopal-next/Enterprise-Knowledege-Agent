import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL,SUPABASE_KEY)

def save_chat(question,answer,tool_used,response_time):

    data = {
        "question": question,
        "answer": answer,
        "tool_used": tool_used,
        "response_time": response_time,
        "created_at": datetime.now().isoformat()
    }

    response = (
        supabase
        .table("chat_history")
        .insert(data)
        .execute()
    )

    return response