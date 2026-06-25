from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    response = supabase.table("chat_history").select("*").limit(4).execute()
    print("✅ Supabase Connected & Query Works")
    print("Data:", response.data)

except Exception as e:
    print("❌ Connection / Query Failed:", e)