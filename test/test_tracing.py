import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2")
endpoint = os.getenv("LANGCHAIN_ENDPOINT")
api_key = os.getenv("LANGCHAIN_API_KEY")
project = os.getenv("LANGCHAIN_PROJECT")

print("--- LangSmith Tracing Configuration ---")
print(f"LANGCHAIN_TRACING_V2: {tracing_enabled}")
print(f"LANGCHAIN_ENDPOINT:   {endpoint}")
print(f"LANGCHAIN_PROJECT:    {project}")

if not api_key or api_key == "your_langsmith_api_key_here":
    print("\n[!] LANGCHAIN_API_KEY is not set!")
else:
    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"LANGCHAIN_API_KEY:    {masked_key} (Configured)")
    print("\n--> Sending test query to Gemini with automatic LangSmith tracing...")

    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    res = llm.invoke("Explain in one short sentence what LangSmith tracing is.")
    print("\nModel Response:")
    print(res.content)
    print("\n[OK] Trace sent successfully to LangSmith!")
    print(f"     View your live trace at: https://smith.langchain.com/o/projects/p/{project}")
