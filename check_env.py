from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENAI_API_KEY")

if key:
    print("✅ OPENAI_API_KEY is loaded")
    print("Key starts with:", key[:7])
else:
    print("❌ OPENAI_API_KEY is NOT loaded")