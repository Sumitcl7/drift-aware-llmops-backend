import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print("Loaded URL:", url)

supabase = create_client(url, key)