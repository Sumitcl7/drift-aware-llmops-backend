from database.supabase_client import supabase

response = supabase.table("interaction_log").select("*").execute()

print(response)