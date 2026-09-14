from google import genai
import os

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("No GEMINI_API_KEY environment variable found.")
else:
    client = genai.Client(api_key=api_key)
    try:
        # gemini-flash-latest is a Google-maintained alias, confirmed
        # working on this account via debug_step6.py.
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents="Say hello"
        )
        print("gemini-flash-latest works:", response.text)
    except Exception as e:
        print("gemini-flash-latest error:", e)
        
    try:
        # Try to list models
        models = client.models.list()
        print("Available models:")
        for m in models:
            print(f"- {m.name}")
    except Exception as e:
        print("List models error:", e)
