"""
Standalone diagnostic script -- NOT part of the app.

Purpose: isolate a single Step 6 (construct elicitation + laddering) call,
with a real image attached, completely outside the Streamlit app. This
reuses your actual llm_client.py and interview_engine.py -- same client,
same model, same Pydantic schemas -- just called directly so we can see
exactly what happens with nothing else in the way.

Usage:
    1. Put a real image named "test_image.jpg" (or .png) in this same folder.
    2. Set your API key:
         $env:GEMINI_API_KEY = "your-key-here"      (PowerShell)
    3. Run:
         python debug_step6.py

This is safe to delete once you're done -- it doesn't modify any of your
app's files or session data.
"""

import os
import sys
import glob
from PIL import Image

from llm_client import ZMETLLMClient, DEFAULT_MODEL
from interview_engine import TriadExtractionResult


def find_test_image():
    """Look for any image file in this folder to use as the test input."""
    candidates = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        candidates.extend(glob.glob(ext))
    return candidates[0] if candidates else None


def main():
    print("=" * 70)
    print("ZMET Step 6 diagnostic -- isolated laddering call")
    print("=" * 70)

    # ---- 1. Check API key ----
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[FAIL] No GEMINI_API_KEY environment variable found.")
        print("Set it first:  $env:GEMINI_API_KEY = \"your-key-here\"")
        sys.exit(1)
    print(f"\n[OK] API key found (starts with: {api_key[:6]}...)")

    # ---- 2. Check model ----
    model_name = os.environ.get("ZMET_TEST_MODEL", DEFAULT_MODEL)
    print(f"[OK] Using model: {model_name}")

    # ---- 3. Find a test image ----
    image_path = find_test_image()
    if not image_path:
        print("\n[FAIL] No image found in this folder.")
        print("Drop any .jpg/.jpeg/.png file next to this script and rerun.")
        sys.exit(1)
    print(f"[OK] Using test image: {image_path}")

    try:
        img = Image.open(image_path)
        img.load()  # force-read the file now, so any corruption fails here, not later
    except Exception as e:
        print(f"\n[FAIL] Could not open image file: {e}")
        sys.exit(1)
    print(f"[OK] Image loaded successfully: {img.size}, mode={img.mode}, format={img.format}")

    # ---- 4. Build the client (same class the app uses) ----
    client = ZMETLLMClient(api_key=api_key, model_name=model_name)

    # ---- 5. Build a minimal Step-6-style prompt, same shape as interview_engine.py ----
    research_topic = "commuting to work"  # placeholder -- change if you want to test topic-awareness
    prompt = f"""
    You are a professional ZMET researcher conducting Kelly Repertory Grid elicitation.
    This interview is specifically about: "{research_topic}".
    The participant was uploaded this single image and was asked what it represents about this topic.

    Analyze the attached image and extract:
    1. The similarity construct (an Attribute: a standardized construct name, e.g., 'Aesthetic Comfort').
    2. The difference construct (an Attribute: e.g., 'Chaotic Environment').
    3. A laddering question probing this construct, referencing the research topic.

    For this single-image test, treat 'similar_image_ids' and 'different_image_id' as the same
    placeholder ID "test_img_01" -- we only care whether the call succeeds and returns valid JSON.

    Return this strictly in the requested JSON structure.
    """

    print("\n" + "-" * 70)
    print("Sending live request to Gemini (image + structured schema)...")
    print("-" * 70)

    # ---- 6. Make the actual call, with nothing hidden ----
    try:
        raw_response = client.generate_content(
            prompt=prompt,
            image=img,
            response_schema=TriadExtractionResult,
        )
        print("\n[OK] Raw response received from Gemini:")
        print(raw_response)

        parsed = TriadExtractionResult.model_validate_json(raw_response)
        print("\n[OK] Successfully parsed into TriadExtractionResult:")
        print(f"  similarity_construct : {parsed.similarity_construct}")
        print(f"  difference_construct : {parsed.difference_construct}")
        print(f"  next_question        : {parsed.next_question}")

        print("\n" + "=" * 70)
        print("RESULT: The AI call + image + schema all work correctly in isolation.")
        print("If the app still shows fallback/questionnaire behavior, the bug is")
        print("in how app.py / interview_engine.py wires this result into the")
        print("session and UI -- NOT in the AI call itself.")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAIL] The call raised an exception:")
        print(f"  {type(e).__name__}: {e}")
        print("\n" + "=" * 70)
        print("RESULT: This is the real, underlying error. Whatever is printed")
        print("above is what's actually breaking Step 6 -- fix this specific")
        print("error and the app should follow.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
