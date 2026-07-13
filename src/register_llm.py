"""Register Gemini LLM config in IRIS AI Hub config store at container startup."""
import json, os, iris

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    raise SystemExit("GEMINI_API_KEY not set — skipping AI Hub registration")

irispy = iris.createIRIS()

# Try to create an AI.LLM config object via the AI Hub API
try:
    # Attempt the documented %ConfigStore.Configuration.Set with a JSON string
    config_json = json.dumps({
        "model": "gemini-2.0-flash",
        "model_provider": "google_genai",
        "api_key": api_key,
    })
    sc = irispy.classMethodValue("%ConfigStore.Configuration", "Set", "AI.LLM.gemini", config_json)
    if sc == 1:
        print("Gemini LLM config registered in AI Hub (JSON method).")
    else:
        err = irispy.classMethodValue("%SYSTEM.Status", "GetErrorText", sc)
        print(f"Warning: Set returned error: {err}")
        # Fallback: store in a simple Global so gaia.py can read it directly
        iris.set(api_key, "^AIHubConfig", "gemini_api_key")
        print("Gemini API key stored in ^AIHubConfig Global as fallback.")
except Exception as e:
    print(f"Warning: AI Hub config registration failed: {e}")
    iris.set(api_key, "^AIHubConfig", "gemini_api_key")
    print("Gemini API key stored in ^AIHubConfig Global as fallback.")
