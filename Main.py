from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
import time
import os
from dotenv import load_dotenv


# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = FastAPI(title="Aria AI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

load_dotenv()

# OPENROUTER_API_KEY ="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"  # 🔑 Apna key daalo
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = [
    "meta-llama/llama-3.3-70b-instruct:free",
    # "stepfun/step-3.5-flash:free",
    "openai/gpt-oss-120b:free",
]
current_model = 0

SYSTEM_PROMPT = SYSTEM_PROMPT = """You are Aria, a smart AI voice assistant and receptionist.

BUSINESS INFO (use this when asked about the business):
- Business Name: Sharma Clinic
- Services: General checkup, Blood tests, X-Ray, Vaccination
- Working Hours: Monday to Saturday, 9 AM to 7 PM. Closed on Sunday.
- Location: 12, MG Road, Near City Mall, Jaipur
- Contact: 98765-43210
- Doctor: Dr. Ramesh Sharma (MBBS, MD)

For general questions, be a helpful friendly assistant.

RULES:
- Max 2-3 sentences per response
- No bullet points or lists
- Respond in the same language the user uses (Hindi, English, or Hinglish)
- Never say you cannot understand Hindi or Hinglish
"""

# ─── In-memory conversation store ─────────────────────────────────────────────
# session_id → history list
sessions: dict = {}


# ─── MODELS ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    aria_response: str


# ─── LLM CALL ─────────────────────────────────────────────────────────────────
def ask_aria(user_input: str, history: list) -> str:
    global current_model
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})

    messages.append({"role": "user", "content": user_input})

    for attempt in range(3):
        model = MODEL[current_model]
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": model,
                    "messages": messages,
                    "max_tokens": 150,
                    "temperature": 0.7
                }),
                timeout=15
        )


            result = response.json()
            print(f"🔍 API Response: {result}")  # Debug ke liye

            if "error" in result:
                if result["error"].get("code") == 429:
                    print(f"⏳ {model} rate limited — next model try kar raha hoon...")
                    current_model = (current_model + 1) % len(MODEL)
                    continue
                raise HTTPException(status_code=500, detail=result["error"]["message"])

            return result["choices"][0]["message"]["content"].strip()

        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Error: {e}")
            current_model = (current_model + 1) % len(MODEL)

    raise HTTPException(status_code=429, detail="Sab models rate limited hain — thodi der baad try karo")



# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Aria AI Assistant API is running! 🤖"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main chat endpoint — message bhejo, Aria ka response pao"""
    session_id = request.session_id
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message empty hai!")

    # Session history get karo ya nayi banao
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    # Aria se response lo
    aria_response = ask_aria(user_message, history)

    # History update karo
    history.append({"user": user_message, "bot": aria_response})

    # History 10 turns tak rakho — memory management
    if len(history) > 10:
        history.pop(0)

    return ChatResponse(
        session_id=session_id,
        user_message=user_message,
        aria_response=aria_response
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Session/history clear karo"""
    if session_id in sessions:
        del sessions[session_id]
    return {"message": f"Session '{session_id}' cleared!"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}