# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from crewai import LLM
from dotenv import load_dotenv
import os, json

# ---------------- Init ----------------
load_dotenv()
app = FastAPI(title="OF Chatbot API (LLM direct)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ⚠️ restreindre en prod
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

MODEL = os.getenv("MODEL", "ollama/gemma3:1b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
llm = LLM(model=MODEL, temperature=TEMPERATURE)

# ------------- Schémas I/O -------------
class AskIn(BaseModel):
    question: str = Field(..., min_length=1, description="Question utilisateur")
    data: dict = Field(..., description="Contexte OF au format JSON (clé/valeur)")

class AskOut(BaseModel):
    answer: str

# ------------- Utils (facultatif) -------------
def _build_prompt(question: str, data: dict) -> str:
    """
    Prompt court, cadré :
      - répond UNIQUEMENT depuis le JSON fourni,
      - pas d'invention ni d'interprétation hors des clés/valeurs,
      - français, 1–2 phrases max,
      - fallback clair si info absente.
    """
    context = json.dumps(data, ensure_ascii=False)
    return (
        "Tu es un assistant qui répond UNIQUEMENT à partir du JSON fourni.\n"
        "Règles:\n"
        "- Français uniquement, 1–2 phrases maximum, ton neutre.\n"
        "- Utilise STRICTEMENT les clés/valeurs du JSON (aucun synonyme inventé, aucune déduction externe).\n"
        "- Si l'information demandée n'est pas présente dans le JSON, réponds EXACTEMENT : "
        "\"Information non trouvée dans le contexte.\"\n\n"
        f"JSON:\n{context}\n\n"
        f"Question:\n{question}\n"
    )

# ------------- Endpoints -------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn) -> AskOut:
    prompt = _build_prompt(payload.question, payload.data)
    try:
        raw = llm.call(prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur LLM: {exc}")

    answer = (str(raw).strip() if raw is not None else "")
    # garde-fou minimal si le modèle renvoie vide
    if not answer:
        answer = "Information non trouvée dans le contexte."
    return AskOut(answer=answer)
