from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional
import logging

from appli.crew import Appli  # logique métier/agent

# --- Application FastAPI ---
app = FastAPI(title="OF Chatbot API (POST minimal)", version="0.2.3")

# --- CORS : suffisant pour appels front simples (adapter en prod) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # en prod : lister vos domaines
    allow_credentials=False,    # True => pas de "*" ; exige origines explicites
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# --- Logging simple ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("appli.api")

# --- Instance unique de l'agent et de la crew (évite de recréer à chaque requête) ---
appli = Appli()
try:
    appli_crew: Optional[object] = appli.crew()
except Exception as exc:
    # Si l'init échoue au démarrage (ex: LLM indisponible), on retentera à la 1ʳᵉ requête
    logger.warning("Initialisation de la crew échouée: %s", exc)
    appli_crew = None

# --- Schéma d'entrée : valide le JSON reçu ---
class AskIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Question utilisateur")
    data: str = Field(..., min_length=1, max_length=10000, description="Contexte pour l'agent")

# --- Schéma de sortie : réponse proprement typée ---
class AskOut(BaseModel):
    answer: str

# --- Healthcheck simple ---
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

# --- Endpoint principal : POST /ask ---
@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn) -> AskOut:
    """
    Reçoit {"question": "...", "data": "..."} ; déclenche l'agent ; renvoie {"answer": "..."}.
    """
    try:
        # Initialise une fois si nécessaire
        global appli_crew
        if appli_crew is None:
            appli_crew = appli.crew()

        # Appel à la logique agent
        result = appli_crew.kickoff(inputs={"question": payload.question, "of_context": payload.data})
    except Exception as exc:
        # Journalise l'erreur côté serveur, renvoie un message générique au client
        logger.exception("Erreur lors de l'exécution de l'agent: %s", exc)
        raise HTTPException(status_code=500, detail="Erreur agent")

    # Normalise la sortie en string et garantit le contrat de réponse
    return AskOut(answer=str(result) if result is not None else "")
