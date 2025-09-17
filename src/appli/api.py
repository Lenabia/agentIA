from fastapi import FastAPI, HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict

from appli.crew import Appli


app = FastAPI(title="OF Chatbot API (minimal)", version="0.1.0")

# CORS ouvert pour faciliter les appels depuis un front
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    """Vérifie que le serveur est démarré."""
    return {"status": "ok"}


@app.get("/ask")
def ask(
    question: str = Query(..., description="Question utilisateur"),
    data: str = Query(..., description="Contexte à passer à l'agent"),
) -> Dict[str, str]:
    """Endpoint minimal: question et data (tous deux obligatoires) → réponse de l'agent."""
    try:
        result = Appli().crew().kickoff(inputs={"question": question, "of_context": data})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur agent: {exc}")
    return {"answer": str(result) if result is not None else ""}

