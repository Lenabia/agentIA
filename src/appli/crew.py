from crewai import Agent, Task, Crew, Process, LLM
import os


class Appli:
    """Fabrique une crew minimale adaptée au chatbot OF.

    Inputs attendus à l'exécution:
    - question: texte de l'utilisateur
    - of_context: texte construit à partir de la ligne CSV de l'OF
    """

    def __init__(self) -> None:
        # Modèle lu depuis l'environnement (fallback sur gemma3:1b)
        model_name = os.getenv("MODEL", "ollama/gemma3:1b")
        self.llm = LLM(model=model_name, temperature=0)

    def crew(self) -> Crew:
        of_agent = Agent(
            role="Assistant OF interne",
            goal=(
                "Répondre précisément aux questions sur un OF en s'appuyant uniquement sur le contexte fourni."
                "Français uniquement."
                "Ne répète pas la phrase de description"
                "sur le contexte fourni (données internes issues du CSV)."
            ),
            backstory=(
                "Assistant interne spécialisé dans la lecture de données de suivi OF. "
            
            ),
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
        )

        of_task = Task(
            description=(
                "Réponds à {question} uniquement avec les infos de {of_context}. "
                "Réponds en Français, concis, précis, sans phrases inutiles."

               
            ),
            expected_output=(
                "Réponse brève en langage naturel, fidèle aux infos disponibles et pertinentes, sans format clé: valeur ni tableau. et pas besoin de répéter cette phrase"
            ),
            agent=of_agent,
        )

        return Crew(
            agents=[of_agent],
            tasks=[of_task],
            verbose=False,
            process=Process.sequential,
        )