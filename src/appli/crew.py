from crewai import Agent, Task, Crew, Process, LLM


class Appli:
    """Fabrique une crew minimale adaptée au chatbot OF.

    Inputs attendus à l'exécution:
    - question: texte de l'utilisateur
    - of_context: texte construit à partir de la ligne CSV de l'OF
    """

    def __init__(self) -> None:
        # LLM Ollama Mistral, temp=0 pour plus de déterminisme des réponses
        self.llm = LLM(model="ollama/mistral:7b", temperature=0)

    def crew(self) -> Crew:
        of_agent = Agent(
            role="Assistant OF interne",
            goal=(
                "Répondre précisément aux questions sur un OF en s'appuyant uniquement "
                "sur le contexte fourni (données internes issues du CSV)."
            ),
            backstory=(
                "Assistant interne spécialisé dans la lecture de données de suivi OF. "
                "Il ne fabrique pas d'informations en dehors du contexte fourni."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        of_task = Task(
            description=(
                "Réponds à la question utilisateur: {question} en te basant uniquement "
                "sur le contexte de l'OF suivant: {of_context}. Si une information est "
                "absente du contexte, indique-le clairement. Fournis une réponse courte, "
                "claire et actionnable. Réponds en français uniquement."
            ),
            expected_output=(
                "Une réponse structurée et fidèle aux données du contexte, listant les "
                "champs clés (date, statuts, remarques, référence, conformité) si présents."
            ),
            agent=of_agent,
        )

        return Crew(
            agents=[of_agent],
            tasks=[of_task],
            verbose=True,
            process=Process.sequential,
        )