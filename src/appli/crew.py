from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

pdf_source = PDFKnowledgeSource(
    file_paths= ["docaresumer.pdf"]
)
# Create a knowledge source
content = 'Users name is John he lives in San Francisco and he is 30 years old.'
string_source = StringKnowledgeSource(content=content)
# Create an LLM with a temperature of 0 to ensure deterministic output
llm = LLM(model="ollama/mistral:7b", temperature=0)
# Create an agent with the knowledge store
agent = Agent(
    role="Resume",
    goal="Tu es un expert en résumé de documents.",
    backstory="Tu a 20 d'expérience dans le résumé de documents.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    knowledge_sources=[pdf_source],
    embedder={"provider": "ollama", "config": {"model": "mxbai-embed-large"}}  # embedder pour vectoriser le knowledge

)

agent_traducteur = Agent(
    role= "Traducteur",
    goal="Tu es un expert en traduction.",
    backstory="Tu a 20 d'expérience dans la traduction.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)
    

task = Task(
    description="résume le document en 100 mots",
    expected_output="Le résumé du texte du document.",
    agent=agent,
)

task = Task(
    description="traduit le document en français",
    expected_output="Le document traduit en français.",
    agent=agent,
)

crew = Crew(
    agents=[agent, agent_traducteur],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
   
)

result = crew.kickoff()