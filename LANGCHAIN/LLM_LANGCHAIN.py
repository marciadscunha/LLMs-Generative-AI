
# Cria um agente para executar código Python
from langchain.agents.agent_toolkits import create_python_agent  
# Ferramentas e inicialização de agentes
from langchain.agents import load_tools, initialize_agent, AgentType  
# Ferramenta para execução de código Python
from langchain.tools.python.tool import PythonREPLTool 
# Modelo de linguagem baseado no GPT 
from langchain.chat_models import ChatOpenAI  
# Interpretador Python em tempo real
from langchain.python import PythonREPL  


# Vou usar o GPT-4 para criar um agente capaz de interpretar e responder perguntas.
llm = ChatOpenAI(model_name="gpt-4")

# ** O PythonREPLTool permite que o agente execute código Python diretamente. **
python_tool = PythonREPLTool()


# Ainda posso add outras ferramentas como APIs externas.
tools = [python_tool]
#tools = load_tools(["python_repl", "serpapi"]) <<<<<<<


# Vou usar o ZERO_SHOT_REACT_DESCRIPTION que permite ao agente decidir qual ferramenta usar com base na descrição da tarefa.
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Vou criar um agente especializado em interpretar e executar código Python.
python_agent = create_python_agent(llm=llm)



#1. Testando:
response_from_agent = agent.run("Qual é o resultado de 10 * (2 + 3)?")
print("Resposta do agente:", response_from_agent) 

# 2. Usando o agente Python diretamente para cálculos matemáticos
response_from_python_agent = python_agent.run("Calcule a raiz quadrada de 16.")
print("Resposta do agente Python:", response_from_python_agent)  

# 3. Testando o Python REPL diretamente
python_repl = PythonREPL()
response_from_repl = python_repl.run("2 ** 3")  
print("Resposta do Python REPL:", response_from_repl)
