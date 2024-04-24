# libs
"""
https://huggingface.co/models?pipeline_tag=text-classification&sort=trending
%pip install torch transformers --quiet
%pip install --upgrade jupyter ipywidgets --quiet
%pip install --upgrade tqdm --quiet
%pip install pandas
"""
from transformers import pipeline

# Inicializar o pipeline de Question Answering
pipe_qa = pipeline("question-answering")

# Definir o contexto
contexto = "Josivaldo e Nelson são dois motoristas avaliados por uma empresa de transporte. \
Josivaldo recebeu uma avaliação muito positiva por seu excelente atendimento na região sul, \
onde demonstrou habilidades excepcionais de direção e serviço ao cliente. \
Por outro lado, Nelson também recebeu uma avaliação alta, destacando-se especialmente em suas atividades na região norte. \
Ambos os motoristas são altamente qualificados, mas parecem ter suas forças em diferentes áreas geográficas."
# Definir a pergunta
pergunta = "Quem deve atender a região norte?"
# Obter a resposta à pergunta com base no contexto fornecido
resposta = pipe_qa(context=contexto, question=pergunta)
# Imprimir a resposta
print("Resposta:", resposta['answer'])