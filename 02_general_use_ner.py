# libs
"""
https://huggingface.co/models?pipeline_tag=text-classification&sort=trending
%pip install torch transformers --quiet
%pip install --upgrade jupyter ipywidgets --quiet
%pip install --upgrade tqdm --quiet
%pip install pandas
"""
from transformers import pipeline
# Carregar o pipeline NER
pipe_ner = pipeline(task="ner")
# Texto de entrada
text = "Brazil is the country of football, consumers tend to buy items that bring them \
       comfort and status such as 'Adidas', 'brazil', 'John John', 'Gap', 'Nike'."
# Executar a extração de entidades no texto
entidades = pipe_ner(text)
entidades