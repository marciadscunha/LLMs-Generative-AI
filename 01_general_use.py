# libs
"""
https://huggingface.co/models?pipeline_tag=text-classification&sort=trending
%pip install torch transformers --quiet
%pip install --upgrade jupyter ipywidgets --quiet
%pip install --upgrade tqdm --quiet
%pip install pandas
"""
from transformers import pipeline
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

# usando o modelo pré treinado de classificacao
#step 1
v_pipe = pipeline(task="text-classification")
# Lendo com pipe
review = "Adorei o produto, otimo custo beneficio."
v_pipe(review)

#step 2
# add model from https://huggingface.co/models?pipeline_tag=text-classification&sort=trending
classifier = pipeline(task="text-classification", 
                      model="nlptown/bert-base-multilingual-uncased-sentiment")

# utilizando com uma unica string
review_hate = "odiei o produto, pessimo custo beneficio."
classifier(review_hate)

# usando lista
list_of_review = [review, review_hate]
list_of_review_class = classifier(list_of_review)

# Criando um DataFrame com os resultados
df_output = pd.DataFrame({
    "Review": list_of_review,
    "Label": [item['label'] for item in list_of_review_class],
    "Score": [item['score'] for item in list_of_review_class]
})

print(df_output)