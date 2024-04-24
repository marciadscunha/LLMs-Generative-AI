# libs
"""
https://huggingface.co/models?pipeline_tag=text-classification&sort=trending
%pip install torch transformers --quiet
%pip install --upgrade jupyter ipywidgets --quiet
%pip install --upgrade tqdm --quiet
%pip install pandas
"""
from transformers import pipeline

# Inicializar o pipeline de zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Descrição do produto
descricao_produto = "The Samsung Galaxy S24 is an advanced and comprehensive Android smartphone across the board\
                    view with some excellent features. It has a large 6.2-inch screen with a resolution\
                    2340x1080 pixels. The features offered by the Samsung Galaxy S24 are many and innovative. \
                     Starting with 5G, which allows data transfer and excellent internet browsing."

# Categorias disponíveis
categories = ["Clothes", "Electronics", "Home Accessories"]
# Classificar a descrição do produto nas categorias disponíveis
resultado = classifier(descricao_produto, candidate_labels=categories)

# Imprimir os resultados
print("Descrição do Produto:", descricao_produto)
print("Categorias Disponíveis:", categories)
print("Classificação Resultante:")
for label, score in zip(resultado['labels'], resultado['scores']):
    print(f" - {label}: {score:.3f}")
