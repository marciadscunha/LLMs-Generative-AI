# libs
"""
https://huggingface.co/models?pipeline_tag=text-classification&sort=trending
%pip install torch transformers sklearn --quiet
%pip install --upgrade jupyter ipywidgets --quiet
%pip install --upgrade tqdm --quiet
%pip install pandas
"""
# Importando as bibliotecas necessárias
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Passo 1: Carregar os dados proprios - your dataset
# Vamos supor que temos dados de tweets e seus sentimentos (positivo ou negativo)
tweets = ["Estou muito feliz hoje!", "Que dia terrível"]  # Lista de tweets
sentimentos = [1, 0]  # Lista de sentimentos (1 para positivo, 0 para negativo)

# Passo 2: Dividir os dados em treino e teste
tweets_treino, tweets_teste, sentimentos_treino, sentimentos_teste = train_test_split(
    tweets, sentimentos, test_size=0.2, random_state=42)

# Passo 3: Carregar o modelo pré-treinado e tokenizer
modelo = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)   
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Passo 4: Tokenização dos dados
tokens_treino = tokenizer(tweets_treino, padding=True, truncation=True, return_tensors='pt')
tokens_teste = tokenizer(tweets_teste, padding=True, truncation=True, return_tensors='pt')

# Passo 5: Criar DataLoader para treino e teste
dados_treino = TensorDataset(tokens_treino['input_ids'], tokens_treino['attention_mask'], torch.tensor(sentimentos_treino))
dados_teste = TensorDataset(tokens_teste['input_ids'], tokens_teste['attention_mask'], torch.tensor(sentimentos_teste))

loader_treino = DataLoader(dados_treino, batch_size=8, shuffle=True)
loader_teste = DataLoader(dados_teste, batch_size=8)

# Passo 6: Definir os parâmetros de treinamento
optimizer = AdamW(modelo.parameters(), lr=1e-5)
epochs = 3

# Passo 7: Treinamento do modelo
for epoch in range(epochs):
    modelo.train()
    for batch in loader_treino:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = modelo(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Passo 8: Avaliação do modelo
modelo.eval()
acuracia = 0
total = 0
with torch.no_grad():
    for batch in loader_teste:
        input_ids, attention_mask, labels = batch
        outputs = modelo(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicoes = torch.argmax(logits, dim=1)
        acuracia += (predicoes == labels).sum().item()
        total += len(labels)

print(f'Acurácia do modelo: {acuracia / total}')
