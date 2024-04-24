# libs
"""
https://huggingface.co/models?pipeline_tag=text-classification&sort=trending
%pip install torch transformers --quiet
%pip install --upgrade jupyter ipywidgets --quiet
%pip install --upgrade tqdm --quiet
%pip install pandas
"""
from transformers import pipeline
# Inicializar o pipeline de sumarização de texto
pipe_sumarizacao = pipeline("summarization", model="Falconsai/text_summarization")
# Contexto sobre o caso de negócio no varejo e-commerce (exemplo usando Shopee)
relatorio_shein = """
A Shein é uma empresa de comércio eletrônico que cresceu rapidamente nos últimos anos. Fundada em 2008, a Shein se destaca por oferecer uma ampla variedade de roupas, acessórios e produtos de beleza a preços acessíveis para consumidores em todo o mundo. A empresa adota uma abordagem de moda rápida, lançando novos produtos regularmente e mantendo-se atualizada com as últimas tendências da moda.
Em 2023, a Shein experimentou um aumento significativo nas vendas, impulsionado pelo crescimento do comércio eletrônico durante a pandemia de COVID-19. A empresa expandiu sua presença global, lançando novos sites e aplicativos em diferentes idiomas e moedas para atender a uma base de clientes cada vez maior. Além disso, a Shein investiu em iniciativas de marketing digital e parcerias com influenciadores para promover seus produtos e alcançar novos públicos.
No entanto, a Shein também enfrentou desafios, incluindo críticas sobre questões de sustentabilidade e práticas de trabalho em suas cadeias de suprimentos. A empresa enfrentou acusações de produção excessiva e descarte de resíduos, bem como denúncias de condições precárias de trabalho em suas fábricas.
Apesar dos desafios, a Shein continua a ser uma das principais empresas de comércio eletrônico do mundo, conhecida por sua ampla seleção de produtos, preços acessíveis e engajamento com a comunidade de moda online.
"""
# Obter um resumo do contexto fornecido
resumo = pipe_sumarizacao(relatorio_shein)
# Imprimir 
print("Resumo do caso de negócio:")
print(resumo[0]['summary_text'])