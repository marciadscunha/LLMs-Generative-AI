# !pip install --upgrade langchain openai
# https://www.sqlitetutorial.net/sqlite-sample-database/

from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_openai import OpenAI
from typing import Optional


def get_database_uri() -> str:
    """
    Retorna a URI de conexão ao banco de dados.
    """
    return "sqlite:///Chinook_Sqlite_Tmp.sqlite"


def initialize_llm(secret_key: str, model: Optional[str] = "gpt-4o-mini") -> OpenAI:
    """
    Inicializa o modelo LLM com a chave de API do OpenAI.
    :param secret_key: Chave de API para acessar o OpenAI.
    :param model: Modelo do OpenAI (default: text-davinci-003).
    :return: Instância do OpenAI configurada.
    """
    try:
        return OpenAI(api_key=secret_key, temperature=0)
    except Exception as e:
        raise ValueError(f"Erro ao inicializar o LLM: {e}")


def connect_to_database(uri: str) -> SQLDatabase:
    """
    Estabelece a conexão com o banco de dados a partir da URI fornecida.
    :param uri: URI do banco de dados.
    :return: Instância de SQLDatabase conectada.
    """
    try:
        return SQLDatabase.from_uri(uri)
    except Exception as e:
        raise ConnectionError(f"Erro ao conectar ao banco de dados: {e}")


def list_table_names(db: SQLDatabase) -> None:
    """
    Exibe os nomes das tabelas utilizáveis do banco de dados.
    :param db: Instância do SQLDatabase.
    """
    try:
        print("Tabelas disponíveis no banco de dados:")
        for table in db.get_usable_table_names():
            print(f"- {table}")
    except Exception as e:
        print(f"Erro ao listar tabelas: {e}")


def create_db_chain(llm: OpenAI, db: SQLDatabase) -> SQLDatabaseChain:
    """
    Cria uma instância de SQLDatabaseChain conectando o LLM ao banco de dados.
    :param llm: Instância do LLM configurada.
    :param db: Instância do SQLDatabase conectada.
    :return: SQLDatabaseChain configurado.
    """
    try:
        return SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)
    except Exception as e:
        raise RuntimeError(f"Erro ao criar a integração LLM-Banco: {e}")


def execute_query(db_chain: SQLDatabaseChain, query: str) -> str:
    """
    Executa uma consulta SQL gerada pela linguagem natural.
    :param db_chain: Instância configurada do SQLDatabaseChain.
    :param query: Consulta em linguagem natural para ser traduzida e executada.
    :return: Resultado da consulta em forma de string.
    """
    try:
        return db_chain.run(query)
    except Exception as e:
        raise RuntimeError(f"Erro ao executar consulta: {e}")


# Como usar
if __name__ == "__main__":
    
    uri = get_database_uri()
    llm = initialize_llm(secret_key=SECRET_KEY)
    db = connect_to_database(uri)

    #list_table_names(db)

    # Criar integração entre LLM e Banco
    db_chain = create_db_chain(llm, db)

    # Executar consulta em linguagem natural
    query = "What tables are there?"
    result = execute_query(db_chain, query)
    print(f"Resultado da consulta: {result}")
