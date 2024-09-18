import pandas as pd
import html
import nltk
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from unidecode import unidecode
import re
import emoji
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources (run once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('rslp', quiet=True)  # For RSLPStemmer

class TextPreprocessor:
    
    def __init__(self, languages=None):
        if languages is None:
            languages = ['portuguese', 'english', 'spanish']
        
        self.stopwords = set()
        for lang in languages:
            try:
                self.stopwords.update(stopwords.words(lang))
            except Exception as e:
                logging.warning(f"Could not load stopwords for language '{lang}': {e}")
        
        self.stemmer = RSLPStemmer()  # Optional stemmer
    
    def remove_location_unit(self, text):
        lista_loc_unit = [
            'sao paulo sp', 'goiania go', 'curitiba pr', 'porto alegre', 'sao jose', 'zona sul', 'cuiaba mt', 'salvador ba', 'nivel superior', 'fortaleza ce', 'porto alegre rs',
            'jundiai sp', 'brasilia df', 'uberlandia mg', 'maringa pr', 'caxias do sul', 'sao bernardo', 'londrina pr', 'recife pe', 'unidade sao', 'rs mg', 'campina grande ms',
            'barueri sp', 'ponta grossa', 'joinville sc', 'sao luis', 'rio verde', 'natal rn', 'teresina pi', 'aguas sertao', 'campina grande', 'joao pessoa', 'paulinia sp', 'randoncorp',
            'rio grande do sul', 'sao mateus', 'florianopolis sc', 'randon caxias', 'zona oeste', 'zona leste', 'zona sul', 'rio preto', 'sao jose dos pinhais', 'vila velha',
            'anapolis go', 'passo fundo', 'regional', 'duque caxias', 'santos sp', 'santa cruz', 'vera cruz', 'minas gerais', 'centro distribuicao', 'cascavel pr', 'pouso alegre', 'sorocaba sp',
            'blumenau sc', 'barra tijuca', 'sede sp', 'barra da tijuca', 'ponta grossa pr', 'chapeco sc', 'campos dos goytacazes', 'afya bh', 'mogi das cruzes', 'taboao da serra', 'sao joao',
            'itajai sc', 'camacari ba', 'uberaba mg', 'dourado sp', 'santo antonio', 'sicredi planalto', 'macae', 'volta redonda', 'sinop mt', 'niteroi rj', 'shopping', 'porto alegre', 'sao jose campos'
        ]

        pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, lista_loc_unit)) + r')\b', flags=re.IGNORECASE)
        return pattern.sub('', text)

    def remove_tags_and_content(self, raw_html):
        """
        Remove HTML tags and decode HTML entities in the input text.
        Args:raw_html (str): The raw HTML text to clean.
        Returns:str: Cleaned text without HTML tags and entities.
        """
        raw_html = html.unescape(raw_html)
        cleanr = re.compile('<[^>]*>')
        cleantext = re.sub(cleanr, '', raw_html)
        cleantext = re.sub('\s+', ' ', cleantext).strip()  # Remove extra spaces
        return cleantext
    
    def replace_terms_en_to_pt(self, text):
        term_mapping = {
            'complete': 'completo',
            'in_progress': 'em progresso',
            'progress': 'progresso',
            'incomplete': 'incompleto'
        }
        pattern = re.compile(r'\b(' + '|'.join(term_mapping.keys()) + r')\b')
        return pattern.sub(lambda x: term_mapping[x.group()], text)
    
    def remove_consecutive_duplicates(self, text):
        """
        Remove consecutive duplicate words in the text.
        Args:text (str): The text to remove duplicates from.
        Returns:str: The text without consecutive duplicates.
        """
        words = text.split()
        deduped_words = [words[0]] if words else []
        for i in range(1, len(words)):
            if words[i] != words[i - 1]:
                deduped_words.append(words[i])
        return ' '.join(deduped_words)
    
    
    def find_requirements_mandatory(self, requirements):
        # Expressões regulares para identificar os requisitos obrigatórios e desejáveis com variações
        regex_obrigatorios = r"(requisitos obrigatorios:|ensino medio|ensino medio completo|requisitos necessarios:|necessario:|obrigatorio:|requisitos indispensaveis:|indispensaveis:|exigido:|perfil obrigatorio:)([\s\S]*?)(requisitos desejaveis:|desejavel:|requisitos diferenciais:|diferenciais:|nao obrigatorio:|nao necessario:|opcional:|outros:|$)"
        obrigatorios = re.search(regex_obrigatorios, requirements, re.IGNORECASE)
        requisitos_obrigatorios = obrigatorios.group(2).strip() if obrigatorios else "Not Found"
        
        return requisitos_obrigatorios
    
    def remove_emoji(self, text):
        return emoji.replace_emoji(text, replace='')


    def preprocess_text(self, text, similar='S'):
        """
        Preprocess the input text by cleaning, removing stopwords, and stemming.
        Args:
            text (str): The text to preprocess.
        Returns:
            str: The preprocessed text.
        """
    
        try:
            text = str(text).lower()                         # Convert to lowercase
            text = self.remove_tags_and_content(text)        # Remove HTML tags
            text = text.replace('-', ' ').replace('&', 'e')
            text = unidecode(text)                           # Remove accents

            # Tratando para ambos textos requisitos e formação
            text = self.replace_terms_en_to_pt(text)         # Replace specific terms en to pt
            #if self.verificar_e_remover(text) is None:
            #    text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers            
            
            # Replace specific terms
            for term in dados_en:
                if term not in text:
                    text = text.replace(term, term.replace('_', ' '))
            #text = re.sub(r'[^a-z\s;,]', '', text)             # Remove special characters except space and semicolon
            text = text.strip()                               # Remove leading and trailing spaces
            text = re.sub(r'\s+', ' ', text)                  # Remove multiple spaces
            text = self.remove_location_unit(text)           # Remove Locations and state    
            text = self.remove_consecutive_duplicates(text)   # Remove consecutive duplicate words
            words = word_tokenize(text)                                     # Tokenize the text
            words = [word for word in words if word not in self.stopwords]  # Remove stopwords
            words = [self.stemmer.stem(word) for word in words]             # Apply stemming (if needed)
            text = ' '.join(words)                                          # Join words back into a single string
            return text
        except Exception as e:
            logging.error(f"Error processing text '{text}': {e}")
            return None


    def preprocess_dataframe(self, df, col_name):
        """
        Apply text preprocessing to a specified column in a DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame containing the text column.
            col_name (str): The name of the column to preprocess.
        Returns: pd.DataFrame: The DataFrame with an additional column of preprocessed text.
        """
        if df.empty:
            raise ValueError("The DataFrame is empty.")
        
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame.")
        
        logging.info(f"Preprocessing column '{col_name}' in DataFrame with {len(df)} rows.")
        
        new_col_name = f'preprocessed_{col_name}'
        df[new_col_name] = df[col_name].apply(self.preprocess_text)
        df[new_col_name] = df[new_col_name].apply(self.remove_emoji)

        print('new_col_name',new_col_name)
        
        
        logging.info(f"Added new column '{new_col_name}' to DataFrame.")
        return df
