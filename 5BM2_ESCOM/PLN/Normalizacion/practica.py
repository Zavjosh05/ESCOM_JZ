import re
import unicodedata
import spacy
from sklearn.feature_extraction.text import CountVectorizer

#Texto de ejemplo
texto = "!Hola¡ Me gustan los PROGRAMAS de Inteligencia Artificial."

#1. Pasar a minúsculas
texto = texto.lower()
print("Texto en minusculas: ",texto)

#2. Eliminar signos de puntuación
texto = re.sub(r'[^\w\s]', '', texto)
print("Texto sin puntuación: ",texto)

#3. Eliminar acentos
texto = ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))
print("Texto sin acentos: ",texto)

#4 y 5. Tokenización y eliminación de stopwords con spaCy
nlp = spacy.load("es_core_news_sm")
doc = nlp(texto)
tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
print("Tokens sin stopwords (spaCy):", tokens)

#6. Lematización
tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
print("Tokens lematizados (spaCy):", tokens)

print("Texto normalizado:", tokens)