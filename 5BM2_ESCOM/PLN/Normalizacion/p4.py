#Zavaleta Guerrero Joshua Ivan
import re
import unicodedata
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def normalizar_texto(texto):
    #1. Pasar a minúsculas
    texto = texto.lower()
    #print("Texto en minusculas: ",texto)

    #2. Eliminar signos de puntuación
    texto = re.sub(r'[^\w\s]', '', texto)
    #print("Texto sin puntuación: ",texto)

    #3. Eliminar acentos
    texto = ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))
    #print("Texto sin acentos: ",texto)

    #4 y 5. Tokenización y eliminación de stopwords con spaCy
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(texto)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    #print("Tokens sin stopwords (spaCy):", tokens)

    #6. Lematización
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    #print("Tokens lematizados (spaCy):", tokens)

    #print("Texto normalizado:", tokens)
    return tokens

#7. Vectorización
Documentos = [
    "El estudiante estudia en la universidad.",
    "Los profesores enseñan programación y matemáticas.",
    "La inteligencia artificial aprende de los datos."
]
vocabulario = []
for i, doc in enumerate(Documentos):
    print(f"\nDocumento {i+1}: {doc}")
    tokens = normalizar_texto(doc)
    vocabulario.extend(tokens)
    print(f"Tokens normalizados del documento {i+1}:", tokens)
print("\nVocabulario de todos los documentos:", vocabulario)
# vectorizer = CountVectorizer(binary=True)
mat = np.zeros((len(Documentos), len(vocabulario)), dtype=int)
#print(mat)
for i, doc in enumerate(Documentos):
    tokens = normalizar_texto(doc)
    for token in tokens:
        if token in vocabulario:
            j = vocabulario.index(token)
            mat[i][j] = 1

print("\nMatriz binaria:\n", mat)
# matriz_binaria = vectorizer.fit_transform(vocabulario)
# print("Vocabulario único:", vectorizer.get_feature_names_out())
# print("\nMatriz binaria:\n", matriz_binaria.toarray())