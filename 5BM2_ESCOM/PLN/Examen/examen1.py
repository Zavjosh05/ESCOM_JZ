#Examen 1 - Procesamiento de Texto
#Nombre: [Joshua Ivan Zavaleta Guerrero]

import re
import unicodedata
import spacy
from sklearn.feature_extraction.text import CountVectorizer

# Objetivo: 
# Evaluar la capacidad del alumno para:
# Aplicar técnicas de normalización y limpieza de texto (PLN).
# Usar expresiones regulares (re) para extraer y validar información.
# Implementar representaciones vectoriales binarias de documentos.
# Interpretar los resultados de la vectorización.


# Parte 1: Limpieza con expresiones regulares (2 puntos)
# Extrae todos los correos electrónicos.
# Extrae todos los números telefónicos.
# Extrae todas las fechas (en formato dd/mm/yyyy).

def extraer_correos(texto):
    patron = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(patron, texto)

def extraer_telefonos(texto):
    patron = r'\+\d{1,2}-\d{1,2}-\d{1,4}-\d{1,4}'
    return re.findall(patron, texto)

def extraer_fechas(texto):
    patron = r'\b\d{2}/\d{2}/\d{4}\b'
    return re.findall(patron, texto)

def parte1(textos):
    print("=== PARTE 1: Extracción con Expresiones Regulares ===")
    correos = []
    telefonos = []
    fechas = []
    for texto in textos:
        correos.extend(extraer_correos(texto))
        telefonos.extend(extraer_telefonos(texto))
        fechas.extend(extraer_fechas(texto))

    print("Correos electrónicos encontrados:", correos)
    print("Números telefónicos encontrados:", telefonos)
    print("Fechas encontradas:", fechas)

    return correos, telefonos, fechas

# Parte 2: Normalización con spaCy (2 puntos)
# Convierte los textos a minúsculas.
# Elimina acentos, stopwords y puntuación.
# Lematiza las palabras (en español).
# Muestra los textos normalizados.

def limpiar_textos(textos, correos, telefonos, fechas):
    textos_limpios = []
    for texto in textos:
        for correo in correos:
            texto = texto.replace(correo, '')
        for telefono in telefonos:
            texto = texto.replace(telefono, '')
        for fecha in fechas:
            texto = texto.replace(fecha, '')
        textos_limpios.append(texto)
    return textos_limpios
    

def normalizar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\D\s]', ' ', texto)
    texto = ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(texto)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return tokens

def parte2(textos, correos, telefonos, fechas):
    print("\n=== PARTE 2: Normalización con spaCy ===")
    textos = limpiar_textos(textos, correos, telefonos, fechas)
    #print("Textos después de eliminar correos, teléfonos y fechas:", textos)
    textos_normalizados = []
    for texto in textos:
        tokens = normalizar_texto(texto)
        textos_normalizados.append(' '.join(tokens))
        print("Texto original:", texto)
        print("Texto normalizado:", ' '.join(tokens))
    return textos_normalizados

# Parte 3: Representación vectorial (4 puntos)
# Genera la matriz binarizada (Bag of Words) de los textos normalizados.
# Muestra el vocabulario generado.
# Imprime la matriz binaria resultante.

def representacion_vectorial(textos_normalizados):
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(textos_normalizados)
    return vectorizer.get_feature_names_out(), X.toarray()

def parte3(textos_normalizados):
    print("\n=== PARTE 3: Representación Vectorial ===")
    vocabulario, matriz = representacion_vectorial(textos_normalizados)
    print("Vocabulario generado:", vocabulario)
    print("Matriz resultante:\n", matriz)

textos = [
    "El correo del estudiante es alumno123@universidad.mx y su teléfono es +52-55-1234-5678.",
    "El profesor imparte Programación y su correo es profe.python@escuela.edu.mx",
    "La clase de Inteligencia Artificial será el 16/09/2025 en el laboratorio 2."
]

correos, telefonos, fechas = parte1(textos)

textos_normalizados = parte2(textos,correos, telefonos, fechas)

parte3(textos_normalizados)