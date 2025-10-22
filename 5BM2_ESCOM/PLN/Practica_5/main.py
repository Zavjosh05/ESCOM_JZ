import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt

docs = [
    "El gato duerme en la cama",
    "el perro duerme en el sofá",
    "el gato corre en el jardín"
]

#Calcula manualmente el TF utilizando los métodos de CountVectorizer y DataFrame para mostrar los resultados.
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
tf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print("Matriz TF:\n", tf_df)

#Calcula el TF-iDF de los documentos anteriores, usando tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(docs)
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("\nMatriz TF-IDF:\n", tfidf_df)

docs2 = [
    "la inteligencia artificial aprende de los datos",
    "la inteligencia humana razona con lógica",
    "el aprendizaje automático usa datos y modelos"
]

#Calcula el TF-IDF y muestra los términos más importantes por documento.
tfidf_vectorizer2 = TfidfVectorizer()
X_tfidf2 = tfidf_vectorizer2.fit_transform(docs2)
tfidf_df2 = pd.DataFrame(X_tfidf2.toarray(), columns=tfidf_vectorizer2.get_feature_names_out())
print("\nMatriz TF-IDF para docs2:\n", tfidf_df2)

#Usando matplotlib, representa tf-idf vs las palabras del vocabulario.
plt.figure(figsize=(10,6))
for i in range(X_tfidf2.shape[0]):
    plt.plot(tfidf_vectorizer2.get_feature_names_out(), X_tfidf2.toarray()[i], marker='o', label=f'Documento {i+1}')
plt.title('TF-IDF vs Vocabulario')
plt.xlabel('Palabras del Vocabulario')
plt.ylabel('TF-IDF')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()