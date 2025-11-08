import nltk
import re
import unicodedata
from nltk import bigrams, FreqDist
from nltk.tokenize import word_tokenize
import random
import pandas as pd

#nltk.download('punkt')

text = """El sol se escondía lentamente detrás de las montañas cuando el viajero llegó al pequeño pueblo.
Las casas, construidas con piedra y madera, reflejaban la luz dorada del atardecer.
En la plaza central, los niños jugaban mientras los ancianos conversaban bajo los árboles.
El viajero se detuvo frente a una fuente antigua y observó su reflejo en el agua tranquila.
Había recorrido muchos caminos, cruzado ríos y desiertos, pero aún no había encontrado lo que buscaba.
Llevaba un mapa gastado y una brújula que apenas funcionaba.
Una mujer se acercó con una sonrisa amable y le ofreció un poco de pan.
Él agradeció el gesto y le preguntó por el camino hacia el norte.
La mujer señaló la carretera que cruzaba el valle y desaparecía entre las colinas.
El viajero respiró profundamente y siguió su marcha.
A medida que avanzaba, la noche cubría el cielo con un manto de estrellas.
Cada paso lo acercaba a su destino, aunque todavía no sabía cuál era.
El viento soplaba suave, trayendo consigo el aroma del bosque y el murmullo de los insectos.
En ese silencio, comprendió que el viaje era más importante que la llegada."""

#instrucciones:
# Convertir el texto a minúsculas, eliminar signos de puntuacion.
# Tokenizarlo (dividir en palabras).

text = text.lower()
text = re.sub(r'[^\w\s]', '', text)
text = ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))
tokens = word_tokenize(text.lower(), language='spanish')
 
#Generacion de bigramas y calculo de probabilidades condicionales
bigrams_list = list(bigrams(tokens))
bigram_freq = FreqDist(bigrams_list)
word_freq = FreqDist(tokens)
# print("Frecuencia de palabras:", word_freq.items())
# print("Frecuencia de bigramas:", bigram_freq.items())
total_bigrams = sum(bigram_freq.values())
#print(bigram_freq.items())
#Formula de probabilidad condicional P(B|A) = P(A,B) / P(A)
bigram_prob = {}
for (w1, w2), freq in bigram_freq.items():
    print(f"Bigrama: ({w1}, {w2}) - Frecuencia: {freq}, frecuencia de '{w1}': {word_freq[w1]}")
    prob = freq / word_freq[w1]
    bigram_prob[(w1, w2)] = prob


#impresion de bigramas y sus probabilidades
print("Bigramas y sus probabilidades condicionales:")
for bg, prob in bigram_prob.items():
    print(f"{bg}: {prob:.4f}")

# 2) Predicción de la siguiente palabra. Implementa una función "def predecir_siguiente(palabra_actual)". Debe recibir una palabra y predecir cuál es la siguiente.
# Ejemplo de uso: 
# print(predecir_siguiente("el"))
# print(predecir_siguiente("principito"))

#la funcion para predecir la siguiente palabra debe de elegir aleatoriamente entre las posibles opciones, ponderando por la probabilidad condicional calculada anteriormente.
def predecir_siguiente(palabra_actual):
    posibles_siguientes = {w2: prob for (w1, w2), prob in bigram_prob.items() if w1 == palabra_actual}
    if not posibles_siguientes:
        return None
    palabras = list(posibles_siguientes.keys())
    probabilidades = list(posibles_siguientes.values())
    siguiente_palabra = random.choices(palabras, weights=probabilidades, k=1)[0]
    return siguiente_palabra

print(predecir_siguiente("el"))
print("\n")

# 3) Predecir texto implementando la función "def generar_texto(palabra_inicial, longitud=15)". Debe recibir una palabra y una longitud X para generar un texto de X cantidad de palabras a partir de la palabra inicial.
# Ejemplo de uso: 
# print(generar_texto("el", 20))

def generar_texto(palabra_inicial, longitud):
    texto_generado = [palabra_inicial]
    palabra_actual = palabra_inicial
    for _ in range(longitud - 1):
        siguiente_palabra = predecir_siguiente(palabra_actual)
        if siguiente_palabra is None:
            break
        texto_generado.append(siguiente_palabra)
        palabra_actual = siguiente_palabra
    return ' '.join(texto_generado)

print(generar_texto("el", 20))
