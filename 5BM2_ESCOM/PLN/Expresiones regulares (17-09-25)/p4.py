import re

patron = re.compile(",")
resultado = patron.findall("Cadena1, Cadena2, Cadena3, Cadena4, Cadena5")
print(resultado)
resultado2 = patron.split("Cadena1, Cadena2, Cadena3, Cadena4, Cadena5")
print(resultado2)