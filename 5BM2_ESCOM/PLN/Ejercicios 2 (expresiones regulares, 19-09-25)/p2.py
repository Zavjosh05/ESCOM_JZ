import re

cadena = "Adios, bienvenido"

if re.match("^Hola",cadena):
    print(cadena,", Empieza con Hola")
else:
    print("La cadena insertada no empieza con el patron indicado")