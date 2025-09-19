import re
patron = re.compile("\d+\.?\d+")
resultado = patron.findall("Esta es una cadena con los n[umeros 14, 15.5 y 0.25")
print(resultado)