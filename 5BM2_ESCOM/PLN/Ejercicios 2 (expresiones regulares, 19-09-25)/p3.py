import re

text = "Hoy es un gran día"

patron = "día$"

if re.search(patron,text):
    print("La cadena insertada:\n",text,"\nTermina con la palabra día")
else:
    print("La cadena insertada no cumple con el patron")