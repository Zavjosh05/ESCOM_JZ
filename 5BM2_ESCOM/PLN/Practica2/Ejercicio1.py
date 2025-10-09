import re

text = "Ana_123, Pedro!, @Luis45, Maria99##, #Sofía, Ñoño"

pattern = r'[A-Za-záéíóúÁÉÍÓÚñÑ]+'

print(re.findall(pattern, text))