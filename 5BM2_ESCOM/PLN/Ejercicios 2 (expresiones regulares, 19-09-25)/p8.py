import re

text = "Hoy es 16/09/2025 y mañana será 17/09/2025"
patter = r'\d{2}/\d{2}/\d{2}'

print(re.findall(patter,text))