import re

text = "Tengo 2 perros, 3 gatos y 1 tortuga"
pattern = re.compile("\d+")

res = pattern.findall(text)

print(res)
