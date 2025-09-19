import re

text = "Aprender Python es divertido"

res = re.search("Python",text)

print(res)
print(res.group())