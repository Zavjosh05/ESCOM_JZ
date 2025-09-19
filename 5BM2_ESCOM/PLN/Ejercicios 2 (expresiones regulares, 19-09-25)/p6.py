import re

text = "Ese producto es malo y feo"

print(re.sub("malo|feo","***",text))