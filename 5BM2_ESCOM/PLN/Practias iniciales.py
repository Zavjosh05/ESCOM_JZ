print("HOla mundo")

x = 1
y = 4
print(x+y,x-y,x*y,x/y,x%y,x//y)

x = 6

if x%2:
	print(f"{x} el numero es impar")
else:
	print(f"{x} el numero es par")
    
print("El numero al aplicar la op factorial")
a = input()
res = 1

for x in range(1,int(a)+1):
	res *= x;

print(f"{res} es el factorial de {a}")

print("Ingresa una cadena")
a = input()

print(a[::-1])

print("Ingrese un texto")
a = input().lower()
vocales = "aeiou"
b = sum(1 for letra in a if letra in vocales)

print(f"la cadena {a} esta en minusculas\ny tiene {b} vocales")

a = input().lower()

if a == a[::-1]:
	print("La palabra es un palindromo")
else:
	print("La palabra no es un palindromo")

import re
res = re.search("af","abcdef")
print(res)