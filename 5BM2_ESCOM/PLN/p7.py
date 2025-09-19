a = input().lower()

if a == a[::-1]:
	print("La palabra es un palindromo")
else:
	print("La palabra no es un palindromo")