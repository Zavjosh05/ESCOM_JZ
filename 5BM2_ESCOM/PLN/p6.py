print("Ingrese un texto")
a = input().lower()
vocales = "aeiou"
b = sum(1 for letra in a if letra in vocales)



print(f"la cadena {a} esta en minusculas\ny tiene {b} vocales")