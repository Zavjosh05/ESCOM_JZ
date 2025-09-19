print("El numero al aplicar la op factorial")
a = input()
res = 1

for x in range(1,int(a)+1):
	res *= x;

print(f"{res} es el factorial de {a}")