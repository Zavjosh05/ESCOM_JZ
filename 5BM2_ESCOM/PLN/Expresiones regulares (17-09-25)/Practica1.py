import re

def extraerEmail(cadena):
    patron = re.compile("Email:\s*([\w\.-]+@[\w\.-]+\.\w+)")
    return patron.findall(cadena)

def extraerNombre(cadena):
    patron = re.compile("Nombre:\s*([\w]+\s[\w]+)")
    return patron.findall(cadena)

def extraerNumero(cadena):
    patron = re.compile("Tel:\s*(\+52\-\d{2}\-\d{4}\-\d{4})")
    return patron.findall(cadena)

def proceso(registros):
    print("Nombre\tEmail\tTelefono\n---------------------------")
    for x in registros:
        nombre = extraerNombre(x)
        email = extraerEmail(x)
        tel = extraerNumero(x)
        if(len(nombre) == 0 or len(email) == 0 or len(tel) == 0):
            continue
        print(nombre[0],"\t",email[0],"\t",tel[0])

data = "Nombre: Juan Pérez, Email: juan.perezexample.com, Tel: +52-55-1234-5678;Nombre: María López, Email: maria.lopez@uni.edu.mx, Tel: +52-81-9876-5432;Nombre: Pedro Ramírez, Email: pedro99@gmail.com, Tel: +52-33-5555-6666"

data2 = re.split(";",data)

proceso(data2)


