#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include "AFD.h"

using namespace std;

void AFD::agregarEstado(const char* nombre, int esInicial, int esAceptacion)
{
    Nodo *nuevoArr = new Nodo[numEstados + 1];
    int i,len;

    for(i = 0; i < numEstados; i++)
        nuevoArr[i] = estados[i];

    delete[] estados;
    estados = nuevoArr;

    estados[numEstados].esInicial = esInicial;
    estados[numEstados].esAceptacion = esAceptacion;

    len = strlen(nombre);
    estados[numEstados].nombre = new char[len + 1];
    strcpy(estados[numEstados].nombre, nombre);
    estados[numEstados].numNombre = len;

    numEstados++;
}

void AFD::agregarSimbolo(const char* simbolo)
{
    char **nuevoArr = new char*[numSimbolos + 1];
    int* nuevoNumAlfabeto = new int[numSimbolos + 1];
    int i, len;

    for(i = 0; i < numSimbolos; i++)
    {
        nuevoArr[i] = alfabeto[i];
        nuevoNumAlfabeto[i] = numAlfabeto[i];
    }

    len = strlen(simbolo);
    nuevoArr[numSimbolos] = new char[len + 1];
    strcpy(nuevoArr[numSimbolos],simbolo);
    nuevoNumAlfabeto[numSimbolos] = len;

    delete[] alfabeto;
    delete[] numAlfabeto;

    alfabeto = nuevoArr;
    numAlfabeto = nuevoNumAlfabeto;
    numSimbolos++;
}

void AFD::agregarTransicion(const char* origen, const char* destino, const char* simbolo)
{
    Nodo *nodoOrigen = nullptr;
    Nodo *nodoDestino = nullptr;
    int i;

    for(i = 0; i < numEstados; i++)
    {
        if(strcmp(estados[i].nombre, origen) == 0) nodoOrigen = &estados[i];
        if(strcmp(estados[i].nombre, destino) == 0) nodoDestino = &estados[i];
    }

    if(nodoOrigen && nodoDestino)
        nodoOrigen->agregarTransicion(nodoDestino, simbolo);
}
//corte
int AFD::evaluarCadena(const char* cadena)
{
    Nodo* actual = nullptr;
    for (int i = 0; i < numEstados; ++i) {
        if (estados[i].esInicial) {
            actual = &estados[i];
            break;
        }
    }

    if (!actual) {
        cout << "Error: no hay estado inicial definido." << endl;
        return 0;
    }

    int lenCadena = strlen(cadena);

    for (int i = 0; i < lenCadena; ++i) {
        char simbolo[2] = {cadena[i], '\0'};
        int encontrado = 0;

        for (int j = 0; j < actual->numSimbolos; ++j) {
            if (strcmp(actual->simbolos[j], simbolo) == 0) {
                actual = actual->transiciones[j];
                encontrado = 1;
                break;
            }
        }

        if (!encontrado) {
            cout << "Simbolo no valido: " << simbolo << endl;
            return 0;
        }
    }

    return actual->esAceptacion;
}

void AFD::imprimirAF() const
{
    cout << "\n--- AFD ---" << endl;
    for (int i = 0; i < numEstados; ++i)
        estados[i].imprimirNodo();
}

int AFD::asegurarUnEstadoInicial() const
{
    int contador = 0;
    for (int i = 0; i < numEstados; ++i)
        if (estados[i].esInicial) contador++;
    return contador == 1;
}

int AFD::verficarEsAfd() const
{
    for (int i = 0; i < numEstados; ++i)
        if (estados[i].numSimbolos > numSimbolos)
            return 0;
    return 1;
}

void AFD::leerDeArchivo(const char* nombreArchivo)
{
    ifstream archivo(nombreArchivo);
    if (!archivo) {
        cout << "No se pudo abrir el archivo: " << nombreArchivo << endl;
        return;
    }

    char linea[256];
    archivo.getline(linea, 256);

    char* token = strtok(linea, " ");
    while (token) {
        agregarSimbolo(token);
        token = strtok(nullptr, " ");
    }

    while (archivo.getline(linea, 256)) {
        char indicador[3] = "";
        char estado[50] = "";
        char destino[50] = "";

        int esInicial = 0, esAceptacion = 0;

        char* ptr = strtok(linea, " ");
        if (!ptr) continue;

        if (strcmp(ptr, "->") == 0) {
            esInicial = 1;
            ptr = strtok(nullptr, " ");
        } else if (strcmp(ptr, "*") == 0) {
            esAceptacion = 1;
            ptr = strtok(nullptr, " ");
        }

        strcpy(estado, ptr);
        agregarEstado(estado, esInicial, esAceptacion);

        for (int i = 0; i < numSimbolos; ++i) {
            ptr = strtok(nullptr, " ");
            if (ptr)
                agregarTransicion(estado, ptr, alfabeto[i]);
        }
    }

    archivo.close();
}

void AFD::leerCadenaDeArchivo(const char* nombreArchivo)
{
    ifstream archivo(nombreArchivo);
    if (!archivo) {
        cout << "No se pudo abrir el archivo: " << nombreArchivo << endl;
        return;
    }

    char buffer[256];
    archivo.getline(buffer, 256);
    archivo.close();

    int len = strlen(buffer);
    cadena = new char[len + 1];
    strcpy(cadena, buffer);
    longitudCadena = len;

    cout << "Cadena leída: " << cadena << endl;
    int resultado = evaluarCadena(cadena);
    cout << "Resultado: " << (resultado ? "ACEPTADA" : "RECHAZADA") << endl;
}