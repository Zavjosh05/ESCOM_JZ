#include "Nodo.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

using namespace std;

void Nodo::agregarTransicion(Nodo* destino, const char* simbolo)
{
    //Aumentar arreglo de transiciones
    Nodo **nuevoArrTrans = new Nodo*[numTransiciones + 1];
    char **nuevoArrSimb = new char*[numSimbolos + 1];
    int *nuevoArrLong = new int[numSimbolos + 1];
    int i, len;

    //copiando valoreszzz
    for(i = 0; i < numTransiciones; i++)
        nuevoArrTrans[i] = transiciones[i];

    for(i = 0; i < numSimbolos; i++)
    {
        nuevoArrSimb[i] = simbolos[i];
        nuevoArrLong[i] = longitudSimbolos[i];
    }

    //Agregar nueva transicion
    nuevoArrTrans[numTransiciones] = destino;

    len = strlen(simbolo);
    nuevoArrSimb[numSimbolos] = new char[len + 1];
    strcpy(nuevoArrSimb[numSimbolos],simbolo);
    nuevoArrLong[numSimbolos] = len;

    //liberar memoria anterior
    delete[] transiciones;
    delete[] simbolos;
    delete[] longitudSimbolos;

    //Actualizacion de punteros
    transiciones = nuevoArrTrans;
    simbolos = nuevoArrSimb;
    longitudSimbolos = nuevoArrLong;

    numTransiciones++;
    numSimbolos++;
}

void Nodo::imprimirNodo() const
{
    int i;

    cout << "Estado: " << nombre;
    if (esInicial) cout << " (Inicial)";
    if (esAceptacion) cout << " (Aceptacion)";
    cout << endl;

    for(i = 0; i < numTransiciones; i++)
    cout << "  -" << simbolos[i] << "-> " << transiciones[i]->nombre << endl;
}