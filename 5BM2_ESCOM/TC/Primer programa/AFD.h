#ifndef AFD_H
#define AFD_H

#include "Nodo.h"

class AFD
{
    public:
        AFD() = default;
        ~AFD() = default;

        Nodo *estados = nullptr; // arreglo de nodos (estados)
        int numEstados = 0; // numero de estados (tamaño del arreglo)

        char **alfabeto = nullptr; // arreglo de simbolos (alfabeto)
        int *numAlfabeto = nullptr; // numero de simbolos y tamaño de cada arreglo de simbolos
        int numSimbolos = 0; // cantidad de simbolos en el alfabeto

        char *cadena = nullptr; // cadena a evaluar
        int longitudCadena = 0; // longitud de la cadena

        void agregarEstado(const char* nombre, int esInicial, int esAceptacion);
        void agregarSimbolo(const char* simbolo);
        void agregarTransicion(const char* estadoOrigen, const char* estadoDestino, const char* simbolo);
        int evaluarCadena(const char* cadena);
        void imprimirAF() const;
        int asegurarUnEstadoInicial() const;
        int verficarEsAfd() const;
        void leerDeArchivo(const char* nombreArchivo);
        void leerCadenaDeArchivo(const char* nombreArchivo);
};

#endif // AF_H