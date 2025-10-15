#ifndef AFD_H
#define AFD_H

#include "Nodo.h"
#include <fstream>

using namespace std;

class AFD
{
    private:
        Nodo **estados = nullptr; // arreglo dinámico de punteros a nodos
        int numEstados = 0;       // número de estados
        char **alfabeto = nullptr; // arreglo de símbolos del alfabeto
        int numAlfabeto = 0;      // número de símbolos
        int *longitudAlfabeto = nullptr; // longitud de cada símbolo

        Nodo *estadoInicial = nullptr; // puntero al estado inicial

    public:
        AFD() = default;
        ~AFD();

        void leerDesdeArchivo(const char *nombreArchivo);
        Nodo* buscarEstado(const char *nombre);
        void agregarEstado(const char *nombre, int esInicial, int esAceptacion);
        void agregarTransicion(const char *origen, const char *destino, const char *simbolo);
        void imprimirAFD() const;

        void leerCadenaDesdeArchivo(const char *nombreArchivo);
        void ingresarCadenaManual();
        int verificarCadenaValida(const char *cadena);
        int evaluarCadena(const char *cadena);
};

#endif // AFD_H
