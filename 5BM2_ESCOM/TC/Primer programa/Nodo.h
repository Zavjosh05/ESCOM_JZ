#ifndef NODO_H
#define NODO_H

class Nodo
{
    public:
        Nodo() = default;
        ~Nodo() = default;

        int esInicial = 0; // 1 si es estado inicial, 0 en otro caso
        int esAceptacion = 0; // 1 si es estado de aceptacion, 0 en otro caso
        char *nombre = nullptr; // nombre del nodo
        int numNombre = 0; // tamaño del nombre

        Nodo **transiciones = nullptr; // arreglo de punteros a nodos (transiciones)
        int numTransiciones = 0; // numero de transiciones (tamaño del arreglo)

        char **simbolos = nullptr; // arreglo de simbolos
        int numSimbolos = 0; // numero de simbolos (tamaño del arreglo)
        int *longitudSimbolos = nullptr; //arreglo con longitudes de cada simbolo

        void agregarTransicion(Nodo* destino, const char* simbolo);
        void imprimirNodo() const;
};

#endif // NODO_H