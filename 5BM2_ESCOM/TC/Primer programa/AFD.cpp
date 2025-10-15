#include "AFD.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

using namespace std;

AFD::~AFD()
{
    for(int i=0;i<numEstados;i++){
        delete[] estados[i]->nombre;
        for(int j=0;j<estados[i]->numSimbolos;j++)
            delete[] estados[i]->simbolos[j];
        delete[] estados[i]->simbolos;
        delete[] estados[i]->longitudSimbolos;
        delete[] estados[i]->transiciones;
        delete estados[i];
    }
    delete[] estados;

    for(int i=0;i<numAlfabeto;i++)
        delete[] alfabeto[i];
    delete[] alfabeto;
    delete[] longitudAlfabeto;
}

Nodo* AFD::buscarEstado(const char *nombre)
{
    for(int i=0;i<numEstados;i++)
        if(strcmp(estados[i]->nombre, nombre) == 0)
            return estados[i];
    return nullptr;
}

void AFD::agregarEstado(const char *nombre, int esInicial, int esAceptacion)
{
    Nodo **nuevoArr = new Nodo*[numEstados + 1];
    for(int i=0;i<numEstados;i++)
        nuevoArr[i] = estados[i];

    Nodo *nuevo = new Nodo();
    int len = strlen(nombre);
    nuevo->nombre = new char[len + 1];
    strcpy(nuevo->nombre, nombre);
    nuevo->numNombre = len;
    nuevo->esInicial = esInicial;
    nuevo->esAceptacion = esAceptacion;

    nuevoArr[numEstados] = nuevo;
    delete[] estados;
    estados = nuevoArr;
    numEstados++;

    if(esInicial) estadoInicial = nuevo;
}


void AFD::agregarTransicion(const char *origen, const char *destino, const char *simbolo)
{
    Nodo *nodoOrigen = buscarEstado(origen);
    Nodo *nodoDestino = buscarEstado(destino);

    if(nodoOrigen && nodoDestino)
        nodoOrigen->agregarTransicion(nodoDestino, simbolo);
}


void AFD::leerDesdeArchivo(const char *nombreArchivo)
{
    ifstream archivo(nombreArchivo);
    if(!archivo.is_open()){
        cout << "Error: no se pudo abrir " << nombreArchivo << endl;
        return;
    }

    char linea[256];

    // ======== Leer lista de estados ========
    archivo.getline(linea, sizeof(linea));
    if(strncmp(linea, "Estados:", 8) == 0){
        char *ptr = strchr(linea, ':');
        ptr++;
        char *token = strtok(ptr, ", \t\r\n");
        while(token){
            agregarEstado(token, 0, 0);
            token = strtok(nullptr, ", \t\r\n");
        }
    }

    // ======== Leer alfabeto ========
    archivo.getline(linea, sizeof(linea));
    if(strncmp(linea, "Alfabeto:", 9) == 0){
        char *ptr = strchr(linea, ':');
        ptr++;
        char *token = strtok(ptr, ", \t\r\n");
        while(token){
            numAlfabeto++;
            alfabeto = (char**)realloc(alfabeto, numAlfabeto * sizeof(char*));
            longitudAlfabeto = (int*)realloc(longitudAlfabeto, numAlfabeto * sizeof(int));

            int len = strlen(token);
            alfabeto[numAlfabeto-1] = new char[len+1];
            strcpy(alfabeto[numAlfabeto-1], token);
            longitudAlfabeto[numAlfabeto-1] = len;

            token = strtok(nullptr, ", \t\r\n");
        }
    }

    // ======== Leer estado inicial ========
    archivo.getline(linea, sizeof(linea));
    if(strncmp(linea, "Inicial:", 8) == 0){
        char *ptr = strchr(linea, ':');
        ptr++;
        char *token = strtok(ptr, ", \t\r\n");
        if(token){
            Nodo *est = buscarEstado(token);
            if(est){
                est->esInicial = 1;
                estadoInicial = est;
            }
        }
    }

    // ======== Leer estados de aceptación ========
    archivo.getline(linea, sizeof(linea));
    if(strncmp(linea, "Finales:", 8) == 0){
        char *ptr = strchr(linea, ':');
        ptr++;
        char *token = strtok(ptr, ", \t\r\n");
        while(token){
            Nodo *est = buscarEstado(token);
            if(est) est->esAceptacion = 1;
            token = strtok(nullptr, ", \t\r\n");
        }
    }

    // ======== Leer línea vacía + encabezado Transiciones ========
    archivo.getline(linea, sizeof(linea)); // línea vacía
    archivo.getline(linea, sizeof(linea)); // "Transiciones:"

    // ======== Leer transiciones ========
    while(archivo.getline(linea, sizeof(linea))){
        if(strlen(linea) == 0) continue;
        char *origen = strtok(linea, " \t\r\n");
        char *simbolo = strtok(nullptr, " \t\r\n");
        char *destino = strtok(nullptr, " \t\r\n");
        if(origen && simbolo && destino)
            agregarTransicion(origen, destino, simbolo);
    }

    archivo.close();
}

void AFD::imprimirAFD() const
{
    cout << "\n--- AFD ---" << endl;
    for(int i=0;i<numEstados;i++)
        estados[i]->imprimirNodo();
}

int AFD::verificarCadenaValida(const char *cadena)
{
    for(int i=0; cadena[i]!='\0'; i++){
        int valido = 0;
        char simb[2] = {cadena[i], '\0'};
        for(int j=0;j<numAlfabeto;j++){
            if(strcmp(simb, alfabeto[j]) == 0){
                valido = 1;
                break;
            }
        }
        if(!valido) return 0;
    }
    return 1;
}

// ----------------------
// Evaluar cadena
// ----------------------
int AFD::evaluarCadena(const char *cadena)
{
    if(!estadoInicial){
        cout << "Error: No hay estado inicial definido." << endl;
        return 0;
    }

    Nodo *actual = estadoInicial;
    for(int i=0; cadena[i]!='\0'; i++){
        char simb[2] = {cadena[i], '\0'};
        Nodo *sig = nullptr;
        for(int j=0;j<actual->numSimbolos;j++){
            if(strcmp(actual->simbolos[j], simb) == 0){
                sig = actual->transiciones[j];
                break;
            }
        }
        if(!sig) return 0;
        actual = sig;
    }

    return actual->esAceptacion;
}

void AFD::leerCadenaDesdeArchivo(const char *nombreArchivo)
{
    ifstream archivo(nombreArchivo);
    if(!archivo.is_open()){
        cout << "Error: No se pudo abrir " << nombreArchivo << endl;
        return;
    }

    char buffer[256];
    archivo.getline(buffer, sizeof(buffer));
    archivo.close();

    if(!verificarCadenaValida(buffer)){
        cout << "Cadena invalida (simbolos fuera del alfabeto)." << endl;
        return;
    }

    cout << "Cadena: " << buffer << endl;
    cout << "Resultado: " << (evaluarCadena(buffer) ? "ACEPTADA" : "RECHAZADA") << endl;
}

void AFD::ingresarCadenaManual()
{
    char buffer[256];
    cout << "Ingrese una cadena: ";
    cin >> buffer;

    if(!verificarCadenaValida(buffer)){
        cout << "Cadena invalida (simbolos fuera del alfabeto)." << endl;
        return;
    }

    cout << "Resultado: " << (evaluarCadena(buffer) ? "ACEPTADA" : "RECHAZADA") << endl;
}
