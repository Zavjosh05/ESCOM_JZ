#include <iostream>
#include <cstring>
#include "AFD.h"

using namespace std;

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cout << "Uso: " << argv[0] << " <archivo_afd.txt> [archivo_cadena.txt]" << endl;
        cout << "Ejemplo: ./afd termina_en_1.txt cadena.txt\n";
        return 1;
    }

    const char* archivoAFD = argv[1];
    const char* archivoCadenas = (argc >= 3) ? argv[2] : nullptr;

    AFD automata;
    automata.leerDesdeArchivo(archivoAFD);
    automata.imprimirAFD();

    if (archivoCadenas) {
        cout << "\n--- Procesando cadenas desde archivo: " << archivoCadenas << " ---\n";
        automata.leerCadenaDesdeArchivo(archivoCadenas);
        cout << "--------------------------------------------\n";
    }

    while (true) {
        cout << "\nSeleccione una opción:\n";
        cout << "1. Ingresar cadena manualmente\n";
        cout << "2. Leer cadena desde archivo (cadena.txt)\n";
        cout << "3. Volver a imprimir el AFD\n";
        cout << "4. Salir\n";
        cout << "Opción: ";

        int opcion;
        cin >> opcion;

        if (opcion == 1) {
            automata.ingresarCadenaManual();
        }
        else if (opcion == 2) {
            automata.leerCadenaDesdeArchivo("cadena.txt");
        }
        else if (opcion == 3) {
            automata.imprimirAFD();
        }
        else if (opcion == 4) {
            cout << "\nSaliendo del programa...\n";
            break;
        }
        else {
            cout << "Opción no válida. Intente de nuevo.\n";
        }
    }

    return 0;
}
