#include <iostream>
#include "AFD.h"
using namespace std;

int main()
{
    AFD automata;

    // Leer el AFD desde un archivo
    cout << "Leyendo AFD desde archivo 'afd.txt'..." << endl;
    automata.leerDeArchivo("afd.txt");
    automata.imprimirAF();

    // Leer una cadena desde archivo y evaluarla
    cout << "\nLeyendo cadena desde archivo 'cadena.txt'..." << endl;
    automata.leerCadenaDeArchivo("cadena.txt");

    return 0;
}
