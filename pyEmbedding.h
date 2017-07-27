
#ifndef pyEmbedding_h
#define pyEmbedding_h

#include <iostream>
#include "ConfigParser.h"
#include <string>
#include <cstring>
#include <Python.h>

using namespace std;

class pyEmbedding{
public:
    //в базовом виде - для обёртывания чего угодно (из указанного в концигах)
    //переопределить в классах-наследниках.
    pyEmbedding(string);
    
    ~pyEmbedding();
    
protected:
    
    const char * modName;
    const char * dictName;
    
    static PyObject * pModule, * pDict, * pClass;
    PyObject * pInstance;
    static unsigned counter;
    static bool verbose; //! test information on/off for debagging
    
    PyObject * Instance() { return pInstance; }
    void setPythonPath(const char * _path);
    void InitPy(string);
    void FinalizePy();
    
      
    //Handles connection errors
    static void FatalError(const char * diag, int code = -1);

    //for testing
    void printEmbInfo();

};


#endif
