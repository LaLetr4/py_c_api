#include <unistd.h>
#include "pyEmbedding.h"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"

using namespace std;

    unsigned pyEmbedding::counter = 0;
    bool pyEmbedding::verbose = 0;
    PyObject * pyEmbedding::pModule, * pyEmbedding::pDict, * pyEmbedding::pClass;

    void pyEmbedding::FatalError(const /*string*/ char * diag, int code/* = -1*/) {
        cerr<<" Error: "<<diag<<endl;
        PyErr_Print();
        cerr<<" Exiting..."<<endl;
        exit(code);
    }

    void pyEmbedding::setPythonPath(const char * _path){
        const char * path="PYTHONPATH";
        setenv(path,_path,0); // doesn't overwrite
        cout<<"Set "<<path<<" = "<<getenv(path)<<endl; 
    }
    
    void pyEmbedding::printEmbInfo(){
        cout<<"Module name is "<<modName<<endl;
        cout<<"Class name is "<<dictName<<endl;
    }

    pyEmbedding::pyEmbedding(string settings_file = "conf.ini"){
        if(!counter) InitPy(settings_file);
        counter++;
        // создать экземпляр класса
        if(PyCallable_Check(pClass)) {
            pInstance = PyObject_CallObject(pClass, NULL);
        } else {
            Py_Finalize();
            FatalError("with python class creation");
        }
    }
    
    void pyEmbedding::InitPy(string settings_file){
        ConfigParser my_cfg(settings_file);
        const char * pytPath = (my_cfg.GetString("pythonPath")).c_str();
        setPythonPath(pytPath);
        Py_Initialize(); // инициализируем Python C Api
        import_array(); // инициализируем NumPy C Api
        modName = (my_cfg.GetString("moduleName")).c_str();
            
        PyObject * pName = PyUnicode_FromString(modName);
        pModule = PyImport_Import(pName);
        Py_DECREF(pName);
        if(pModule == NULL) {
            FatalError(strcat("Bad module ",modName));
        }
        pDict = PyModule_GetDict(pModule);
        if(pDict == NULL) {
        Py_DECREF(pModule);
        FatalError(strcat("Bad dictionary in module ", modName));
        }
            dictName = (my_cfg.GetString("className")).c_str();
        pClass = PyDict_GetItemString(pDict, dictName);
        if(pClass == NULL) {
        Py_DECREF(pDict);
        Py_DECREF(pModule);
        FatalError(strcat("Wrong class ", dictName));
        }
        if(verbose > 0) cerr<<" Module ``"<<modName<<"'' is succesfully loaded"<<endl;
    }
    
    void pyEmbedding::FinalizePy() {
        Py_DECREF(pDict);
        Py_DECREF(pModule);
        Py_Finalize();
    }
    
    pyEmbedding::~pyEmbedding() {
        Py_DECREF(pInstance);
        counter--;
        if(!counter) FinalizePy();
    }
    

    
