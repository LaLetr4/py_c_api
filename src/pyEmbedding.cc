
#include <iostream>
#include <cstring>
#include <unistd.h>
#include "pyEmbedding.h"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"

using std::cout;
using std::cerr;
using std::endl;

pyEnv & pyEnv::get() {
  static pyEnv _env;
  return _env;
}
std::string pyEnv::PYTHONPATH = ".";
pyEnv::pyEnv(): verbose(true) {
  const char * path = "PYTHONPATH";
  setenv(path, PYTHONPATH.c_str(), 0);
  cout<<"Set "<<path<<" = "<<getenv(path)<<endl;
  Py_Initialize(); // инициализируем Python C Api
  import_array(); // инициализируем NumPy C Api
  if (verbose)
    cerr<<"pyEnv::pyEnv()"<<endl;
}
pyEnv::~pyEnv() {
  if (verbose) {
    cerr<<"pyEnv::~pyEnv()"<<endl;
    cerr<<"Cleaning instances..."<<endl;
  }
  while(!gInstances.empty()) {
    pyInstance * ref = gInstances.back();
    delete ref;
  }
  if (verbose){    
    cerr<<"Cleaning classes..."<<endl;
  }
  while(!gClasses.empty()) {
    pyClass * ref = gClasses.rbegin()->second;
    delete ref;
  }
  if (verbose){
    cerr<<"Cleaning modules..."<<endl;
  }
  while(!gModules.empty()) {
    pyModule * ref = gModules.rbegin()->second;
    delete ref;
  }
  if (verbose){
    cerr<<"Finalizing python interpreter..."<<endl;
  }
  Py_Finalize();
  if (verbose)
    cerr<<"Python interpreter finalized..."<<endl;
}
void pyEnv::fatalError(const char * diag, const char * spec, int code/* = -1*/) {
  cerr<<" Error: "<<diag;
  if(spec) cerr<<" "<<spec;
  cerr<<endl;
  PyErr_Print();
  cerr<<" Exiting..."<<endl;
  exit(code);
}

pyArgSetter::pyArgSetter(const char * script) {
  // first entry should refer to the script file to be executed
  // if there isn’t a script that will be run, it can be an empty string
  addArg(script);
}
pyArgSetter::~pyArgSetter() { }
void pyArgSetter::printArgs() {
  std::cerr<<"pyArgSetter::printArgs:"<<endl;
  for(unsigned i = 0; i < args.size(); i++)
    std::cerr<<"  args["<<i<<"] = \""<<args[i]<<"\""<<endl;
}
void pyArgSetter::addArg(const char * arg) {
  args.push_back(arg);
}
void pyArgSetter::set() {
  pyEnv::get();
  PySys_SetArgv(args.size(), const_cast<char**>(args.data()));
//   PyRun_SimpleString("import sys");
//   PyRun_SimpleString("print sys.argv");
  args.resize(1);
}
void pyArgSetter::setArgs(const char * _args) {
  char * str = new char[strlen(_args)];
  strcpy(str, _args);
  char * pch = strtok(str, " ");
  while (pch) {
    args.push_back(pch);
    pch = strtok(0, " ");
  }
  set();
  delete [] str;
}

pyModule * pyModule::import(const char * modName) {
  auto & gModules = pyEnv::get().gModules;
  if(!gModules.count(modName))
    gModules[modName] = new pyModule(modName);
  return gModules[modName];
}
pyModule::pyModule(const char * modName): pyNamed(modName) {
  PyObject * pName = PyUnicode_FromString(modName);
  _module = PyImport_Import(pName);
  Py_DECREF(pName);
  if(_module == NULL) {
    pyEnv::get().fatalError("Bad module", modName);
  };
  _dict = PyModule_GetDict(_module);
  if(_dict == NULL) {
    pyEnv::get().fatalError("Bad dictionary in module", modName);
  }
}
pyModule::~pyModule() {
  if(pyEnv::get().verbose)
    cerr<<"pyModule::~pyModule(): "<<name<<endl;
  auto & gModules = pyEnv::get().gModules;
  if(!gModules.count(name))
    pyEnv::get().fatalError("No module in table named", name.c_str());
  gModules.erase(gModules.find(name));
  Py_DECREF(_dict);
  Py_DECREF(_module);
}
pyClass * pyModule::getClass(const char * className) {
  auto & gClasses = pyEnv::get().gClasses;
  if(!gClasses.count(className)) {
    gClasses[className] = new pyClass(className, _dict);
  }
  return gClasses[className];
}

pyClass::pyClass(const char * className, PyObject * pDict): pyNamed(className) {
  _class = PyDict_GetItemString(pDict, className);
  if(_class == NULL) {
    pyEnv::get().fatalError("Wrong class", className);
  }
  if(!PyCallable_Check(_class)) {
    pyEnv::get().fatalError("Not a callable python class", className);
  }
}
pyClass * pyClass::importFrom(const char * className, const char * modName) {
  return pyModule::import(modName)->getClass(className);
}
pyInstance * pyClass::instance() {
  PyObject * pInstance = PyObject_CallObject(_class, NULL);
  return new pyInstance(pInstance, name);
}
pyClass::~pyClass() {
  if(pyEnv::get().verbose)
    cerr<<"pyClass::~pyClass(): "<<name<<endl;
  auto & gClasses = pyEnv::get().gClasses;
  if(!gClasses.count(name))
    pyEnv::get().fatalError("No module in table named", name.c_str());
  gClasses.erase(gClasses.find(name));
  Py_DECREF(_class);
}

pyInstance::pyInstance(PyObject * pInstance, std::string _name): pyNamed(_name) {
  _instance = pInstance;
  pyEnv::get().gInstances.push_back(this);
}
pyInstance * pyInstance::instanceOf(const char * className, const char * modName) {
  return pyModule::import(modName)->getClass(className)->instance();
}
pyInstance::~pyInstance() {
  if(pyEnv::get().verbose)
    cerr<<"pyInstance::~pyInstance(): "<<name<<endl;
  auto & gInstances = pyEnv::get().gInstances;
  for(auto it = gInstances.begin(); it != gInstances.end(); it++)
    if(*it == this) {
      gInstances.erase(it);
      break;
    }
  Py_DECREF(_instance);
}
void pyInstance::call(const char * methodName) {
  PyObject * pValue = PyObject_CallMethod(_instance, const_cast<char*>(methodName), "()");
  if(checkCall(pValue, methodName))
    Py_DECREF(pValue);
}
void pyInstance::call(const char * methodName, int i) {
  PyObject * pValue = PyObject_CallMethod(_instance, const_cast<char*>(methodName), "(i)", i);
  if(checkCall(pValue, methodName))
    Py_DECREF(pValue);
}
void pyInstance::call(const char * methodName, PyObject * pArg) {
  PyObject * pName = PyUnicode_FromString(methodName);
  PyObject * pValue = PyObject_CallMethodObjArgs(_instance, pName, pArg, NULL);
  Py_DECREF(pName);
  if(checkCall(pValue, methodName))
    Py_DECREF(pValue);
}
void pyInstance::call(const char * methodName, const char * s) {
  PyObject * pValue = PyObject_CallMethod(_instance, const_cast<char*>(methodName), "(s)", s);
  if(checkCall(pValue, methodName))
    Py_DECREF(pValue);
}
pyInstance * pyInstance::get(const char * methodName) {
  PyObject * pValue = PyObject_CallMethod(_instance, const_cast<char*>(methodName), "()");
  if(checkCall(pValue, methodName))
    return new pyInstance(pValue, methodName);
  return 0;
}
bool pyInstance::checkCall(PyObject * pValue, char const * methodName) {
  if(pValue == NULL) {
    cerr<<" Error: Method "<<methodName<<" call failed"<<endl;
    PyErr_Print();
    return false;
  }
  if(pyEnv::get().verbose) cerr<<" Method "<<methodName<<" is called"<<endl;
  return true;
}
