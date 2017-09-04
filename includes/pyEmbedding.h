
#ifndef pyEmbedding_h
#define pyEmbedding_h

#include <string>
#include <vector>
#include <list>
#include <map>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"

using namespace std;

class pyModule;
class pyClass;
class pyInstance;

class pyEnv { // singleton
private:
  pyEnv();
  pyEnv(const pyEnv &);
  pyEnv & operator=(pyEnv &);
public:
  static std::string PYTHONPATH;
  static pyEnv & get();
  ~pyEnv();
  std::map<std::string, pyModule*> gModules;
  std::map<std::string, pyClass*> gClasses;
  std::list<pyInstance*> gInstances;
  bool verbose; // test information on/off for debagging
  void fatalError(const char * diag, const char * spec = 0, int code = -1);
};

class pyArgSetter {
private:
  std::vector<const char *> args;
public:
  pyArgSetter(const char * script = "");
  virtual ~pyArgSetter();
  virtual void addArg(const char * arg);
  virtual void set();
  virtual void setArgs(const char * _args);
  virtual void printArgs();
};

class pyNamed {
protected:
  std::string name;
public:
  pyNamed(std::string _name): name(_name) { }
  std::string getName() { return name; }
};
class pyModule: public pyNamed {
private:
  pyModule(const char * modName); // Prevent construction outside pyModule::import
  pyModule(const pyModule &); // Prevent copy-construction
  pyModule & operator=(const pyModule &); // Prevent assignment
protected:
  PyObject * _module, * _dict;
public:
  static pyModule * import(const char * modName); // фабричный метод
  virtual pyClass * getClass(const char * className); // фабричный метод
  virtual ~pyModule();
};

class pyClass: public pyNamed {
  friend class pyModule;
private:
  // Prevent construction outside pyModule::getClass
  pyClass(const char * className, PyObject * pDict);
  pyClass(const pyClass &); // Prevent copy-construction
  pyClass & operator=(const pyClass &); // Prevent assignment
protected:
  PyObject * _class;
public:
  static pyClass * importFrom(const char * className, const char * modName); // фабричный метод
  virtual pyInstance * instance();
  virtual ~pyClass();
};

class pyInstance: public pyNamed {
  friend class pyClass;
private:
  // Prevent construction outside pyClass::instance & pyInstance::get
  pyInstance(PyObject * pInstance, std::string _name);
  pyInstance(const pyInstance &); // Prevent copy-construction
  pyInstance & operator=(const pyInstance &); // Prevent assignment
protected:
  PyObject * _instance;
public:
  static pyInstance * instanceOf(const char * className, const char * modName); // фабричный метод
  virtual ~pyInstance();
  virtual void call(const char * methodName);
  virtual void call(const char * methodName, int i);
  virtual void call(const char * methodName, float f);
  virtual void call(const char * methodName, PyObject * pArg);
  virtual void call(const char * methodName, const char * s);
  virtual pyInstance * get(const char * methodName);
  virtual long getInt(const char * methodName);
  virtual long getAttrInt(const char * attrName);
  virtual long toInt(PyObject * pInt, const char * diag = "");
  operator PyObject*() { return _instance; }
  static bool checkCall(PyObject * pValue, char const * methodName);
};


#endif
