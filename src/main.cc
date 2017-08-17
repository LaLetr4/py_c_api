
#include <iostream>
#include "pyEmbedding.h"
#include "ConfigParser.h"

using std::cout;
using std::cerr;
using std::endl;

class configLoader: public ConfigParser { // mediator
public:
  configLoader(string filename): ConfigParser(filename) {
    pyEnv::PYTHONPATH = GetString("pythonPath");    //установка переменной PYTHONPATH
    pyArgSetter py_args;
    py_args.setArgs("camera_testing");  //передача аргументов в интерпретатор
  }
  pyInstance * loadInstance() {
    return pyInstance::instanceOf(GetCString("className"), GetCString("moduleName"));
  }
};

int main() {
    configLoader conf("conf_sys.ini");
    pyInstance * sys = conf.loadInstance();
    sys->call("test");
    //   return 0;
    //test camera obtain
    pyInstance * camera = sys->get("getCamera"); // hangs
    camera->call("check_config_loaded");
    camera->call("finalise");

    cout<<"finish"<<endl;
    return 0;
}
