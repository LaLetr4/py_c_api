
#include <iostream>
#include "pyEmbedding.h"
#include "ConfigParser.h"
// #include "MarsCamera.h"
// #include "MarsCameraGen.h"
// #include "MarsSystem.h"
// #include "TestMarsCamera.h"
using std::cout;
using std::cerr;
using std::endl;

class configLoader: public ConfigParser { // mediator
public:
  configLoader(string filename): ConfigParser(filename) {
    pyEnv::PYTHONPATH = GetString("pythonPath");
    pyArgSetter py_args;
    py_args.setArgs("camera_testing");
  }
  pyInstance * loadInstance() {
    return pyInstance::instanceOf(GetCString("className"), GetCString("moduleName"));
  }
};

int main() {
//     MarsCamera camera("conf.ini"); //создали элемент класса   
    /*
    camera.printEmbInfo();
    camera.testCall("my_test_function");
    
    camera.verbose = 1;  //подробный тестовый вывод, по умолчанию 0
//       camera.Connect("192.168.0.135"); //соединились 
//     //camera.Connect("192.168.0.29"); //соединились 
//     //   camera.Connect("192.168.0.46"); //соединились 
//      camera.ReadConf();
//     camera.Disconnect();//отключились
// #else
    TestMarsCamera test;
//     test.Connect("192.168.0.135"); //соединились 
//     test.ReadConf();
    test.OMRValue("GainMode", MarsCamera::SuperHighGain); //поcтавили в OMR "GainMode" равным kSuperHighGain
    test.WriteOMR(); //отправили на томограф (влить в предыдущую фукнцию?)
    cout<<" Written OMR"<<endl;

    test.DACValue("Threshold0", 60); //уcтановили Threshold0=80 в DAC "Threshold0"
    test.WriteDAC();
//     cout<<" Written DACs"<<endl; 
//     test.TestMaskUpload();
//     test.TestImageDownload("23");
//     test.Disconnect();//отключились
// #endif
*/

//     MarsSystem sys("conf_sys.ini");
  configLoader conf("conf_sys.ini");
  pyInstance * sys = conf.loadInstance();
  sys->call("test");
//   return 0;
  //test camera obtain
  pyInstance * camera = sys->get("getCamera"); // hangs
  camera->call("check_config_loaded");
  camera->call("finalise");
//     sys.Call("reinitialiseCamera");

    /*
    //hard camera test
    PyObject * pValue = PyObject_CallMethod(camera, "getChipCount", "()");
    if(pValue == NULL) {
        cerr<<" Error: Method test call failed"<<endl;
        PyErr_Print();
    } else {
        cerr<<" Method test is called"<<endl;
    }
    */
/*    
    //test log obtain
    PyObject * log = sys.Get("getLog");
    
   PyObject * pValue = PyObject_CallMethod(log, "test", "(s)", "test marslog");
//    PyObject * pValue = PyObject_CallMethod(log,"print_log", "(s)", "Test of marslog print_log method");
    if(pValue == NULL) {
        cerr<<" Error: Method test call failed"<<endl;
        PyErr_Print();
    }else{
        cerr<<" Method test is called"<<endl;
    }*/
    
    
    
//     sys.test();
//     Py_DECREF(log);
    

//     MarsCameraGen cam("conf.ini");
//       cam.test();
//     cam.verbose = true;    //verbose informaion for testing
//     if (cam.verbose){
//         cout<<"verbose info is switched on\n";
//     }
// //     cam.getChipCount();
//     cam.testConfig_loaded();
    
    cout<<"finish"<<endl;
    return 0;
}
