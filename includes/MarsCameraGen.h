#include <iostream>
#include "pyEmbedding.h"

using namespace std;

class MarsCameraGen: public pyEmbedding {
public:
//         static unsigned counter;
//     unsigned pyEmbedding::counter = 0;

  MarsCameraGen (string settings_file = "conf.ini"): pyEmbedding(settings_file) {
    cout<<"MarsCameraGen is initialized\n";
  }

  virtual ~MarsCameraGen () { }

  void getChipCount() {
    Call("getChipCount");
  }

  void testConfig_loaded() {
    Call("check_config_loaded");
  }

  void test() {
    Call("test");
  }
//     ~MarsCameraGen() {
// //     Py_DECREF(pInstance);
//     counter--;
//     cout<<"test1\n";
// //     if(!counter) FinalizePy();
//     if (verbose)
//         cout<<"pyEmbedding is destroyed";
//     }
    
    /*
    void FinalizePy() {
        Py_DECREF(pClass);
        cout<<"1"<<endl;
        Py_DECREF(pDict);
                cout<<"2"<<endl;
        Py_DECREF(pModule);
                cout<<"3"<<endl;
        Py_Finalize();
                cout<<"4"<<endl;
}*/
private:
  static PyObject * pModule, * pDict, * pClass;
  PyObject * pInstance;
};
