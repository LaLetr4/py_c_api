
#include <iostream>

using namespace std;

class MarsSystem: public pyEmbedding {
public:
  MarsSystem (string settings_file = "conf.ini"): pyEmbedding(settings_file) {
    cout<<"MarsSystem is initialized \n";
  }
  virtual ~MarsSystem () { }
  void test(){
    Call("test");
  }
};
