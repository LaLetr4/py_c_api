
#include "cameraWrapper.h"
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

int main() {
  cameraWrapper camera("conf_sys.ini");
  cout<<"chips = "<<camera.getChipsNumber()<<endl;
  int n_thr = 2;
  int thr[2] = {1, 2};
  camera.setThresholds(0, n_thr, thr);
  vector<uint16_t> & bmp = camera.acquire();
  cout<<"width = "<<camera.getWidth()<<endl;
  cout<<"height = "<<camera.getHeight()<<endl;
  cout<<"image size = "<<bmp.size()<<endl;
  for(unsigned i = 0; i < 10 && i < bmp.size(); i++)
    cout<<i<<' '<<bmp[i]<<endl;
  cout<<"finish"<<endl;
  return 0;
}
