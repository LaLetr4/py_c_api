
#include "cameraWrapper.h"
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

int main() {
  cameraWrapper camera("conf_sys.ini", "test");
  cout<<"chips = "<<camera.getChipsNumber()<<endl;
  camera.setBias(10);
  //установка порогов
  int n_thr = 2;
  int thr[2] = {1, 2};
  camera.setThresholds(0, n_thr, thr);  // chip_id, сколько порогов будем устанавливать, какие значения (будут установлены по очереди, начиная с 0)
  camera.setOneThreshold(0, 3, 75); // устанавливает значение 75 для порога 3 чипа 0
  //получение картинки и её параметров
  vector<uint16_t> & bmp = camera.acquire();
  cout<<"width = "<<camera.getWidth()<<endl;
  cout<<"height = "<<camera.getHeight()<<endl;
  cout<<"image size = "<<bmp.size()<<endl; 
  
  camera.saveAsTxt("img", "test_image_txt"); // сохрание картинки в текстовый файл
  
//   for(unsigned i = 0; i < 10 && i < bmp.size(); i++) //todo: вспомнить, зачем это тут было
//     cout<<i<<' '<<bmp[i]<<endl;
  cout<<"finish"<<endl;
  return 0;
}
