
#include "MarsCamera.h"
#include <iostream>
#include <fstream>
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;

int main() {
  MarsCamera camera; //создали элемент класса
  camera.Connect("192.168.0.44"); //соединились по IP

  camera.OMRValue("GainMode", kSuperHighGain); //поcтавили в OMR "GainMode" равным kSuperHighGain
  camera.WriteOMR(); //отправили на томограф (влить в предыдущую фукнцию?)
  cout<<" Written OMR"<<endl;

  camera.DACValue("Threshold0", 80); //уcтановили Threshold0=80 в DAC "Threshold0"
  camera.WriteDAC();
  cout<<" Written DACs"<<endl;

/* TODO ?? если нужно ??
  camera.upload_image(numpy.zeros([256, 256]), 0)
  camera.upload_image(numpy.zeros([256, 256]), 1)
  print "uploaded image"

  camera.test_mask_read_write(mask_K09 & 0x0fff)
  print "tested mask read and write"
*/
  //получаем с камеры снимок с выдержской 3 сек, ожиданием 1 сек и параметром l
  vector<uint16_t> & bitmap = camera.GetImage('l', 3., 1.);

  if (fmod(sqrt(bitmap.size()),1) != 0) {
    cout<<"bitmap is not a square!"<<endl;
  } else {
    size_t image_size = bitmap.size();
    size_t width = sqrt(image_size);
    ofstream fout("image_cpp");
    for(size_t i = 0; i < image_size; i++){
      fout<<bitmap[i]<<' ';
      if (image_size%width == 0) fout<<endl;
    }
    fout.close();
  }
  camera.Disconnect();//отключились
}
