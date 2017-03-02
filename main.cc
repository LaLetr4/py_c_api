
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
  size_t image_size;
  //получаем с камеры снимок (размером image_size) с выдержской 3 сек, ожиданием 1 сек и параметром l
  uint16_t * bitmap = camera.GetImage('l', 3., 1., &image_size);

//   cout<<"test print of obtained bitmap: \n";
//   for(size_t i = 0; i < 256; i++)
//     for(size_t j = 0; j < 256; j++)
// //         cout<<"i="<<i<<" j="<<j<<' '<<bitmap[i+j*image_size];
//         cout<<' '<<bitmap[i+j*256];
//     cout<<endl;
//
    if (fmod(sqrt(image_size),1)!=0){
      cout<<"bitmap is not a square!"<<endl;
    }else{
      int width = sqrt(image_size);
      ofstream fout("image_cpp");
      for(size_t i = 0; i < image_size; i++){
        fout<<bitmap[i]<<' ';
        if (image_size%width == 0) fout<<endl;
      }
      fout.close();
    }
  camera.Disconnect();//отключились
}
