
#include "MarsCamera.h"
#include "TestMarsCamera.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <typeinfo>
#include "ConfigParser.h"
#include "pyEmbedding.h"
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;

int main() {
    ConfigParser my_cfg();
    pyEmbedding myPyEmb("pyMarsCamera_new\0","marsCameraClient\0", "/usr/lib/marsgui:/usr/lib/marsgui/marsct:/home/marsadmin/drv_py_nw/v2\0");
// #ifndef TEST
//     MarsCamera camera; //создали элемент класса
//     //     camera.verbose = 1;  //подробный тестовый вывод, по умолчанию 0
//       camera.Connect("192.168.0.135"); //соединились 
//     //camera.Connect("192.168.0.29"); //соединились 
//     //   camera.Connect("192.168.0.46"); //соединились 
//      camera.ReadConf();
//     camera.Disconnect();//отключились
// #else
//     TestMarsCamera test;
//     test.Connect("192.168.0.135"); //соединились 
//     test.ReadConf();
//     test.OMRValue("GainMode", MarsCamera::SuperHighGain); //поcтавили в OMR "GainMode" равным kSuperHighGain
//     test.WriteOMR(); //отправили на томограф (влить в предыдущую фукнцию?)
//     cout<<" Written OMR"<<endl;
// 
//     test.DACValue("Threshold0", 60); //уcтановили Threshold0=80 в DAC "Threshold0"
//     test.WriteDAC();
//     cout<<" Written DACs"<<endl; 
//     test.TestMaskUpload();
//     test.TestImageDownload("23");
//     test.Disconnect();//отключились
// #endif
    return 0; 
}
