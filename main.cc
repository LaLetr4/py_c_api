
#include "MarsCamera.h"
#include <iostream>
#include <fstream>
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;

int main() {
    MarsCamera camera; //создали элемент класса
//     camera.verbose = 1;  //подробный тестовый вывод, по умолчанию 0
//     camera.Connect("192.168.0.135"); //соединились по IP
//     camera.Connect("192.168.0.29"); //соединились по IP
    camera.Connect("192.168.0.46"); //соединились по IP

    camera.OMRValue("GainMode", MarsCamera::SuperHighGain); //поcтавили в OMR "GainMode" равным kSuperHighGain
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
    camera.TestMaskUpload();
    camera.TestImageDownload();
    camera.Disconnect();//отключились
}
