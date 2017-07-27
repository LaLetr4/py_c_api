
// #include "MarsCamera.h"
#include "TestMarsCamera.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "service.h"
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;

/*
================================================
IP детекторов:
192.168.0.135
192.168.0.29
192.168.0.4
================================================

*/


int main() {
    setPythonPath();   
    TestMarsCamera test;
    test.Connect("192.168.0.135"); //соединились 
    test.ReadConf();
    test.OMRValue("GainMode", MarsCamera::SuperHighGain); //поcтавили в OMR "GainMode" равным kSuperHighGain
    test.WriteOMR(); //отправили на томограф (влить в предыдущую фукнцию?)
    cout<<" Written OMR"<<endl;

    test.DACValue("Threshold0", 60); //уcтановили Threshold0=80 в DAC "Threshold0"
    test.WriteDAC();
    cout<<" Written DACs"<<endl; 
    test.TestMaskUpload();
    test.TestImageDownload("23");
    test.Disconnect();//отключились
// #endif
    return 0; 
}
