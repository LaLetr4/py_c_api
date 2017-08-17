
#ifndef testMarsCamera_h
#define testMarsCamera_h

#include <iostream>
#include <fstream>
#include "MarsCamera.h"

using std::cout;
using std::endl;

class TestMarsCamera: public MarsCamera {
public:
  TestMarsCamera(string settings_file = "conf.ini"): MarsCamera(settings_file) {
    //cout<<"testCamera object is created"<<endl;
  }
  ~TestMarsCamera() { ; }
  void TestMaskUpload() {
    if (verbose) {
      cout<<"Test Mask Upload"<<endl;
    }
//     UploadMask("/etc/mars/config/GaAs-N1-8-V5-05Nov2015-1455_software_colour_csm_full_0_mask.npy");
//     UploadMask("/etc/mars/config/CZT-2x1-5.3-sn0105-09Jun2017-2051_colour_csm_full_0_mask.npy");
//     UploadMask("/etc/mars/config/CZT-3x1-5.3-sn0105-18июн2017-1104_colour_csm_full_0_mask.npy");
//     UploadMask("/etc/mars/config/CZT-3x1-5.3-sn0105-19июн2017-1217_colour_csm_full_0_mask.npy");
//     UploadMask("/etc/mars/config/CZT-3x1-5.3-sn0105-19июн2017-1217_colour_csm_full_1_mask.npy");
    UploadMask("/etc/mars/config/CZT-3x1-5.3-sn0105-19июн2017-1217_colour_csm_full_2_mask.npy");
//     std::cout<<"Upload is tested"<<std::endl;
//     PyErr_Print();
//     PyErr_Clear();
  }
  
  void TestImageDownload(std::string test_image){
    if (verbose){
        cout<<"Test Image Download"<<endl;
    }
    vector<uint16_t> & bitmap = GetImage(MarsCamera::LFrame, 3., 1.);

  if (fmod(sqrt(bitmap.size()),1) != 0) {
    cout<<"bitmap is not a square!"<<endl;
  } else {
    size_t image_size = bitmap.size();
    size_t width = sqrt(image_size);
    cout<<"img_size = "<<image_size<<", width = "<<width<<endl;
    std::string img_name = test_image;//"image_cpp2";
    std::string dir_name = "img";
    std::ofstream fout(dir_name+"/"+img_name);
    cout<<"image \""<<img_name<<"\" is created in \""<<dir_name<<"/\""<<endl;
    for(size_t i = 0; i < image_size; i++){
      fout<<bitmap[i]<<' ';
      if (i%width == width - 1) fout<<endl;
    }
    fout.close();
  }
}
};

#endif
