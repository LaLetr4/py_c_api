
#include "MarsCamera.h"
#include <ctime>
#include <unistd.h>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"

unsigned MarsCamera::counter = 0;
bool MarsCamera::verbose = 0;
PyObject * MarsCamera::pModule, * MarsCamera::pDict, * MarsCamera::pClass;

void MarsCamera::FatalError(const char * diag, int code/* = -1*/) {
  cerr<<" Error: "<<diag<<endl;
  PyErr_Print();
  cerr<<" Exiting..."<<endl;
  exit(code);
}
void MarsCamera::InitPy() {
  Py_Initialize(); // инициализируем Python C Api
  import_array(); // инициализируем NumPy C Api

  PyObject * pName = PyUnicode_FromString("pyMarsCamera_new");
  pModule = PyImport_Import(pName);
  Py_DECREF(pName);
  if(pModule == NULL) {
    FatalError("Bad module ``pyMarsCamera''");
  }
  pDict = PyModule_GetDict(pModule);
  if(pDict == NULL) {
    Py_DECREF(pModule);
    FatalError("Bad dictionary in module ``pyMarsCamera''");
  }
  pClass = PyDict_GetItemString(pDict, "marsCameraClient");
  if(pClass == NULL) {
    Py_DECREF(pDict);
    Py_DECREF(pModule);
    FatalError("Wrong class ``marsCameraClient''");
  }
  if(verbose > 0) cerr<<" Module ``pyMarsCamera'' is succesfully loaded"<<endl;
}
void MarsCamera::FinalizePy() {
//   Py_DECREF(pClass);
  Py_DECREF(pDict);
  Py_DECREF(pModule);
  Py_Finalize();
}
MarsCamera::MarsCamera() {
  if(!counter) InitPy();
  counter++;
  // создать экземпляр класса
  if(PyCallable_Check(pClass)) {
    pInstance = PyObject_CallObject(pClass, NULL);
  } else {
    Py_Finalize();
    FatalError("with python class creation");
  }
  //здесь будет нормальная подгрузка конфигов
  OMR = PyDict_New();//create new python dictionary
  OMRValue("GainMode", SuperHighGain);
  DAC = PyDict_New();
  DACValue("Threshold0", 80);
  DACValue("Threshold1", 90);
  DACValue("I_Krum", 30);
  DACValue("I_Preamp", 150);
  DACValue("I_Shaper", 150);
  DACValue("V_Gnd", 135);
  DACValue("V_Cas", 174);
  DACValue("V_Fbk", 177);
}
MarsCamera::~MarsCamera() {
  Py_DECREF(pInstance);
  counter--;
  if(!counter) FinalizePy();
}
void MarsCamera::Call(const char * methodName) {
  PyObject * pValue = PyObject_CallMethod(pInstance, const_cast<char*>(methodName), "()");
  if(CheckCall(pValue, methodName))
    Py_DECREF(pValue);
}
void MarsCamera::Call(const char * methodName, int i) {
  PyObject * pValue = PyObject_CallMethod(pInstance, const_cast<char*>(methodName), "(i)", i);
  if(CheckCall(pValue, methodName))
    Py_DECREF(pValue);
}
void MarsCamera::Call(const char * methodName, PyObject * pArg) {
  PyObject * pName = PyUnicode_FromString(methodName);
  PyObject * pValue = PyObject_CallMethodObjArgs(pInstance, pName, pArg, NULL);
  Py_DECREF(pName);
  if(CheckCall(pValue, methodName))
    Py_DECREF(pValue);
}
void MarsCamera::Call(const char * methodName, const char * s) {
  PyObject * pValue = PyObject_CallMethod(pInstance, const_cast<char*>(methodName), "(s)", s);
  if(CheckCall(pValue, methodName))
    Py_DECREF(pValue);
}
bool MarsCamera::CheckCall(PyObject * pValue, char const * methodName) {
  if(pValue == NULL) {
    cerr<<" Error: Method "<<methodName<<" call failed"<<endl;
    PyErr_Print();
    return false;
  }
  if(verbose > 0) cerr<<" Method "<<methodName<<" is called"<<endl;
  return true;
}
void MarsCamera::DictValue(PyObject * pDict, const char * key, int val, const char * dictName) {
  PyObject * pValue = Py_BuildValue("i", val);
  bool res = PyDict_SetItemString(pDict, key, pValue);
  Py_DECREF(pValue);
  if(res == -1)
    cerr<<" Warning: Failed to set "<<dictName<<" key ``"<<key<<"'' to ``"<<val<<"''"<<endl;
}

// void MarsCamera::PRN(PyObject * obj) {
//   PyObject * pFunc = PyDict_GetItemString(pDict, "test_print");
//   if(pFunc == NULL) {
//     FatalError("Wrong function ``test_print''");
//   }
//   PyObject * pValue = PyObject_CallFunctionObjArgs(pFunc, obj, NULL);
//   if(CheckCall(pValue, "test_print"))
//     Py_DECREF(pValue);
//   Py_DECREF(pFunc);
// }

void MarsCamera::TakePicture(double exposure_sec, double wait_sec){
  if(verbose > 0)
    cout<<" Take ("<<exposure_sec<<" second exposure) an image"<<endl;
  clock_t t = clock();
  Call("acquire", static_cast<int>(exposure_sec*1000));
  t = clock() - t;
  if(verbose > 0)
    cout<<" Acquired image"
        <<" over a period of "<<static_cast<float>(t)/CLOCKS_PER_SEC
        <<" for the exposure "<<exposure_sec<<" seconds"<<endl;
  usleep(wait_sec*1e6);
}

vector<uint16_t> & MarsCamera::GetImage(FrameCounter cnt, double exposure_sec, double wait_sec) {
  TakePicture(exposure_sec, wait_sec); // делаем снимок
  return DownloadImage(cnt, wait_sec); // скачиваем картинку
}

vector<uint16_t> & MarsCamera::DownloadImage(FrameCounter cnt, double wait_sec) {
  Call("download_image", cnt); // подгружаем сделанную картинку в поле frame
  usleep(wait_sec*1e6);

  // достаём картинку из поля frame
  PyObject * pFrameArray = PyObject_GetAttrString(pInstance, "frame");
  if(pFrameArray == NULL) {
    cerr<<" Warning: Unable to get ``frame'' attribute"<<endl;
    return bitmap;
  }
  if(!PyList_Check(pFrameArray)) {
    cerr<<" Warning: ``frame'' attribute is not a list!"<<endl;
    Py_DECREF(pFrameArray);
    return bitmap;
  }
  PyObject * pFrame = PyList_GetItem(pFrameArray, cnt);
  if(!PyArray_Check(pFrame)) {
    cerr<<" Warning: ``frame["<<cnt<<"]'' is not a numpy array!"<<endl;
    Py_DECREF(pFrame);
    Py_DECREF(pFrameArray);
    return bitmap;
  }
  PyArrayObject * pArray = reinterpret_cast<PyArrayObject*>(pFrame);
  int n_dim = PyArray_NDIM(pArray);
  if(n_dim != 2) {
    cerr<<" Warning: ``frame["<<cnt<<"]'' is not a two-dimensional numpy array!"<<endl;
    Py_DECREF(pFrame);
    Py_DECREF(pFrameArray);
    return bitmap;
  }
  npy_intp * dims = PyArray_DIMS(pArray);
  size_t sz = dims[0]*dims[1];
  uint16_t * data = static_cast<uint16_t*>(PyArray_DATA(pArray));
  bitmap.assign(data, data+sz);
  Py_DECREF(pFrame);
  Py_DECREF(pFrameArray);
  return bitmap;
}

void MarsCamera::UploadMask(const char * maskName, uint16_t code /** = 0x0fff*/) {
  PyObject * pValue =
    PyObject_CallMethod(pInstance, const_cast<char*>("upload_mask"), "(si)", maskName, code);
  if(CheckCall(pValue, "upload_mask"))
    Py_DECREF(pValue);
}

void MarsCamera::TestMaskUpload(){
    if (verbose){
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

void MarsCamera::TestImageDownload(){
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
    ofstream fout("img/image_cpp2");
    for(size_t i = 0; i < image_size; i++){
      fout<<bitmap[i]<<' ';
      if (i%width == width - 1) fout<<endl;
    }
    fout.close();
  }
}

















































