
#include "MarsCamera.h"
#include <ctime>
#include <unistd.h>
#include <iostream>
#include <string.h>
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::string;
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"

unsigned MarsCamera::counter = 0;
bool MarsCamera::verbose = 0;
PyObject * MarsCamera::pModule, * MarsCamera::pDict, * MarsCamera::pClass;

void MarsCamera::DictValue(PyObject * pDict, const char * key, int val, const char * dictName) {
  PyObject * pValue = Py_BuildValue("i", val);
  /*bool*/int res = PyDict_SetItemString(pDict, key, pValue);
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


















































