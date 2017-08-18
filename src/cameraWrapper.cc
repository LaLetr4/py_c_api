
#include "cameraWrapper.h"
#include "ConfigParser.h"
#include <iostream>
using std::cerr;
using std::endl;

cameraWrapper::cameraWrapper(string config): sys(0), camera(0) {
  ConfigParser cfg(config);
  pyEnv::PYTHONPATH = cfg.GetString("pythonPath"); // установка переменной PYTHONPATH
  pyArgSetter py_args;
  py_args.setArgs("camera_testing");  //передача аргументов в интерпретатор
  sys = pyInstance::instanceOf(cfg.GetCString("className"), cfg.GetCString("moduleName"));
  camera = sys->get("getCamera");
  camera->call("check_config_loaded");
}
cameraWrapper::~cameraWrapper() {
  camera->call("finalise");
}
long cameraWrapper::getChipsNumber() {
  return camera->getAttrInt("_chipcount");
//   return camera->getInt("getChipCount");
}
void cameraWrapper::setThresholds(int chip_id, int number_of_thresholds, int * thresholds) {
  if(chip_id >= getChipsNumber()) {
    cerr<<"chip_id is greater than number of chips in camera! Exiting..."<<endl;
    return;
  }
//   set_threshold(self, threshold, counter=0, chipid=None) -- python method signature
  for(int i = 0; i < number_of_thresholds; i++) {
    PyObject * pValue =
        PyObject_CallMethod(*camera, const_cast<char*>("set_threshold"),
                            "(iii)", thresholds[i], i, chip_id);
    if(pyInstance::checkCall(pValue, "set_threshold"))
      Py_DECREF(pValue);
  }
}
vector<uint16_t> & cameraWrapper::acquire(float expose_time) {
  bitmap.clear();
//   acquire(self, exptime=None, chipid="all", attempts=0)
  if(expose_time == 0)
    camera->call("acquire");
  else
    camera->call("acquire", expose_time);
//   numpy.array = build_multiframe(self, frames=None, enabled_chips="all")
  PyObject * pFrame = PyObject_CallMethod(*camera, const_cast<char*>("build_multiframe"), "()");
  if(!pyInstance::checkCall(pFrame, "build_multiframe")) {
    cerr<<" Warning: failed getting image!"<<endl;
    return bitmap;
  }
/*  if(!PyArray_Check(pFrame)) { // segfault here !!!
    cerr<<" Warning: frame is not a numpy array!"<<endl;
    Py_DECREF(pFrame);
    return bitmap;
  }*/
  PyArrayObject * pArray = reinterpret_cast<PyArrayObject*>(pFrame);
  int n_dim = PyArray_NDIM(pArray);
  if(n_dim != 2) {
    cerr<<" Warning: returned frame is not a two-dimensional numpy array!"<<endl;
    Py_DECREF(pFrame);
    return bitmap;
  }
  npy_intp * dims = PyArray_DIMS(pArray);
  size_t sz = dims[0]*dims[1];
  dim_w = dims[0];
  dim_h = dims[1];
  uint16_t * data = static_cast<uint16_t*>(PyArray_DATA(pArray));
  bitmap.assign(data, data+sz);
  Py_DECREF(pFrame);
  return bitmap;
}
