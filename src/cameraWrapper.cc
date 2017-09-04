
#include "cameraWrapper.h"
#include "ConfigParser.h"
#include <iostream>
#include <fstream>

// #include <sys/types.h>
// #include <sys/stat.h>
// #include <unistd.h>
// #include <stdio.h>

using std::cerr;
using std::endl;

cameraWrapper::cameraWrapper(string config, string _is_test): sys(0), camera(0), is_test(false) {
  ConfigParser cfg(config);
  pyEnv::PYTHONPATH = cfg.GetString("pythonPath"); // установка переменной PYTHONPATH
  if(_is_test == "test") {
    cerr<<"is_test == \"test\""<<endl;
    is_test = true;
    pyArgSetter py_args;
    py_args.setArgs("camera_testing");  //передача аргументов в интерпретатор
  }
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
//   set_threshold(self, threshold, counter=0, chipid=None) -- python method signature
  for(int i = 0; i < number_of_thresholds; i++) {
    setOneThreshold(chip_id, i, thresholds[i]);
  }
}

void cameraWrapper::setOneThreshold(int chip_id, int threshold_num, int threshold_val) {
  if(chip_id >= getChipsNumber()) {
    cerr<<"chip_id is greater than number of chips in camera! Exiting..."<<endl;
    return;
  }
  PyObject * pValue =
      PyObject_CallMethod(*camera, const_cast<char*>("set_threshold"),
                          "(iii)", threshold_val, threshold_num, chip_id);
  if(pyInstance::checkCall(pValue, "set_threshold 1"))
    Py_DECREF(pValue);
  long g = getThreshold(chip_id, threshold_num);
  if(g != threshold_val) {
    cerr<<" Warning!"<<endl
        <<"  cameraWrapper::setOneThreshold("<<chip_id<<", "<<threshold_num<<", "<<threshold_val<<")"<<endl
        <<"  threshold_val != getThreshold("<<chip_id<<", "<<threshold_num<<") == "<<g<<endl;
  }
}
long cameraWrapper::getThreshold(int chip_id, int threshold_num) {
  if(chip_id >= getChipsNumber()) {
    cerr<<"chip_id is greater than number of chips in camera! Exiting..."<<endl;
    return 0;
  }
  PyObject * pInt
      = PyObject_CallMethod(*camera, const_cast<char*>("get_threshold"),
                            "(ii)", threshold_num, chip_id);
  return camera->toInt(pInt, "get_threshold");
}
void cameraWrapper::setBias(int bias, int step) {
  PyObject * pValue =
      PyObject_CallMethod(*camera, const_cast<char*>("set_hv"),
                          "(ii)", bias, step);
  if(pyInstance::checkCall(pValue, "set_hv"))
    Py_DECREF(pValue);
  if(is_test) return;
  long g = getBias();
  if(g != bias) {
    cerr<<" Warning!"<<endl
        <<"  cameraWrapper::setBias("<<bias<<", "<<step<<")"<<endl
        <<"  bias != getBias() == "<<g<<endl;
  }
}
void cameraWrapper::biasOff() {
  int base_hv = camera->getAttrInt("base_hv");
  setBias(base_hv);
}
long cameraWrapper::getBias() {
  return camera->getInt("get_hv");
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
  //test build one frame
//     PyObject * pFrame = PyObject_CallMethod(*camera, const_cast<char*>("build_frame"), "()");
//   if(!pyInstance::checkCall(pFrame, "build_frame")) {
//     cerr<<" Warning: failed getting image!"<<endl;
//     return bitmap;
//   }
  //test
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

void cameraWrapper::saveAsTxt(std::string image_name, std::string folder) {
  size_t image_size = bitmap.size();
  size_t width = getWidth();
//   if (fmod(sqrt(bitmap.size()),1) != 0) {    //todo: придумать актуальную проверку на размер картинки
//     cout<<"bitmap is not a square!"<<endl;
//   } else {
  std::ofstream fout(folder+"/"+image_name);
  for(size_t i = 0; i < image_size; i++){
    fout<<bitmap[i]<<' ';
    if (i%width == width - 1) fout<<endl;
  }
  fout.close();
//   cout<<"image file descriptor = "<<fileno(fout)<<endl;
  cout<<"image \""<<image_name<<"\" is created in \""<<folder<<"/\""<<endl;
}
