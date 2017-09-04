
#pragma once

#include "pyEmbedding.h"

class cameraWrapper {
private:
  pyInstance * sys, * camera;
  /** array for storing current image */
  vector<uint16_t> bitmap;
  unsigned dim_w, dim_h;
  bool is_test;
public:
  cameraWrapper(string config, string _is_test = "");
  ~cameraWrapper();
  long getChipsNumber();
  void setThresholds(int chip_id, int number_of_thresholds, int * thresholds);  //установка значения нескольких порогов
  void setOneThreshold(int chip_id, int threshold_num, int threshold_val); //установка значения одного порога определённым значением
  long getThreshold(int chip_id, int threshold_num);
  void setBias(int bias = 1, int step = 100);
  void biasOff();
  long getBias();
  vector<uint16_t> & acquire(float expose_time = 0);
  vector<uint16_t> & getAcquiredImage() { return bitmap; }  //указатель на последний сделанный снимок
  unsigned getHeight() { return dim_h; }
  unsigned getWidth() { return dim_w; }
  void saveAsTxt(std::string image_name, std::string folder = ".");
};
