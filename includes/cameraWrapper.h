
#pragma once

#include "pyEmbedding.h"

class cameraWrapper {
private:
  pyInstance * sys, * camera;
  /** array for storing current image */
  vector<uint16_t> bitmap;
  unsigned dim_w, dim_h;
public:
  cameraWrapper(string config);
  ~cameraWrapper();
  long getChipsNumber();
  void setThresholds(int chip_id, int number_of_thresholds, int * thresholds);
  vector<uint16_t> & acquire(float expose_time = 0);
  vector<uint16_t> & getAcquiredImage() { return bitmap; }
  unsigned getHeight() { return dim_h; }
  unsigned getWidth() { return dim_w; }
};
