
#pragma once

#include <Python.h>
#include <cstdint>
#include <vector>
using std::vector;

class MarsCamera {
public:
  MarsCamera();
  ~MarsCamera();

  void Connect(const char * url) { Call("connect", url); }
  void Disconnect() { Call("disconnect"); }

  /**
    Sets OMR's key key to the value val
  */
  void OMRValue(const char * key, int val) { DictValue(OMR, key, val, "OMR"); }

  /**
    Sets DAC's key key to the value val
  */
  void DACValue(const char * key, int val) { DictValue(DAC, key, val, "DAC"); }

  /**
    Sets OMR fields to requested values
  */
  void WriteOMR() { Call("write_OMR", OMR); }

  /**
    Writes the DACs to the Medipix sensor
  */
  void WriteDAC() { Call("write_DACs", DAC); }

  /**
    Takes a picture with exposure exposure_sec and sleeps for wait_sec time interval
  */
  void TakePicture(double exposure_sec, double wait_sec);

  /**
    1) Downloads the picture to computer with cnt (L or H) threshold
    2) Converts it from numpy to uint16_t
    3) Returns image (array)
  */
  vector<uint16_t> & DownloadImage(char cnt, double wait_sec = 1);
  /**
    1) Takes a picture with exposure exposure_sec and sleeps for wait_sec time interval
    2) Downloads the picture to computer with cnt (L or H) threshold
    3) Converts it from numpy to uint16_t
    4) Returns image (array)
  */
  vector<uint16_t> & GetImage(char cnt, double exposure_sec, double wait_sec = 1);
  PyObject * Instance() { return pInstance; }
  static bool verbose; //! test information on/off for debagging
private:
  /** variables */
  static unsigned counter;
  static PyObject * pModule, * pDict, * pClass;
  PyObject * pInstance;
  PyObject * OMR, * DAC;

  /** array for storing current image */
  vector<uint16_t> bitmap;

  /** functions */
  /**
    Inintializes the Python interpreter
  */
  static void InitPy();

  /**
    Finalizes the interpreter, frees memory
  */
  static void FinalizePy();

  /**
    Calls method methodName, maybe with some parameters
  */
  void Call(const char * methodName);
  void Call(const char * methodName, int i);
  void Call(const char * methodName, const char * s);
  void Call(const char * methodName, PyObject * pValue);

  /**
    Adds in dictionary pDict named dictName value val by the key key
  */
  void DictValue(PyObject * pDict, const char * key, int val, const char * dictName = "");

  /**
    Check if method methodName was called correctly
    pValue is result of method's work
  */
  static bool CheckCall(PyObject * pValue, char const * methodName);

  /**
    Handles connection errors
  */
  static void FatalError(const char * diag, int code = -1);

  static int ParseCounter(char cnt);
//   void PRN(PyObject * obj); //test print method for debugging
};
enum Gain { kSuperHighGain = 0, kLowGain = 1, kHighGain = 2, kSuperLowGain = 3 }; // Some thing we need for OMR
