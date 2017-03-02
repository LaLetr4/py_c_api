############################################
# CONSTANTS!! Please do not edit via code. #
############################################
##
# MXR DAC settings.
#
MXR_DAC = ["IKrum", "Disc", "Preamp", "BuffAnalogA", "BuffAnalogB", "DelayN", "THLFine",
           "THLCoarse", "THHFine", "THHCoarse", "FBK", "GND", "THS"]

MXR_RANGE = {"IKrum": 256,
             "Disc": 256,
             "Preamp": 256,
             "BuffAnalogA": 256,
             "BuffAnalogB": 256,
             "DelayN": 256,
             "THLFine": 1024,
             "THLCoarse": 16,
             "THHFine": 1024,
             "THHCoarse": 16,
             "FBK": 256,
             "GND": 256,
             "THS": 256}

# Some good default values to set the MXR to.
MXR_VALUES = {"IKrum": 20,
              "Disc": 128,
              "Preamp": 128,
              "BuffAnalogA": 128,
              "BuffAnalogB": 128,
              "DelayN": 128,
              "THLFine": 700,
              "THLCoarse": 7,
              "THHFine": 1023,
              "THHCoarse": 7,
              "FBK": 120,
              "GND": 100,
              "THS": 142}

MXR_SCAN = {"IKrum": 15,
            "Disc": 11,
            "Preamp": 7,
            "BuffAnalogA": 3,
            "BuffAnalogB": 4,
            "DelayN": 9,
            "THLFine": 6,
            "THLCoarse": 6,
            "THHFine": 12,
            "THHCoarse": 12,
            "FBK": 10,
            "GND": 13,
            "THS": 1}
##
# Timepix DAC settings.
#
TPX_DAC = ["IKrum", "Disc", "Preamp", "BuffAnalogA", "BuffAnalogB", "Hist", "THLFine",
           "THLCoarse", "Vcas", "FBK", "GND", "THS"]

TPX_RANGE = {"IKrum": 256,
             "Disc": 256,
             "Preamp": 256,
             "BuffAnalogA": 256,
             "BuffAnalogB": 256,
             "Hist": 256,
             "THLFine": 1024,
             "THLCoarse": 16,
             "Vcas": 256,
             "FBK": 256,
             "GND": 256,
             "THS": 256}

# Some good default values to set the timepix to.
TPX_VALUES = {"IKrum": 20,
              "Disc": 128,
              "Preamp": 128,
              "BuffAnalogA": 128,
              "BuffAnalogB": 128,
              "Hist": 128,
              "THLFine": 700,
              "THLCoarse": 7,
              "Vcas": 128,
              "FBK": 140,
              "GND": 128,
              "THS": 142}

TPX_SCAN = {"IKrum": 15,
            "Disc": 11,
            "Preamp": 7,
            "BuffAnalogA": 3,
            "BuffAnalogB": 4,
            "Hist": 9,
            "THLFine": 6,
            "THLCoarse": 6,
            "Vcas": 12,
            "FBK": 10,
            "GND": 13,
            "THS": 1}

##
# Medipix3 DAC settings.
#
MP3_DAC = ["Threshold0", "Threshold1", "Threshold2", "Threshold3", "Threshold4", "Threshold5",
           "Threshold6", "Threshold7", "Preamp", "Ikrum", "Shaper", "Disc", "Disc_LS", "ThresholdN",
           "DAC_pixel", "Delay", "TP_BufferIn", "TP_BufferOut", "RPZ", "GND", "TP_REF",
           "TP_REFA", "TP_REFB"]  # Should cas and fbk be in this list??

MP3_RANGE = {"Threshold0": 512,
             "Threshold1": 512,
             "Threshold2": 512,
             "Threshold3": 512,
             "Threshold4": 512,
             "Threshold5": 512,
             "Threshold6": 512,
             "Threshold7": 512,
             "Preamp": 256,
             "Ikrum": 256,
             "Shaper": 256,
             "Disc": 256,
             "Disc_LS": 256,
             "ThresholdN": 256,
             "DAC_pixel": 256,
             "Delay": 256,
             "TP_BufferIn": 256,
             "TP_BufferOut": 256,
             "RPZ": 256,
             "GND": 256,
             "FBK": 256,
             "Cas": 1500,
             "TP_REF": 256,
             "TP_REFA": 512,
             "TP_REFB": 512,
             "Cas": 1500,
             "FBK": 256}

# Some good default values to set the medipix3 to
MP3_VALUES = {"Threshold0": 30,
              "Threshold1": 511,
              "Threshold2": 511,
              "Threshold3": 511,
              "Threshold4": 511,
              "Threshold5": 511,
              "Threshold6": 511,
              "Threshold7": 511,
              "Preamp": 100,
              "Ikrum": 30,
              "Shaper": 100,
              "Disc": 150,
              "Disc_LS": 200,
              "ThresholdN": 36,
              "DAC_pixel": 180,
              "Delay": 128,
              "TP_BufferIn": 128,
              "TP_BufferOut": 128,
              "RPZ": 255,
              "GND": 128,
              "FBK": 128,
              "Cas": 1499,
              "TP_REF": 128,
              "TP_REFA": 256,
              "TP_REFB": 256}

MP3_SCAN = {"Threshold0": 1,
            "Threshold1": 2,
            "Threshold2": 3,
            "Threshold3": 4,
            "Threshold4": 5,
            "Threshold5": 6,
            "Threshold6": 7,
            "Threshold7": 8,
            "Preamp": 9,
            "Ikrum": 10,
            "Shaper": 11,
            "Disc": 12,
            "Disc_LS": 13,
            "ThresholdN": 14,
            "DAC_pixel": 15,
            "Delay": 16,
            "TP_BufferIn": 17,
            "TP_BufferOut": 18,
            "RPZ": 19,
            "GND": 20,
            "TP_REF": 21,
            "FBK": 22,
            "Cas": 23,
            "TP_REFA": 24,
            "TP_REFB": 25,
            "band_gap_output": 26,
            "band_gap_temperature": 27,
            "dac_bias": 28,
            "dac_cascade_bias": 29}

##
# Medipix3RX DAC settings.
#
RX3_DAC = ["Threshold0", "Threshold1", "Threshold2", "Threshold3", "Threshold4", "Threshold5",
           "Threshold6", "Threshold7", "I_Preamp", "I_Ikrum", "I_Shaper", "I_Disc", "I_Disc_LS",
           "I_Shaper_test", "I_DAC_DiscL", "I_DAC_test", "I_DAC_DiscH", "I_Delay", "I_TP_BufferIn",
           "I_TP_BufferOut", "V_Rpz", "V_Gnd", "V_Tp_ref", "V_Fbk", "V_Cas", "V_Tp_refA", "V_Tp_refB"]

RX3_RANGE = {"Threshold0": 512,
             "Threshold1": 512,
             "Threshold2": 512,
             "Threshold3": 512,
             "Threshold4": 512,
             "Threshold5": 512,
             "Threshold6": 512,
             "Threshold7": 512,
             "I_Preamp": 256,
             "I_Ikrum": 256,
             "I_Shaper": 256,
             "I_Disc": 256,
             "I_Disc_LS": 256,
             "I_Shaper_test": 256,
             "I_DAC_DiscL": 256,
             "I_DAC_test": 256,
             "I_DAC_DiscH": 256,
             "I_Delay": 256,
             "I_TP_BufferIn": 256,
             "I_TP_BufferOut": 256,
             "V_Rpz": 256,
             "V_Gnd": 256,
             "V_Tp_ref": 256,
             "V_Fbk": 256,
             "V_Cas": 256,
             "V_Tp_refA": 512,
             "V_Tp_refB": 512}

# Some good default values to set the medipix3 to
RX3_VALUES = {"Threshold0": 511,
              "Threshold1": 511,
              "Threshold2": 511,
              "Threshold3": 511,
              "Threshold4": 511,
              "Threshold5": 511,
              "Threshold6": 511,
              "Threshold7": 511,
              "I_Preamp": 100,
              "I_Ikrum": 30,
              "I_Shaper": 125,
              "I_Disc": 125,
              "I_Disc_LS": 100,
              "I_Shaper_test": 100,
              "I_DAC_DiscL": 53,
              "I_DAC_test": 100,
              "I_DAC_DiscH": 67,
              "I_Delay": 150,
              "I_TP_BufferIn": 128,
              "I_TP_BufferOut": 4,
              "V_Rpz": 255,
              "V_Gnd": 143,
              "V_Tp_ref": 120,
              "V_Fbk": 197,
              "V_Cas": 185,
              "V_Tp_refA": 50,
              "V_Tp_refB": 255}

RX3_SCAN = {"Threshold0": 1,
            "Threshold1": 2,
            "Threshold2": 3,
            "Threshold3": 4,
            "Threshold4": 5,
            "Threshold5": 6,
            "Threshold6": 7,
            "Threshold7": 8,
            "I_Preamp": 9,
            "I_Ikrum": 10,
            "I_Shaper": 11,
            "I_Disc": 12,
            "I_Disc_LS": 13,
            "I_Shaper_test": 14,
            "I_DAC_DiscL": 15,
            "I_DAC_test": 30,
            "I_DAC_DiscH": 31,
            "I_Delay": 16,
            "I_TP_BufferIn": 17,
            "I_TP_BufferOut": 18,
            "V_Rpz": 19,
            "V_Gnd": 20,
            "V_Tp_ref": 21,
            "V_Fbk": 22,
            "V_Cas": 23,
            "V_Tp_refA": 24,
            "V_Tp_refB": 25,
            "band_gap_output": 26,
            "band_gap_temperature": 27,
            "dac_bias": 28,
            "dac_cascade_bias": 29}

# enumerations for the types of medipix detectors that we have support for so far.
MXR_TYPE = 1
TPX_TYPE = 2
MP3_TYPE = 3
RX3_TYPE = 4
MXR_REVISION = 2
MP3_REVISION = 3
MP3p1_REVISION = 4
RX3_REVISION = 5
THLD0_NAME = {MXR_TYPE: "THLFine", TPX_TYPE: "THLFine", MP3_TYPE: "Threshold0", RX3_TYPE: "Threshold0"}
MAX_THLD = {MXR_TYPE: MXR_RANGE["THLFine"], TPX_TYPE: TPX_RANGE["THLFine"], MP3_TYPE: MP3_RANGE["Threshold0"], RX3_TYPE: RX3_RANGE["Threshold0"]}
RANGES = {MXR_TYPE: MXR_RANGE, TPX_TYPE: TPX_RANGE, MP3_TYPE: MP3_RANGE, RX3_TYPE: RX3_RANGE}
DAC_NAMES = {MXR_TYPE: MXR_DAC, TPX_TYPE: TPX_DAC, MP3_TYPE: MP3_DAC, RX3_TYPE: RX3_DAC}
DAC_VALUES = {MXR_TYPE: MXR_VALUES, TPX_TYPE: TPX_VALUES, MP3_TYPE:MP3_VALUES, RX3_TYPE: RX3_VALUES}
TYPE_STR = {MXR_TYPE: "MXR", TPX_TYPE: "TPX", MP3_TYPE: "MP3", RX3_TYPE: "3RX"}
REVISION_STR = {MXR_REVISION: "MXR", MP3_REVISION: "Medipix3.0", MP3p1_REVISION: "Medipix3.1", RX3_REVISION: "Medipix3RX"}
