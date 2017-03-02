####################################################################
# pyMarsCamera.py
#
#Read from the V5 camera, which is a network address as configured in the /etc/mars/config/dummy_camera.cfg
#network:
#                ip_addr,192.168.0.11
#
#
# On timings from oct 8 2014, DJS examined a tcp dump of two frames from the camera being transported to the ubuntu host.
# Generated a report the packet time and description with the explanation below
#
# time            description
# 0.1175 ub host sends read matrix command
# 0.1187 camera replies ack
# 0.1559 camera sends pixel matrix frame
# 0.1741    ub host sends ack
# 0.2075 ub host sends read matrix command
# 0.2085 camera sends ack
# 0.2447 camera sends pixel matrix frame
# 0.2669    ub host sends ack
#
# Discussion.
# There is 60ms between ub sending "read matrix command"    and ub sending ack to one incoming image frame.
#
# There is a 37 ms delay for the camera to acquire, build and begin sending the frame
#
# There is a 22 ms delay betweeen the start of frame sending, and the ub sending the ack.
# ===> Each frame is 60ms - acquire and transport time.
# ===> The ubuntu host took 30ms to handle the incoming frame in the above example...
#
# The main thread will initiate the sending of a packet. This raises an event on the self.sndEvent
# event class, which triggers the transmit thread. Measured a 20-30ms delay from when the main
# thread raises the event to whe packet is actually sent. This is python threading byting us.
#
#STANDALONE mode
#You can run this code in standalone mode, where it sets all dacs etc, grabs one frame, and displays it.
#
# #I (derek) ran this code with

#export PYTHONPATH=".:/home/derek/mercurial/MARS-interfaces/python"; python pyMarsCamera.py -s 1e-5
#
# and had a 10 microsecond long exposure. The frame was almost all zeros with the occasional pixel at a value of 1.
#
# Alternatively, you can run with the default exposure, 10 seconds and use
#
#export PYTHONPATH=".:/home/derek/mercurial/MARS-interfaces/python"; python pyMarsCamera.py
#
#At the time of writing the docs:::
#An exposure of less than 1e-7 takes essentially no time to acquire, however the image
#contains a values in the range of 0..4096. This is meaningless..
#
#Exposures longer than 43    seconds are disabled. It appears that the time used is exposure_time % 43
#Don't ask me why.
#Help:..
# you can run it with (assuming PYTHONPATH is already set) with
# -h (report help)
# -n (or --nox) and it just acquires an image and closes. During testing, ran this 1000s of time to test start/stop of code.
# -s SECS    (or --seconds SECS) to set the aquire time to SECS (the default is 10 secs)
####################################################################
from threading import Event as Event
from threading import Lock as Lock
import marsthread
from marsobservable import marsObservable
from DAC_config import RX3_SCAN, RX3_RANGE
import base64
import bitstring
import operator
import numpy
import socket
import sys
import datetime
import time
import struct
from libmarscamera_py import serialiseMatrix, deserialiseMatrix, demuxMatrix, deshiftMatrix
import matplotlib.pyplot as plt

hasCameraLock, takesCameraLock = marsthread.create_lock_decorators("_camera_lock")

# Gain modes
SuperHighGain = 0
HighGain = 2
LowGain = 1
SuperLowGain = 3



DACBITVALS = [30, 39, 48, 57, 66, 75, 84, 93, 102, 110, 118, 126, 134, 142, 150, 158, 166, 174, 182, 190, 198, 206, 214, 222, 230, 238, 247, 256]
DACNAMES = ["Threshold0", "Threshold1", "Threshold2", "Threshold3", "Threshold4", "Threshold5", "Threshold6", "Threshold7", "I_Preamp", "I_Ikrum", "I_Shaper", "I_Disc", "I_Disc_LS", "I_Shaper_test", "I_DAC_DiscL", "I_DAC_test", "I_DAC_DiscH", "I_Delay", "I_TP_BufferIn", "I_TP_BufferOut", "V_Rpz", "V_Gnd", "V_Tp_ref", "V_Fbk", "V_Cas", "V_Tp_refA", "V_Tp_refB"]

OMR_VALS = [0, 3, 4, 5, 7, 8, 9, 11, 14, 15, 18, 19, 20, 21, 22, 23, 28, 35, 37, 42, 47, 48]
OMRNAMES = ["M", "CRW_SRW", "Polarity", "PS", "Disc_CSM_SPM", "Enable_TP", "CountL", "ColumnBlock", "ColumnBlockSel", "RowBlock", "RowBlockSel", "Equalization", "ColourMode", "CSM_SPM", "InfoHeader", "FuseSel", "FusePulseWidth", "GainMode", "SenseDAC", "ExtDAC", "ExtBGSel"]

ALLOWED_OMR = ["Polarity", "Disc_CSM_SPM", "Enable_TP", "Equalization", "ColourMode", "CSM_SPM", "GainMode", "SenseDAC", "ExtDAC", "ExtBGSel"]

debug = "v5camera_debug" in sys.argv
#debug = False
if debug:
    print "V5 camera debugging selected"

print_pm_time="pm_time" in sys.argv

def dprint(*args):
    """
    Print only if debugging is enabled
    """
    if debug:
        for arg in args:
            sys.stdout.write(str(arg) + "\n")
        sys.stdout.flush()

#A decorator for a function, reports how long the function took.
def measure_duration(func):
    def f(*args, **kwargs):
        startTime = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            et = time.time() - startTime
            print "func %s took %8.5f seconds"%(str(func.__name__), et)

    f.__name__ = func.__name__
    return f

class marsCameraOMR(marsObservable):
    """
    An object to represent a Medipix OMR (Operation Mode Register).

    This is just a set of values with a few bits per value (typically 1) detailing
    settings of the Medipix chip. An example is CSM_SPM, which sets whether the chip is
    in Charge Summing Mode (CSM) or not (SPM). See the Medipix manual for more documentation.
    """
    def __init__(self, data_string):
        """
        Initialise the OMR, either as the 48-bit data it comes as from the Medipix chip,
        or as a dictionary of key-value for each setting.
        """
        if issubclass(type(data_string), str):
            if len(data_string) == 6:
                bitarr = bitstring.BitArray(length=48)
                bitarr.bytes = data_string
                bitarr = bitarr[::-1]
                for i, name in enumerate(OMRNAMES):
                    #print "setting omr values to:", name, bitarr[OMR_VALS[i]:OMR_VALS[i+1]][::-1]
                    setattr(self, name, bitarr[OMR_VALS[i]:OMR_VALS[i+1]][::-1])

        elif issubclass(type(data_string), dict):
            for i, name in enumerate(OMRNAMES):
                if name in data_string:
                    setattr(self, name, data_string[name])
                else:
                    setattr(self, name, 0)

    def build_OMR(self, safe=True):
        """
        Create the 48-bit Medipix-representation of the OMR from the internal fields
        of this object. Used for sending the values to the v5camera.
        """
        if safe:
            names = ALLOWED_OMR
        else:
            names = OMRNAMES

        bitarr = bitstring.BitArray(length=48)
        for key in names:
            i = OMRNAMES.index(key)
            if hasattr(self, key):
                val = getattr(self, key)
            else:
                val = 0
            try:
                bitarr[OMR_VALS[i]:OMR_VALS[i+1]] = val
                bitarr[OMR_VALS[i]:OMR_VALS[i+1]] = bitarr[OMR_VALS[i]:OMR_VALS[i+1]][::-1]
            except Exception as msg:
                #print msg
                bitarr[OMR_VALS[i]:OMR_VALS[i+1]] = 0

        #bitarr = bitarr[::-1]
        mask0 = (bitarr[0:8] + bitarr[8:16])
        mask1 = (bitarr[16:24] + bitarr[24:32])
        mask2 = (bitarr[32:40] + bitarr[40:48])
        # print "OMR bits set are:", mask0.bin, mask0.hex, mask1.bin, mask1.hex, mask2.bin, mask2.hex #"%04x"%(bitarr[0:16].uint), "%04x"%(bitarr[17:32].uint), "%04x"%(bitarr[33:48].uint)
        # print "build_OMR: %04x %04x %04x"%(mask0.uint, mask1.uint, mask2.uint)
        return mask0.uint, mask1.uint, mask2.uint
        #self.run_command("setOMRconfig", ["%04x"%(mask0.uint), "%04x"%(mask1.uint), "%04x"%(mask2.uint)])

class marsCameraInfoHeader(marsObservable):
    """
    An object for detailing the info header that comes back from the Medipix chip
    when InfoHeader=1 is enabled in the OMR during a matrix read.

    The info header typically contains the current OMR, the fuses from the chip, the
    DAC string, and a whole bunch of unused data. It is 256-bits long.

    TODO: Parse the DAC string part of the infoheader (am I remembering this wrong?)
    """
    def __init__(self, data_string):
        if issubclass(type(data_string), str):
            if len(data_string) == 32:
                omr_bytes = data_string[26:32]
                fuse_bytes = data_string[20:24]
                self.omr = marsCameraOMR(omr_bytes)
                self.fuse = struct.unpack('>I', numpy.frombuffer(fuse_bytes[::-1], dtype='uint8'))[0]


class marsCameraImage(marsObservable):
    """
    An object for representing Medipix images.
    """
    def __init__(self, data_string, deshift=True):
        if issubclass(type(data_string), str):
            if len(data_string) == 98304 + 56: # Multi frame acquire response
                #header
                assert(ord(data_string[0]) == 0xbb)
                assert(ord(data_string[1]) == 0xaa)
                assert(ord(data_string[2]) == 0xbb)
                assert(ord(data_string[3]) == 0xaa)
                #multiframe information
                self.timestamp = struct.unpack('>I', numpy.frombuffer(data_string[4:8][::-1], dtype='uint8'))[0]
                self.shutterduration =    struct.unpack('>I', numpy.frombuffer(data_string[8:12][::-1], dtype='uint8'))[0]
                self.frameindex =    struct.unpack('>I', numpy.frombuffer(data_string[12:16][::-1], dtype='uint8'))[0]
                self.framecount =    struct.unpack('>I', numpy.frombuffer(data_string[16:20][::-1], dtype='uint8'))[0]
                self.counter = data_string[20]
                if self.counter == 'l':
                    self.counter = 0
                elif self.counter == 'h':
                    self.counter = 1
                #network diagnostics
                self.pending_matrices = struct.unpack('>H', numpy.frombuffer(data_string[21:23][::], dtype='uint8'))[0]
                self.overflow_indicator = ord(data_string[23])
                #Medipix data
                id_bytes = data_string[24:56]
                self.id_info = marsCameraInfoHeader(id_bytes[0:32])
                data_string = data_string[56:]
            else:
                self.pending_matrices = self.overflow_indicator = self.timestamp = self.shutterduration = self.frameindex = self.framecount = self.idinfo = self.counter = None
            if len(data_string) == 98304: # Get the Medipix frame
                frame = numpy.frombuffer(data_string, dtype='uint8')
                print len(frame)
                print "------------------"
                print len(demuxMatrix(frame))
                print "------------------"
                print len(deserialiseMatrix(demuxMatrix(frame)))

                imageframe = deserialiseMatrix(demuxMatrix(frame)) # convert from the multiplexed serialised format to a 256x256 array

                new_frame = numpy.zeros([256, 256])

                for i in range(256 / 8): # some column rearrangements.
                    main_location = i * 8
                    new_frame[:, main_location:main_location+8] = imageframe[:, main_location:main_location+8][:,::-1]

                imageframe = new_frame.astype('uint16')
                if deshift:
                    self.shifted = False
                    imageframe = deshiftMatrix(imageframe)
                else:
                    self.shifted = True
                self.image = imageframe
            else:
                raise IOError("Invalid mask data length")
        elif issubclass(type(data_string), numpy.ndarray):
            self.image = data_string

    def get_image(self, deshift=False):
        """
        Get the image.

        Image might already be deshifted, in which case you do NOT want to do it again.
        """
        if deshift and self.shifted:
            return deshiftMatrix(self.image)
        else:
            return self.image

    def deshift_image(self):
        """
        Reverse the LFSR of each pixel's counts.
        """
        self.image = deshiftMatrix(self.image)

    def get_serialised_string(self):
        """
        Convert the Medipix image into the serialised version that upload_image requires.
        """
        imageframe = self.image.astype('uint16').reshape([256, 256])
        # Some weird shuffling is needed here, although may remove this as it could
        #    turn out to actually be needed in the readMatrix stuff.
        new_mask = numpy.zeros([256, 256])
        for i in range(256 / 8 / 4):
            main_location = i * 4 * 8
            for j in range(4):
                main_start = main_location + j * 8
                other_side_start = main_location + (3 - j) * 8
                new_mask[:,main_start:main_start+8] = imageframe[:,other_side_start:other_side_start+8][:,::-1]


        imageframe = new_mask.astype('uint16').flatten()
        configurationMatrix = serialiseMatrix(imageframe)
        return configurationMatrix


@hasCameraLock
@marsthread.hasRWLock
@marsthread.hasLock
class marsCameraClient(marsObservable):
    """
    Networking client for talking to the v5 camera, with many specific
        commands for this purpose.

    This implements both the message-handling interface to the V5 camera
        as well as the individual commands that are sent to the camera.

    The bias voltage control requires a separate instance of this object to
        be created, connected to the bias voltage server. Commands that
        would be sent to the camera can not be sent to the bias voltage
        server and vice verca. There is no regulation of this in the
        objects themselves.

    TODO: Regulate the use of high voltage/camera commands depending on
        what type of server the object is connected to.

    Ideally the objects should be created via the UDPListener class, which
        will automatically identify the servers that can be connected to
        and create the correct marsCameraClient for it.
    """

    def __init__(self, print_receive=False):
        self.print_receive = print_receive
        marsObservable.__init__(self)
        self.pg_time = time.time()
        self.connected = False
        self.socket = socket.socket()
        self.recv_buffer = {}
        self._msg_id = 0
        self.frame = [[],[]]
        self.imageframe = [[],[]]
        self.configurationmatrix = [[],[]]
        self.downloaded_image = [[],[]]
        self.multi_frames = []
        self.waiting_for_multi_frames = 0
        self.waiting_for_temperature = False
        self._multiframe_timeout = 1.0
        self.fail_count = 0
        self.temperature = 0
        self.mf_sock = None
        self.counter = 0
        self.acquiring = False
        self.tx_queue = []
        self.OMR = {}
        self.dacvals = {}
        self.waiting_for_readmatrix = False
        self.waiting_for_ADCs = False
        self.waiting_for_DAC_scan = False
        self.ADC_val = [0 for i in range(2, 14)]
        self.DAC_scan_vals = []
        self.fuses = 0
        self.extdac = 2048
        self.hv = 0
        self.read_hv_v = 0
        self.waiting_for_hv = False
        self.read_hv_a = 0
        self.waiting_for_hv_current = False
        self.extbg = 2048
        self.rcvEvent = Event()
        self.sndEvent = Event()
        self.threads_lock = Lock()

    def connect(self, address, port = 1234):
        """
        Open up the socket connection, setup networking threads and command callbacks
        """
        self.address = address
        self.port = port
        try:
            self.socket.connect((address, port))
        except:
            self.connected = False
            raise
        else:
            self.connected = True
            self.fail_count = 0
        #print "Set buffer size for send and receive to 256 K"
        try:
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        except Exception as msg:
            print "Exception on setting NODELAY " + str(msg)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256000)
        except Exception as msg:
            print "Exception on setting socket send buffer size to 256 K " + str(msg)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256000)
        except Exception as msg:
            print "Exception on setting socket receive buffer size to 256 K " + str(msg)

        marsthread.new_thread(self.recv)
        marsthread.new_thread(self.transmit)

        self.add_observer(self.shutter_done, key="shutterDone")
        self.add_observer(self.print_pm_received, key="pixelMatrix")
        self.add_observer(self.set_frame, key="pixelMatrix")

        self.add_observer(self.recv_ADCs, key="ADCs")
        #self.add_observer(self.print_msg, key="ADCs")
        self.add_observer(self.print_msg, key="unknownCommand")
        #self.add_observer(self.print_msg, key="testResults")
        self.add_observer(self.recv_DAC_scan, key="testResults")
        self.add_observer(self.recv_hv_v, key="hvVolts")
        self.add_observer(self.recv_hv_a, key="hvCurrent")
        #self.add_observer(self.print_msg, key="all")
        self.add_observer(self.recv_fuses, key="id")
        self.add_observer(self.recv_temperature, key="temperature")
        self.add_observer(self.read_dacs, key="DACs")
        #self.add_observer(self.print_msg, key="shutterDone")
        #self.initial_setup()
        return self.connected

    def disconnect(self):
        """
        Disconnect the socket, and set the flags to close the networking threads.
        """
        self.connected = False
        self.acquiring = False #If we have to close down while acquiring an image, we should stop waiting..
        self.rcvEvent.set()
        self.sndEvent.set()
        ##self.socket.close() - we can only do this if neither tx or rx is running. the send/receive threads might be reading or writing..
        self.threads_lock.acquire()
        if not (hasattr(self, "receive_running") or hasattr(self, "transmit_running")):
            self.socket.close()
        self.threads_lock.release()

    def print_msg(self, message_components):
        """
        A generic callback to print the network command received from the v5 readout.
        """
        #print ",".join(message_components)
        l = message_components[:-1]
        #print message_components[0][1:] + ": " + ',    '.join(["-".join(["0x%02X"%(ord(ch)) for ch in base64.b64decode(l_item)]) for l_item in l])
        print message_components[0][1:] + ": " + ','.join([base64.b64decode(l_item) for l_item in l[2:]])

    def recv_ADCs(self, message_components):
        """
        A callback for setting the ADC values from the ADC read command.
        """
        #print ",".join([base64.b64decode(x) for x in message_components[2:-1]])
        self.ADC_val = [int(base64.b64decode(message_components[i])) for i in range(2,len(message_components)-1)]
        if len(self.ADC_val) == 1:
            dtype = 'uint8'
        else:
            dtype = 'uint16'
        self.waiting_for_ADCs = False

    def recv_DAC_scan(self, message_components):
        """
        A callback for setting the full scan of ADC values from a DAC scan command.
        """
        datalines = (",".join([base64.b64decode(message_components[i]) for i in range(2, len(message_components) - 1)])).strip('\r\n').strip(',').split('\r\n')
        tmp = [[int(data) for data in dataline.split(',')] for dataline in datalines[1:]]
        self.DAC_scan_vals = [[tmp[j][i] for j in range(len(tmp))] for i in range(len(tmp[0]))] #transpose
        self.waiting_for_DAC_scan = False

    def recv_hv_v(self, message_components):
        """
        A callback for setting the high voltage from the read HV command.
        """
        self.read_hv_v = base64.b64decode(message_components[2])
        self.waiting_for_hv = False

    def recv_hv_a(self, message_components):
        """
        A callback for setting the leakage current from the read HV current command.
        """
        self.read_hv_a = base64.b64decode(message_components[2])
        self.waiting_for_hv_current = False

    def recv_fuses(self, message_components):
        """
        A callback for setting the fuses from the read fuses command.
        """
        IH_bytes = base64.b64decode(message_components[2])
        infoheader = marsCameraInfoHeader(IH_bytes)
        self.fuses = infoheader.fuse
        self.waiting_for_fuses = False

    def recv_temperature(self, message_components):
        """
        A callback for setting the temperature of the MitySOM FPGA sensor from the read temperature command
        """
        recv_temp = base64.b64decode(message_components[2])
        self.temperature = recv_temp
        self.waiting_for_temperature = False

    def read_dacs(self, message_components):
        """
        A callback for setting the current dacvals to what is on the sensor from a read_dacs command.
        """
        #print "Read_dacs " + ",".join(message_components)
        val = base64.b64decode(message_components[2])
        bitarr = bitstring.BitArray(bytes=val)[::-1]
        for i,dacname in enumerate(DACNAMES):
            bitval = bitarr[DACBITVALS[i]:DACBITVALS[i+1]][::-1]
            #print "print dacs:",dacname, ":", bitval.uint
            self.dacvals[dacname] = int(bitval.uint)
            dprint(dacname, ":", bitval.uint)

    @takesCameraLock
    def set_hv(self, hv):
        """
        Set the high voltage.

        If 0, turn it off, if > 0, turn it on.

        TODO: Use the correct polarity for the chip,
        instead of assuming its a CdTe/CZT/GaAs.
        """
        if self.connected:
            self.hv = hv
            self.run_command("hvSetPolarity", ["-"])
            if hv > 0:
                print "turning HV on with val", hv
                self.run_command("hvSetVolts", [str(hv)])
                self.run_command("hvOn", [])
            else:
                print "Turning hv off"
                self.run_command("hvOff", [])

    def get_hv(self):
        """
        Get the HV currently being set to the chip.
        """
        return self.hv

    @takesCameraLock
    def read_hv(self):
        """
        Read the HV from the ADC monitor.
        """
        if self.connected:
            self.waiting_for_hv = True
            self.run_command("hvReadVolts")
            while self.waiting_for_hv and self.connected:
                time.sleep(5e-3)
            return self.read_hv_v
        else:
            return 0.0

    @takesCameraLock
    def read_hv_current(self):
        """
        Read the leakage current of the HV circuit.
        """
        self.waiting_for_hv_current = True
        self.run_command("hvReadCurrent")
        while self.waiting_for_hv_current:
            time.sleep(5e-3)
        return self.read_hv_a

    @takesCameraLock
    def write_DACs(self, dacvals = {}, extdac=None, extbg=None):
        """
        Write the DACs to the Medipix sensor

        Any dacvalues supplied are set, but any not in the dictionary
        uses the internal values of the current object.
        """
        if extdac is None:
            extdac = self.extdac
        else:
            self.extdac = extdac
        if extbg is None:
            extbg = self.extbg
        else:
            self.extbg = extbg
        bitarr = bitstring.BitArray(length=256)
        for dacname in DACNAMES:
            i = DACNAMES.index(dacname)
            if dacname in dacvals:
                dprint ("Setting", dacname, DACBITVALS[i], DACBITVALS[i+1], int(dacvals[dacname]))
                #print "Setting dac " + dacname + " " + str(DACBITVALS[i]) + " " + str(DACBITVALS[i+1]) + " " + str(int(dacvals[dacname]))
                self.dacvals[dacname] = dacvals[dacname]
                val = int(dacvals[dacname])
            elif dacname in self.dacvals:
                val = int(self.dacvals[dacname])
            else:
                val = 128
            try:
                bitarr[DACBITVALS[i]:DACBITVALS[i+1]] = val
                bitarr[DACBITVALS[i]:DACBITVALS[i+1]] = bitarr[DACBITVALS[i]:DACBITVALS[i+1]][::-1]
            except Exception as msg:
                dprint(msg)
                bitarr[DACBITVALS[i]:DACBITVALS[i+1]] = 0


        bbytes = bitarr[::-1].bytes
        #print ",".join([hex(ord(bbytes[i])) for i in range(len(bbytes))])
        #print base64.b64encode(bbytes)
        #bbytes = bbytes[::-1]

        self.run_command("writeDACs", [str(extdac), str(extbg), bbytes])
        ##self.run_command("readDACs",    [])

    @takesCameraLock
    def write_OMR(self, omrvals = {}):
        """
        Set OMR fields to requested values.

        Note: Some OMR fields are not customisable
        from the user side of the software.

        Also note: setOMRconfig doesn't actually
        send the command to the Medipix chip, so
        a write_DACs is done here to make sure
        it is propagated.
        """
        bitarr = bitstring.BitArray(length=48)

        for key in ALLOWED_OMR:
            i = OMRNAMES.index(key)
            if key in omrvals:
                self.OMR[key] = omrvals[key]

        #sanity checks
        marsOMR = marsCameraOMR(self.OMR)
        tmp0, tmp1, tmp2 = marsOMR.build_OMR()
        # print "OMR bits set are:", mask0.bin, mask0.hex, mask1.bin, mask1.hex, mask2.bin, mask2.hex #"%04x"%(bitarr[0:16].uint), "%04x"%(bitarr[17:32].uint), "%04x"%(bitarr[33:48].uint)
        self.run_command("setOMRconfig", ["%04x"%(tmp0), "%04x"%(tmp1), "%04x"%(tmp2)])
        self.write_DACs({})    # The "setOMRconfig" doesn't actually write the OMR, just leaves it so the next command sets it off. Write the current DACs to allow the OMR to be processed.

    @property
    def msg_id(self):
        """
        Get a msg_id for the next message to send (just always increase by one so its unique).
        """
        self._msg_id += 1
        return self._msg_id

    @msg_id.setter
    def msg_id(self, val):
        raise LookupError("Not allowed to set the msg_id directly")

    @marsthread.reader
    def check_tx_queue(self):
        """
        For the send thread. Check if there is something to send.
        """
        return len(self.tx_queue) > 0

    @marsthread.modifier
    def pop_tx_queue(self):
        """
        Get the latest network command to send.
        """
        return self.tx_queue.pop(0)

    @marsthread.modifier
    def push_tx_queue(self, message):
        """
        Add a network command to send when the send thread is able to.
        """
        self.tx_queue.append(message)
        self.sndEvent.set()

    def transmit(self):
        """
        Transmit - to be run in a separate thread, and send messages on
            the socket to the V5 camera

        Repeatedly checks the transmit queue for messages, and puts them
            on to the socket.

        Keeps running until the "connected" variable is set false by
        the "disconnect" method.
        """
        self.threads_lock.acquire()
        self.transmit_running = True
        self.threads_lock.release()

        while self.connected:
            if not self.check_tx_queue():
                self.sndEvent.wait()
                self.sndEvent.clear()
            if not self.connected:
                break
            if self.check_tx_queue():
                dprint("about to send command")
                self.send_command(self.pop_tx_queue())
                dprint("command sent")

        self.threads_lock.acquire()
        if not hasattr(self, "receive_running"):
            self.socket.close()
        delattr(self, "transmit_running")
        self.threads_lock.release()

    def recv(self):
        """
        Recv - to be run in a separate thread and listen on the socket for
            messages from the V5 camera.

        It can receive three types of messages:
        1. Information packets, with headers back from the MARS camera about
            something useful. (e.g. the matrix back from a "downloadImage")
        2. Ack packets. Responses back from the V5 camera that a command has
            been received and processed correctly.
        3. Random information packets. For example when the high voltage is read,
            the serial port prints out some high voltage messages that do not
            follow the typical camera message structure. These types of
            messages are ignored. Random network packets that somehow get
            detected on the interface are also ignored by this.

        The thread should run continuously while "self.connected" is true, which
        is turned off only by the "disconnect" method.

        recv time for entire packet (since ! began) is    0.00687 for 131095 bytes
        recv time for entire packet (since ! began) is    0.00664 for 131095 bytes
        recv time for entire packet (since ! began) is    0.00402 for 131095 bytes = which is 8 ms to read in one frame
        """
        self.threads_lock.acquire()
        self.receive_running = True
        self.threads_lock.release()
        message = ""
        start_time = time.time()
        self.socket.settimeout(1.0)
        while self.connected:
            try:
                ch = self.socket.recv(64)
            except socket.timeout:
                time.sleep(5e-5)
            else:
                if self.print_receive:
                    print ch
                message += ch
                if "!" in ch:
                    start_time = time.time()
                if "\n" in ch:
                    try:
                        #print "recv time for entire packet (since ! began) is %8.5f for %d bytes"%((time.time() - start_time), len(message))## <<<Generates above timing figures.
                        #dprint(time.time() - self.pg_time, "recv time was:", time.time() - start_time, message[:40])
                        messages = message.split('\n')
                        try:
                            for msg in messages[:-1]:
                                if "!" in msg:
                                    try:
                                        self.add_msg_to_buffer(msg[msg.index('!'):]) ### This is 32 ms..
                                    except Exception as msg:
                                        print "Readout communication error. Returned message:", msg[msg.index("!"):]
                                else:
                                    print "Invalid data back on message:", msg
                        finally:
                            self.rcvEvent.set()
                            message = messages[-1]
                    except Exception as msg:
                        print "Error during receive:", str(msg)
                        messages = ""
                #     self.add_msg_to_buffer(message)
                #     message = ""
        self.threads_lock.acquire()
        if not hasattr(self, "transmit_running"):
            self.socket.close()
        delattr(self, "receive_running")
        self.threads_lock.release()

    def add_msg_to_buffer(self, message):
        """
        Called by the receive thread. Put received frame into a buffer.

        Main program thread reads the buffer in response to events.

        Non-ack commands are handled via the marsObserver methods, so a
            callback needs to be set up for the command. Otherwise
            an ack is acknowledged.

        Execution time of this method is:::
        Add Msg    0.02077 for 131072 bytes
        Add Msg    0.02562 for 131072 bytes
        Add Msg    0.02033 for 131072 bytes
        Add Msg    0.02191 for 131072 bytes
        Add Msg    0.01898 for 131072 bytes
        Add Msg    0.02014 for 131072 bytes ===>> which 20 milliseconds per frame
        This execution time is the time to to base64 decode, check checksum and extract to usable frame.
        """
        message_components = message.split(',')
        msg_id = int(message_components[1])
        message_name = message_components[0][1:]
        if "commandFailed" in message_name:
            print message_name + " " + message
        #print "RX:: " + datetime.datetime.now().strftime("%H:%M:%S.%f") + " " + message_name + " " + str(msg_id)
        if message_name != 'ack':
            try:
                self.possibly_send_ack(message_components)
            except Exception as msg:
                pass
            else:
                self.notify(message=message_components, key=message_name)
        else:
            dprint("\n", time.time() - self.pg_time, "Received:", message[:128])
            #print "Add msg to buffer for id " + str(msg_id) + " " + str(message_name)
            self._rwlock.acquireW()
            try:
                self.recv_buffer[msg_id] = message_components
            finally:
                self._rwlock.releaseW()

    def check_cs(self, message_components):
        """
        Check that the checksum in a message matches what is calculated from its data.

                Time required for this method to run, measured with two frames.
        0.01590 for a message of length 131093 bytes
        0.01373 for a message of length 131093 bytes, which is 16ms per frame.
        ===>>12 for the get_checksum code, and then 4ms for the splitting etc.
        """
        csmessage = ','.join(message_components[:-1]) + ','
        tmp = message_components[-1]
        their_cs = tmp[:tmp.index("#")]
        our_cs = self.get_checksum(csmessage)
        ret = our_cs == their_cs
        if ret == False:
            print "CHECKsums do not match. Sorry"
            raise IOError("checksums don't match: %s, %s"%(our_cs, their_cs))
        return ret

    def possibly_send_ack(self, message_components):
        """
        Send an ack to a received message, but only if its a valid message (e.g. checksum is correct)
        """
        if self.check_cs(message_components):
            msg_id = message_components[1]
            new_message = "!ack," + msg_id + ","
            new_message = new_message + self.get_checksum(new_message) + "#"
            ##print datetime.datetime.now().strftime("%H:%M:%S.%f") + "    tx ACK " + msg_id
            self.tx_command(new_message)
        else:
            #print "bad message received."
            raise IOError("Probably not the best error type")

    @takesCameraLock
    def acquire(self, exptime):
        """
        Expose the shutter manually, and wait for the acquire to finish.
        """
        self.acquiring = True
        self.run_command("openShutter", [str(int(exptime))])
        while self.acquiring:
            time.sleep(5e-4)

    @takesCameraLock
    def get_multiframes(self, while_waiting_fn=None):
        """
        Return the response from the last multiframe acquire, waiting
            if the multiframe response hasn't finished.

        while_waiting_fn is a polling function to call while waiting
            for the multiframes to finish. Typically scan_module's
            "check_scan_running" to allow scan aborts to run in the
            middle of a multiframe acquire without hanging.
        """
        TIME_COMMAND = False
        start_time = time.time()

        try:
            while self.waiting_for_multi_frames == 2 and time.time() - start_time < self.multiframe_timeout:
                time.sleep(1e-3)
                if while_waiting_fn is not None:
                    while_waiting_fn()
            if TIME_COMMAND:
                total_time = time.time() - start_time
                dprint("Took %0.2f (readout avg=%0.4f) seconds to finish multiframe acquire (%d frames, %d ms)"%(total_time, total_time / frame_count - float(exptime_ms) / 1000.0, frame_count, exptime_ms))
            while self.waiting_for_multi_frames == 1 and time.time() - start_time < self.multiframe_timeout:
                time.sleep(1e-3)
                if while_waiting_fn is not None:
                    while_waiting_fn()
            if time.time() - start_time > self.multiframe_timeout:
                print "v5camera.multiple_acquire. Time out during mask acquire, sending cancel"
                raise Exception
        except Exception as msg:
            self.run_command("abortMultipleReadout", [])
            self.waiting_for_multi_frames = 0
        if self.waiting_for_multi_frames == -1:
            self.run_command("abortMultipleReadout", [])
            self.waiting_for_multi_frames = 0

        return self.multi_frames

    @takesCameraLock
    def multiframe_acquire(self, exptime_ms, frame_count, spacing_time_ms=0, counters="both", sync_mode="standalone", parallelisation=8, while_waiting_fn=None):
        """
        Run a multiple acquire (possibly of frame_count=1) to get the data
            from the camera as quick as possible.

        First a socket is created for the connection to receive the multiple frames.

        Secondly a multipleReadout command is sent to the V5 camera server
        Options are:
            exptime_ms - the exposure time of each frame
            frame_count - the number of frames per counter to be read
            spacing_time_ms - the time from frame to frame (start of frame
                to next start of frame)
            counters - which counter to readout (typically "both")
            sync_mode - Used for multiple readouts, to determine if the
                synchronisation is master or slave. "standalone" to ignore.
            parallelisation - how many bits to simultaneously read from the
                Medipix chip at once (8 is max and only supported number currently).

        Also arguments to this function (but not the v5 camera's multiple acquire):
            hang - Whether to wait for the multiframes to finish and return
                them, or exit immediately and require a call to "get_multiframes"
            while_waiting_fn - A function to periodically call while waiting for
                the multiframes to finish acquiring and downloading.

        TODO: Use the IP address of the machine, instead of just assuming it is 192.168.0.1
        """
        hang = True
        if self.mf_sock is None:
            self.mf_sock = socket.socket()
            self.mf_sock.bind(('0.0.0.0', 7823))
            self.mf_sock.settimeout(5.0)
        self.multiframe_timeout = (exptime_ms + 225.0) / 1000.0 * frame_count * 2.0 + 2.0
        if frame_count > 1:
            dprint("multiframe-acquire called with exptime, frame_count, spacing_time, counters, sync_mode, parallel:", exptime_ms, frame_count, spacing_time_ms, counters, sync_mode, parallelisation)

        self.waiting_for_multi_frames = 2

        self.mf_start_time = time.time()
        self.run_command("multipleReadout", [frame_count, exptime_ms, '192.168.0.1', 7823, sync_mode, spacing_time_ms, counters, parallelisation])
        self.mf_sock.listen(1)

        if frame_count > 1:
            dprint("Getting socket for multiple readout")
        tmp_sock, tmp_addr = self.mf_sock.accept()

        try:
            tmp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024000)
        except Exception as msg:
            dprint("Exception on setting socket send buffer size to 1024 K " + str(msg))
        try:
            tmp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, frame_count * 2 * 1024000)
        except Exception as msg:
            dprint("Exception on setting socket receive buffer size to large amount " + str(msg))

        if frame_count > 1:
            dprint("starting new thread")
        marsthread.new_thread(self.read_multiframe, tmp_sock, frame_count, counters)
        if hang:
            self.tmp_sock = tmp_sock
            return self.get_multiframes(while_waiting_fn=while_waiting_fn)


    def cancel_multiframe(self):
        """
        Cancel the multiple readout command currently running on
            the v5 camera.
        """
        self.waiting_for_multi_frames = -1 # this triggers a cancel from the multiframe acquire command

    def read_multiframe(self, sock, frame_count, counters):
        """
        Read the multiple acquire data on a socket, and
            automatically parse it into a list of
            marsCameraImage objects.

        Parses the data in to marsCameraImages as it is
            downloading currently.

        TODO: A notify every X frames to allow the GUI to
            update the image as the multiple acquire is
            running.
        """
        # header, timestamp, shutter, frameindex, framecount, counter, unused, sensorId, pixelmatrix

        self.multi_frames = []
        try:
            FRAMELENGTH = 4 + 4 + 4 + 4 + 4 + 1 + 3 + 32 + 98304
            frame_data = ""
            ctr = 0
            read_amount = 0
            sock.settimeout(5.0)
            if counters == "both":
                total_len = FRAMELENGTH * frame_count * 2
                total_frames = frame_count * 2
            else:
                total_len = FRAMELENGTH * frame_count
                total_frames = frame_count
            ctr_timeout = time.time()
            while len(frame_data) < total_len and self.waiting_for_multi_frames:
                try:
                    ch = sock.recv(1024)
                except socket.timeout:
                    dprint("Socket timeout on multiframe read")
                    break
                if len(ch) > 0:
                    frame_data += ch
                    read_amount += len(ch)
                if read_amount >= FRAMELENGTH:
                    read_amount -= FRAMELENGTH
                    ctr += 1
                    ctr_timeout = time.time()
                    try:
                        frame = marsCameraImage(frame_data[FRAMELENGTH*(ctr-1):FRAMELENGTH*ctr])
                    except AssertionError:
                        dprint("Exception in frame header at frame_index=", ctr-1)
                    self.multi_frames.append(frame)
                    if ctr%20 == 0: # show progress
                        sys.stdout.write('.')
                        sys.stdout.flush()
                if time.time() - ctr_timeout > 5.0:
                    dprint("Timeout since last image read (images read = %d)"%(ctr))
                    break

            if frame_count > 1:
                dprint("finished receiving frames at:", time.time(), time.time() - self.mf_start_time)


            if self.waiting_for_multi_frames == 2:
                self.waiting_for_multi_frames = 1

            # Commented out: code for parsing the marsCameraImages at the end of the multiple Acquire. Might
            # still be needed if the "on the fly" code slows down the system too much.
            #
            # self.missing_frames = ["c%d-%d"%(counter, 2*i+counter) for counter in range(2) for i in range(frame_count)]
            # for i in range(ctr):
            #     if len(frame_data) < FRAMELENGTH*(i+1):
            #         frame = marsCameraImage(numpy.zeros([256, 256]))
            #         frame.counter = divmod(ctr, 2)
            #     else:
            #         try:
            #             frame = marsCameraImage(frame_data[FRAMELENGTH*i:FRAMELENGTH*(i+1)])
            #         except AssertionError:
            #             dprint("Exception in frame header at frame_index=", i)
            #         if i % 50 == 3:
            #             dprint(i, "frame stats:", frame.pending_matrices, frame.highwatermark, frame.overflow_indicator)
            #         try:
            #             self.missing_frames.remove("c%d-%d"%(frame.counter, frame.frameindex))
            #         except Exception as msg:
            #             dprint("unable to remove field from missing_frames:", msg)

            #     self.multi_frames.append(frame)
            # if len(self.missing_frames) > 0:
            #     dprint("Missing frames from scan are:", self.missing_frames)


        finally:
            if self.waiting_for_multi_frames != -1:
                self.waiting_for_multi_frames = 0


    def shutter_done(self, message):
        """
        Callback that happens when an acquire / shutter finishes.

        Just sets the "acquiring" variable to False so that the
            acquire method knows when to exit.
        """
        self.acquiring = False
        #    self.download_image(0)
        #    self.download_image(1)

    @takesCameraLock
    def download_image(self, counter):
        """
        Manually download a counter from the Medipix into local memory.
        """
        MATRIX_READ_TIMEOUT = 10.0
        t = time.time()
        self.pm_start_time = t
        self.waiting_for_readmatrix = True
        self.counter = counter
        if counter == 0:
            self.run_command("readMatrix", ["L"])
        else:
            self.run_command("readMatrix", ["H"])

        while time.time() - t < MATRIX_READ_TIMEOUT and self.waiting_for_readmatrix:
            time.sleep(5e-4)

        if time.time() - t > MATRIX_READ_TIMEOUT:
            self.waiting_for_readmatrix = False
            print("Error receiving pixel matrix... timeout. Waited %8.5f seconds."%(time.time() - t))

    @takesCameraLock
    def get_id(self):
        """
        Get the fuses from the Medipix chip.
        """
        ID_READ_TIMEOUT = 5.0
        t = time.time()
        self.waiting_for_fuses = True
        self.run_command("getID", [])
        while time.time() - t < ID_READ_TIMEOUT and self.waiting_for_fuses:
            time.sleep(5e-4)

        if time.time() - t > ID_READ_TIMEOUT:
            self.waiting_for_fuses = False
            print "Error receiving fuses.. timeout ***************************************************"

        #print "fuses received are:", self.fuses
        return self.fuses # TODO get the right ones here

    @takesCameraLock
    def get_temperature(self):
        """
        Read the temperature off the MitySOM FPGA sensor.
        """
        TEMPERATURE_READ_TIMEOUT = 1.0
        t = time.time()
        self.waiting_for_temperature = True
        self.run_command("getTemperature", [])
        while time.time() - t < TEMPERATURE_READ_TIMEOUT and self.waiting_for_temperature:
            time.sleep(5e-4)

        if time.time() - t > TEMPERATURE_READ_TIMEOUT:
            self.waiting_for_temperature = False
            print "Error receiving temperature... timeout *******************************************"

        dprint("MitySOM Temperature read: ", self.temperature)

        return float(self.temperature)

    @takesCameraLock
    def read_ADC(self, dac_name=None):
        """
        Read all the ADCs. Store them all into ADC_val, but only return
            the ADC for the SenseDAC on the Medipix chip.

        Has the support to set the SenseDAC for the requested DAC,
            however    if dac_name is None then it will use the
            currently configured SenseDAC.
        """
        ADC_READ_TIMEOUT = 5.0
        t = time.time()
        self.waiting_for_ADCs = True
        if dac_name is not None:
            for key in RX3_SCAN.keys():
                if dac_name.lower() == key.lower():
                    self.write_OMR({"SenseDAC":RX3_SCAN[key]})
                    break
        self.run_command("getADCs", [])
        while time.time() - t < ADC_READ_TIMEOUT and self.waiting_for_ADCs:
            time.sleep(5e-4)

        if time.time() - t > ADC_READ_TIMEOUT:
            self.waiting_for_ADCs = False
            print "Error receiving ADCs.. timeout ******************************************************"

        #print str(dac_name) + " has a value of " + str(self.ADC_val)
        return self.ADC_val[11]

    @takesCameraLock
    def run_dac_scan(self, dacname):
        """
        The V5 camera readout has a DAC scan command, for iterating a single DAC
            over its full range of values and receiving the ADC results for
            each value. Due to the lack of network latency this is considerably
            faster than doing it by hand.

        This method runs that DAC scan on the camera.
        """
        DAC_SCAN_TIMEOUT = 15.0
        t = time.time()
        self.waiting_for_DAC_scan = True

        dac_index = RX3_SCAN[dacname]
        self.run_command("test", ["dacScan", "linear", dac_index])

        while time.time() - t < DAC_SCAN_TIMEOUT and self.waiting_for_DAC_scan:
            time.sleep(5e-4)

        if time.time() - t > DAC_SCAN_TIMEOUT:
            self.waiting_for_DAC_scan = False
            print "Error receiving ADCs via DAC scan.. timeout *****************"

        return self.DAC_scan_vals

    def get_power_supply_voltages(self, read=False):
        """
        When read_ADC is called, it gets data from all the ADCs, but only
        returns the SenseDAC ADC from the Medipix. Internally it still
        stores the ADCs for the Medipix supply voltages.

        This method returns the Medipix supply voltages from those ADCs.

        if read is True, then it will run the read_ADC command first.
        """
        if read:
            self.read_ADC()
        return self.ADC_val[8:11]

    @takesCameraLock
    def upload_image(self, imageframe, counter = 1):
        """
        Upload the image of the configuration matrix to the Medipix detector.

        The configuration matrix is the per-pixel DAC settings that turn off
            and on pixels in the Medipix detector, and equalises their
            energy offsets so that they all respond at roughly equal
            energy settings.

        In the Medipix3RX, only counterH has valid data, however I still
            think both should be set (counterL to all 0s)
        """
        image = marsCameraImage(imageframe)
        configurationMatrix = image.get_serialised_string()
        self.imageframe[counter] = imageframe
        self.configurationmatrix[counter] = configurationMatrix

        if counter == 0:
            self.run_command("writeMatrix", ["L", configurationMatrix])
        else:
            self.run_command("writeMatrix", ["H", configurationMatrix])

    def set_frame(self, message_components):
        """
        Callback from the download_image function, which sets the frame from the data received.

        This is only valid for the original "acquire" command, not the multiple Acquire command.

        I suspect these measurement numbers are out of date:

        How long does this take to run. Measure for four different frames
        Set frame took    0.00672 seconds for 131093 bytes
        Set frame took    0.01171 seconds for 131093 bytes
        Set frame took    0.00651 seconds for 131093 bytes
        Set frame took    0.00694 seconds for 131093 bytes                --- which means 10ms """
        try:
            framedata = base64.b64decode(message_components[2])
            imageframe = marsCameraImage(framedata, deshift=False)
            self.downloaded_image[self.counter] = imageframe.get_image()

            self.frame[self.counter] = imageframe.get_image(deshift=True)
        except Exception as msg:
            print "set frame failed:", msg
        finally:
            #print "Frame set with mean (%5.3f), min(%d) and max(%d)"%(self.frame[self.counter].mean(), self.frame[self.counter].min(), self.frame[self.counter].max())
            self.waiting_for_readmatrix = False

    def print_pm_received(self, message):
        """
        A callback to print the time that the pixel matrix was received,
            for timing of the readout delays. For debugging purposes
            only and not currently used.
        """
        if print_pm_time:
            elapsed = time.time() - self.pm_start_time
            nBytes = len(message[2])
            # print ("\n" + "%5.3f"%elapsed + " seconds to receive pixel matrix of %d bytes"%nBytes)    + "     ==>Mbits/sec    = %6.3f"%((nBytes*8)/(elapsed*1e6))

    @marsthread.reader
    def check_msgid_on_buffer(self, msg_id):
        """
        Check that a response has been put in the receive
            buffer for a given msg_id (e.g. an Ack for
            the msg_id of a command you've sent).
        """
        return msg_id in self.recv_buffer.keys()

    @marsthread.modifier
    def get_msg_from_buffer(self, msg_id):
        """
        Get the msg_id from the receive buffer

        Note: This should only ever get called after
        check_msgid_on_buffer has already declared that
        it is safe to do so, and should always only get
        called from the same thread.
        """
        return self.recv_buffer.pop(msg_id)

    def build_message(self, name, params):
        """
        Build the readout network message in the correct format.

        !<command_name>,base64(param1),base64(param2),...,checksum#
        """
        ##print "Build " + name + " " + str(self.msg_id)
        message = "!" + str(name) + "," + str(self.msg_id) + ","
        #print message, str(params)
        for param in params:
            #if isinstance(param, type("")) and len(param) == 1:
            #    message += param + ","
            #else:
            if issubclass(type(param), numpy.ndarray):
                message += base64.b64encode(param) + ","
            else:
                message += base64.b64encode(str(param)) + ","

        message += self.get_checksum(message) + "#"

        return message

    def slow_checksum(message):
        """
        Build a checksum, but not in the fastest way.

        A lovely comment from DJS here on the usefulness of the networking protocol:

        A one line calculation of checksum, that is 18-20ms, instead of 12ms below. If you want faster checksums, use C.
        However, since the TCP code in the kernel does checksum protection, what is the point of this code?
        """
        cs = reduce(operator.xor, (ord(s) for s in message), 0)
        return "%02X"%(cs)

    def get_checksum(self, message):
        """
        Builds a checksum from a message.

        It is worth noting that the multiframe acquire code does not use this, however we may be able to
        get some speedup by doing this in C as DJS says.

        Used to process all frames. Time required from runs by DJS on old and tired PC.
        0.01178 for a message of length 131089 bytes
        0.01177 for a message of length 131089 bytes     - which is 12 milliseconds per frame.
        """
        cs = 0
        for ch in message:
            cs ^= ord(ch)
        return "%02X"%(cs)

    def run_command(self, name, params = []):
        """
        Build and push a command to the readout to be processed.

        Arguments:
            name - the name of the command you wish to run on the v5camera
            params - a list of each parameter you want to send with the command

        This is the command you use for generic testing via a console.
        """
        message = self.build_message(name, params)
        dprint(message[0:100])
        print "MESSAGE:", message[0:100]
        self.push_tx_queue(message)

    def noisefloor_scan(self, values=range(0, 100, 2)):
        """
        A quick and dirty noisefloor routine with the V5camera.

        Only used for debugging.
        """
        frames = []
        self.acquire(5)
        self.acquire(5)
        for i, val in enumerate(values):
            self.write_DACs({"Threshold0":511})
            self.write_DACs({"Threshold0":val})
            self.acquire(5)
            self.download_image(0)
            frames.append(self.frame[0].copy())
            dprint("Threshold 0 set to", val, "with frame mean", self.frame[0].mean(), "and pixels counting:", (self.frame[0] > 10).sum())

        self.last_frames = numpy.array(frames)



    @marsthread.takesLock
    def tx_command(self, message):
        """
        Actually transmit the message to the readout.

        Message must be in the correct format for the v5 camera server.
        """
        self.socket.settimeout(1.0)
        ##print "TX:: " + datetime.datetime.now().strftime("%H:%M:%S.%f") + "    sending " + message[:90]
        if self.connected:
            self.socket.sendall(message)

    @marsthread.takesLock
    def send_command(self, message, tries=0):
        """
        Send a network message to the readout.

        Check for ACKs to confirm it was sent, and retry on a fail.

        Retries by recursively calling itself, labelling how many tries it is up to.
        """
        TIMEOUT = 1.0*2
        MAX_TRIES = 5
        start_time = time.time()
        self.tx_command(message)
        msg_id = int(message.split(',')[1])
        while self.connected and (self.check_msgid_on_buffer(msg_id) is False and time.time() - start_time < TIMEOUT):
            #sys.stdout.write('.')
            #sys.stdout.flush()
            time.sleep(5e-3)

        if time.time() - start_time >= TIMEOUT:
            if (tries < MAX_TRIES) and self.connected:
                ret = self.send_command(message, tries = tries + 1)
            else:
                print "V5 camera.send_command. Failed too many times. Aborting.", message
                self.fail_count += 1
                if self.fail_count >= 3:
                    print "V5 camera.send_command. Aborted too many commands due to failure. Disconnecting connection."
                    self.disconnect()
                return None
        else:
            if not self.connected:
                return None

            if self.check_msgid_on_buffer(msg_id):
                ret = self.get_msg_from_buffer(msg_id)
                #print "RX:: " + datetime.datetime.now().strftime("%H:%M:%S.%f") + "    receive " + ret[0] + " " + ret[1]
            else:
                self.rcvEvent.wait()
                self.rcvEvent.clear()
                ret = self.get_msg_from_buffer(msg_id)
                #print "RX:: " + datetime.datetime.now().strftime("%H:%M:%S.%f") + "    receive " + ret[0] + " " + ret[1]
            try:
                cs_true = self.check_cs(ret)
            except Exception as msg:
                print "Exception on checksum " + str(msg)
                dprint("Failure to check checksum.", msg)
                cs_true = False
            if not cs_true:
                if (tries < MAX_TRIES) and self.connected:
                    dprint("bad checksum on ack, retrying")
                    print "bad checksum on ack - retrying " + str(type(ret))
                    ret =    self.send_command(message, tries = tries + 1)
                else:
                    dprint("V5 camera.send_command. Failed too many times. Aborting")
                    self.fail_count += 1
                    if self.fail_count >= 3:
                        print "V5 camera.send_command. Aborted too many commands due to failure. Disconnecting connection."
                        self.disconnect()
                    return None
        return ret

    def test_mask_read_write(self, mask, counter=0):
        """
        A test function

        Writes the given mask to the Medipix chip, and
            then reads it back.

        Returns True if all pixels are the same, and False if not.
        """
        self.upload_image(mask, counter=counter)
        self.download_image(counter=counter)
        image = self.downloaded_image[counter]
        total = (image.flatten().astype('uint16') != mask.flatten().astype('uint16')).sum()
        if total != 0:
            print "Error: ", total, " pixels did not match from mask read write test."
        return total == 0


    def generate_mask(self, pixel_pattern, spacing):
        """
        Generates the masks that are used for testing matrix reads and writes

        A value for each pixel, and a possible spacing to make sure the columns/rows are done right.

        Spacing can be a tuple to make the row/columns have different patterns.
        """
        mask = numpy.zeros([256, 256]).astype('uint16')
        if isinstance(spacing, int):
            mask[::spacing, ::spacing] = pixel_pattern
        elif isinstance(spacing, tuple):
            mask[::spacing[0], ::spacing[1]] = pixel_pattern
        return mask

    def tune_DAC_from_ADC(self, dacname, dacval):
        """
        Calculate a DAC value by setting it to a requested ADC value.

        Uses a binomial algorithm to find the correct value.
        """
        top_val = RX3_RANGE[dacname] - 1
        bot_val = 0
        while top_val - bot_val > 1:
            #print "Write_DACS " + dacname + " " + str(top_val)
            self.write_DACs({dacname:top_val})
            v_top_val = self.read_ADC(dacname)
            #print "Write_DACS " + dacname + " " + str(bot_val)
            self.write_DACs({dacname:bot_val})
            v_bot_val = self.read_ADC(dacname)
            if abs(dacval - v_bot_val) < abs(dacval - v_top_val):
                top_val = (int((top_val + bot_val)/2.0))
            else:
                bot_val = (int((top_val + bot_val)/2.0))

        self.write_DACs({dacname:top_val})
        v_top_val = self.read_ADC(dacname)
        self.write_DACs({dacname:bot_val})
        v_bot_val = self.read_ADC(dacname)

        if abs(dacval - v_bot_val) < abs(dacval - v_top_val):
            return bot_val
        else:
            return top_val

    def tune_DACs_from_ADCs(self, dac_dict):
        """
        Perform the DAC tuning for multiple DACs at once.

        dac_dict should contain a dictionary of DACs to mV readings,
            and this function will aim to make each DAC match
            that mV reading on the ADC.
        """
        for dac in dac_dict.keys():
            self.tune_DAC_from_ADC(dac, dac_dict[dac])

    def initial_setup(self):
        """
        Startup procedures for the best usage of the Medipix with the v5 readout.
        """
        self.run_command("writeMedipixSupplies", ["1510", "1510", "2510"])     #Set voltage for VDD, VDDA, VDDD, which should be measured at 1.5, 1.5 and 2.5
        self.write_OMR({"GainMode":0, "Disc_CSM_SPM":0, "Polarity":0, "Equalization":0, "ColourMode":0, "CSM_SPM":0, "ExtDAC":0, "ExtBGSel":0, "EnableTP":0, "SenseDAC":0})
        self.run_command("hardwareReadout", ["T"])
#        self.write_DACs({'V_Rpz': 255, 'V_Tp_refA': 50, 'V_Tp_refB': 255, 'I_Shaper_test': 100, 'I_DAC_test': 100, 'V_Cas': 184, 'V_Tp_ref': 120, 'I_Ikrum': 30, 'Threshold1': 300, 'Threshold0': 30, 'Threshold3': 300, 'Threshold2': 300, 'Threshold5': 300, 'Threshold4': 300, 'Threshold7': 300, 'Threshold6': 300, 'I_Disc_LS': 100, 'I_DAC_DiscL': 64, 'I_DAC_DiscH': 69, 'I_Disc': 125, 'I_Shaper': 200, 'I_TP_BufferOut': 4, 'V_Fbk': 181, 'V_Gnd': 141, 'I_TP_BufferIn': 128, 'I_Delay': 30, 'I_Preamp': 250}, extdac=2000, extbg=2000)
        self.write_DACs({'V_Rpz': 255, 'V_Tp_refA': 50, 'V_Tp_refB': 255, 'I_Shaper_test': 100, 'I_DAC_test': 100, 'V_Cas': 174, 'V_Tp_ref': 120, 'I_Ikrum': 30, 'Threshold1': 50, 'Threshold0': 30, 'Threshold3': 50, 'Threshold2': 50, 'Threshold5': 50, 'Threshold4': 300, 'Threshold7': 300, 'Threshold6': 300, 'I_Disc_LS': 100, 'I_DAC_DiscL': 64, 'I_DAC_DiscH': 69, 'I_Disc': 125, 'I_Shaper': 200, 'I_TP_BufferOut': 4, 'V_Fbk': 177, 'V_Gnd': 135, 'I_TP_BufferIn': 128, 'I_Delay': 30, 'I_Preamp': 150}, extdac=2048, extbg=2048)
        self.upload_image(numpy.zeros([256, 256]), counter=0)
        self.upload_image(numpy.zeros([256, 256]), counter=1)

    def test_matrix_read_write(self):
        """
        A full suite of read/write tests on the pixel matrix of the v5 camera.

        Tests that a) the read and write are workign correctly and b) that row/columns are being done correctly.

        pixel_patterns is what gets applied to the tested pixels.
        Spacings determines the gaps between pixels that are active for the test.
        """
        pixel_patterns = [0x0, 0xfff, 0xaaa, 0x555, 0xf00, 0x0f0, 0x00f, 0x333, 0x666, 0x777, 0x700, 0x070, 0x007, 0x800, 0x080, 0x008, 0x100, 0x010, 0x001]
        spacings = [1, 2, (7,11)]
        counters = [0, 1]
        success = 0
        total = 0
        exc_count = 0
        img_count = 0
        for spacing in spacings:
            for pix_pat in pixel_patterns:
                print "Testing pix_pat: 0x%03X"%(pix_pat), "spacing: ", spacing
                mask = self.generate_mask(pix_pat, spacing)
                for counter in counters:
                    try:
                        if self.test_mask_read_write(mask, counter):
                            success += 1
                        img_count = img_count + 1
                        if img_count - success > 5:
                            print "Pixel matrix read write test: Aborting due to too many failures"
                            return False
                    except Exception as msg:
                        exc_count = exc_count + 1
                        if exc_count > 5:
                            raise
                        print "Exception in testing the individual mask:", msg
                    total += 1

        print "%0.1f%% of read write tests passed (%d / %d)"%((100.0 * float(success) / float(total)), success, total)
        return success == total

    def test_adc_tuning(self):
        """
        Test the key dacs that require tuning actually meet their required values.
        """
        final_voltages = {"V_Cas":850, "V_Gnd":650, "V_Fbk":850}
        print "V5 camera tuning voltages to:", final_voltages
        self.tune_DACs_from_ADCs(final_voltages)
        total_error = 0.0
        for dacname in final_voltages.keys():
            dacval = self.dacvals[dacname]
            v_dacval = self.read_ADC(dacname)
            error = abs(float(v_dacval) - float(final_voltages[dacname])) / float(final_voltages[dacname]) * 100
            print dacname, " measured at ", v_dacval, "mV (error = %0.1f%%)"%(error)
            total_error += error

        print "Average error was: %0.1f%%"%(total_error / len(final_voltages))

        return (error / len(final_voltages)) < 1.0

    def test_get_id(self):
        """
        Test the fuse retrieval works.

        TODO: confirm whether it was received or not.
        """
        self.run_command("getID", [])
        return True

    def key_press(self, *args):
        """
        Disconnect on a key-press.. only used for the ifdef==main stuff below.
        """
        self.disconnect()

    def get_frame(self, counter):
        return self.frame[counter].tostring()
    
    """
    For debagging, normally commented
    """
    #def test_print(smth):
    #print smth

    def destroy(self, *args):
        """
        A destroy/disconnect method.. only used for the ifdef==main stuff below.
        """
        print "Initiate close now."
        self.disconnect()
        time.sleep(1.0)

# automatic locks for making variables thread-safe.
marsthread.create_RWlock_variable(marsCameraClient, "_waiting_for_readmatrix", "waiting_for_readmatrix")
marsthread.create_RWlock_variable(marsCameraClient, "_multiframe_timeout", "multiframe_timeout")
marsthread.create_RWlock_variable(marsCameraClient, "_acquiring", "acquiring")
marsthread.create_RWlock_variable(marsCameraClient, "_waiting_for_ADCs", "waiting_for_ADCs")
marsthread.create_RWlock_variable(marsCameraClient, "_waiting_for_DAC_scan", "waiting_for_DAC_scan")
marsthread.create_RWlock_variable(marsCameraClient, "_waiting_for_fuses", "waiting_for_fuses")
marsthread.create_RWlock_variable(marsCameraClient, "_waiting_for_multi_frames", "waiting_for_multi_frames")
marsthread.create_RWlock_variable(marsCameraClient, "_waiting_for_temperature", "waiting_for_temperature")
marsthread.create_RWlock_variable(marsCameraClient, "_temperature", "temperature")
marsthread.create_RWlock_variable(marsCameraClient, "_multi_frames", "multi_frames")

client = marsCameraClient()
client.connect('192.168.0.44')
client.acquire(3000)
time.sleep(1)
client.download_image(1)
time.sleep(1)

win = gtk.Window(gtk.WINDOW_TOPLEVEL)
win.set_title("Exposure " + str(args.seconds) + " seconds")
win.connect("destroy", client.destroy)
win.connect("key_press_event", client.key_press)
htbox = gtk.HBox()
image_display = av.Image_Display(client.frame[1], title="Image")
htbox.pack_start(image_display)
win.add(htbox)
win.show_all()
print "Examine with resize etc."
gtk.main()
client.disconnect()