####################################################################
# pyMarsCamera.py
#
#Read from the V5 camera, which is a network address as configured in the /etc/mars/config/dummy_camera.cfg
#network:
#        ip_addr,192.168.0.11
#
#
# On timings from oct 8 2014, DJS examined a tcp dump of two frames from the camera being transported to the ubuntu host.
# Generated a report the packet time and description with the explanation below
#
# time      description
# 0.1175 ub host sends read matrix command
# 0.1187 camera replies ack
# 0.1559 camera sends pixel matrix frame
# 0.1741  ub host sends ack
# 0.2075 ub host sends read matrix command
# 0.2085 camera sends ack
# 0.2447 camera sends pixel matrix frame
# 0.2669  ub host sends ack
#
# Discussion.
# There is 60ms between ub sending "read matrix command"  and ub sending ack to one incoming image frame.
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
#Exposures longer than 43  seconds are disabled. It appears that the time used is exposure_time % 43
#Don't ask me why.
#Help:..
# you can run it with (assuming PYTHONPATH is already set) with
# -h (report help)
# -n (or --nox) and it just acquires an image and closes. During testing, ran this 1000s of time to test start/stop of code.
# -s SECS  (or --seconds SECS) to set the aquire time to SECS (the default is 10 secs)
####################################################################
from threading import Event as Event
from threading import Lock as Lock
from marsct import marsthread
from marsct import marslog as ml
marslog = ml.logger(source_module="pyMarsCamera", group_module="camera")
from marsct.marsobservable import marsObservable
from marsct.DAC_config import RX3_SCAN, RX3_RANGE
import base64
import bitstring
import operator
#import gtk
import numpy
#import pygtk
import socket
import sys
import datetime
import time
import struct
from libmarscamera_py import serialiseMatrix, deserialiseMatrix, demuxMatrix, deshiftMatrix
import matplotlib.pyplot as plt

# Gain modes
SuperHighGain = 0
HighGain = 2
LowGain = 1
SuperLowGain = 3

MF_PORTS = []

MULTIFRAME_IDLE, MULTIFRAME_RUNNING, MULTIFRAME_TRIGGER_ABORT, MULTIFRAME_ABORTING, MULTIFRAME_FAILED_START = range(5)

def get_mf_port():
	global MF_PORTS
	if len(MF_PORTS) == 0:
		new_port = 7823
	else:
		new_port = max(MF_PORTS) + 1
	MF_PORTS.append(new_port)
	return new_port

hasSocketLock, takesSocketLock = marsthread.create_lock_decorators("_socket_lock")


DACBITVALS = [30, 39, 48, 57, 66, 75, 84, 93, 102, 110, 118, 126, 134, 142, 150, 158, 166, 174, 182, 190, 198, 206, 214, 222, 230, 238, 247, 256]
DACNAMES = ["Threshold0", "Threshold1", "Threshold2", "Threshold3", "Threshold4", "Threshold5", "Threshold6", "Threshold7", "I_Preamp", "I_Ikrum", "I_Shaper", "I_Disc", "I_Disc_LS", "I_Shaper_test", "I_DAC_DiscL", "I_DAC_test", "I_DAC_DiscH", "I_Delay", "I_TP_BufferIn", "I_TP_BufferOut", "V_Rpz", "V_Gnd", "V_Tp_ref", "V_Fbk", "V_Cas", "V_Tp_refA", "V_Tp_refB"]

OMR_VALS = [0, 3, 4, 5, 7, 8, 9, 11, 14, 15, 18, 19, 20, 21, 22, 23, 28, 35, 37, 42, 47, 48]
OMRNAMES = ["M", "CRW_SRW", "Polarity", "PS", "Disc_CSM_SPM", "Enable_TP", "CountL", "ColumnBlock", "ColumnBlockSel", "RowBlock", "RowBlockSel", "Equalization", "ColourMode", "CSM_SPM", "InfoHeader", "FuseSel", "FusePulseWidth", "GainMode", "SenseDAC", "ExtDAC", "ExtBGSel"]

ALLOWED_OMR = ["Polarity", "Disc_CSM_SPM", "Enable_TP", "Equalization", "ColourMode", "CSM_SPM", "GainMode", "SenseDAC", "ExtDAC", "ExtBGSel"]


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
		marsObservable.__init__(self)
		self.marslog = ml.logger(source_module="pyMarsCamera", group_module="camera", class_name="marsCameraOMR", object_id=self)
		if issubclass(type(data_string), str):
			if len(data_string) == 6:
				bitarr = bitstring.BitArray(length=48)
				bitarr.bytes = data_string
				bitarr = bitarr[::-1]
				for i, name in enumerate(OMRNAMES):
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
				bitarr[OMR_VALS[i]:OMR_VALS[i+1]] = 0

		#bitarr = bitarr[::-1]
		mask0 = (bitarr[0:8] + bitarr[8:16])
		mask1 = (bitarr[16:24] + bitarr[24:32])
		mask2 = (bitarr[32:40] + bitarr[40:48])
		return mask0.uint, mask1.uint, mask2.uint

class marsCameraInfoHeader(marsObservable):
	"""
	An object for detailing the info header that comes back from the Medipix chip
	when InfoHeader=1 is enabled in the OMR during a matrix read.

	The info header typically contains the current OMR, the fuses from the chip, the
	DAC string, and a whole bunch of unused data. It is 256-bits long.

	TODO: Parse the DAC string part of the infoheader (am I remembering this wrong?)
	"""
	def __init__(self, data_string):
		marsObservable.__init__(self)
		self.marslog = ml.logger(source_module="pyMarsCamera", group_module="camera", class_name="marsCameraInfoHeader", object_id=self)
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
		marsObservable.__init__(self)
		self.marslog = ml.logger(source_module="pyMarsCamera", group_module="camera", class_name="marsCameraImage", object_id=self)
		if issubclass(type(data_string), str):
			if len(data_string) == 98304 + 56: # Multi frame acquire response
				#header
				assert(ord(data_string[0]) == 0xbb)
				assert(ord(data_string[1]) == 0xaa)
				assert(ord(data_string[2]) == 0xbb)
				assert(ord(data_string[3]) == 0xaa)
				#multiframe information
				self.timestamp = struct.unpack('>I', numpy.frombuffer(data_string[4:8][::-1], dtype='uint8'))[0]
				self.shutterduration =  struct.unpack('>I', numpy.frombuffer(data_string[8:12][::-1], dtype='uint8'))[0]
				self.frameindex =  struct.unpack('>I', numpy.frombuffer(data_string[12:16][::-1], dtype='uint8'))[0]
				self.framecount =  struct.unpack('>I', numpy.frombuffer(data_string[16:20][::-1], dtype='uint8'))[0]
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
		#  turn out to actually be needed in the readMatrix stuff.
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


class marsCameraMessage(marsObservable):
	"""
	A class for holding camera message objects.
	"""
	def __init__(self, data_string_or_command, msg_id=None, params=None):
		marsObservable.__init__(self)
		self.marslog = ml.logger(source_module="pyMarsCamera", group_module="camera", class_name="marsCameraMessage", object_id=self)

		if params is None:
			message = data_string_or_command
			message_components = message.split(',')
			self.msg_id = int(message_components[1])
			self.message_name = message_components[0][1:] # remove the !
			self.checksum = message_components[-1].strip()[:-1] # remove the #
			self.params = [base64.b64decode(p) for p in message_components[2:-1]]
		else:
			self.msg_id = msg_id
			self.params = params
			self.message_name = data_string_or_command
			self.checksum = self.get_checksum()
		self.msg_time = time.time()

	def check_cs(self):
		"""
		Check that the checksum in a message matches what is calculated from its data.

				Time required for this method to run, measured with two frames.
		0.01590 for a message of length 131093 bytes
		0.01373 for a message of length 131093 bytes, which is 16ms per frame.
		===>>12 for the get_checksum code, and then 4ms for the splitting etc.
		"""
		their_cs = self.checksum
		our_cs = self.get_checksum()
		ret = our_cs == their_cs
		if ret == False:
			self.marslog.print_log("Checksums do not match on message:", self.message_name, self.msg_id, level="error", method="check_cs")
			raise IOError("checksums don't match: %s, %s"%(our_cs, their_cs))
		return ret

	def get_checksum(self, message=None):
		"""
		Builds a checksum from a message.

		It is worth noting that the multiframe acquire code does not use this, however we may be able to
		get some speedup by doing this in C as DJS says.

		Used to process all frames. Time required from runs by DJS on old and tired PC.
		0.01178 for a message of length 131089 bytes
		0.01177 for a message of length 131089 bytes   - which is 12 milliseconds per frame.
		"""
		if message is None:
			message = self.build_message(no_checksum=True)
		cs = 0
		for ch in message:
			cs ^= ord(ch)
		return "%02X"%(cs)

	def build_message(self, no_checksum=False):
		"""
		Build the readout network message in the correct format.

		!<command_name>,base64(param1),base64(param2),...,checksum#
		"""
		message = "!" + str(self.message_name) + "," + str(self.msg_id) + ","
		for param in self.params:
			if issubclass(type(param), numpy.ndarray):
				message += base64.b64encode(param) + ","
			else:
				message += base64.b64encode(str(param)) + ","

		if no_checksum is False:
			message += self.get_checksum(message) + "#"

		return message

	def slow_checksum(self, message=None):
		"""
		Build a checksum, but not in the fastest way.

		A lovely comment from DJS here on the usefulness of the networking protocol:

		A one line calculation of checksum, that is 18-20ms, instead of 12ms below. If you want faster checksums, use C.
		However, since the TCP code in the kernel does checksum protection, what is the point of this code?
		"""
		if message is None:
			message = self.build_message(no_checksum=True)

		cs = reduce(operator.xor, (ord(s) for s in message), 0)
		return "%02X"%(cs)

@hasSocketLock
@marsthread.hasRWLock
@marsthread.hasLock
class marsReadoutProtocol(marsObservable):
	def __init__(self, print_receive=False):
		marsObservable.__init__(self)
		self.marslog = ml.logger(source_module="pyMarsCamera", group_module="camera", class_name="marsReadoutProtocol", object_id=self)

		self.print_receive = print_receive
		# socket buffers
		self.rx_raw_buffer = ""
		self.rx_buffer = []
		self.old_rx_buffer = []
		# socket info
		self.connected = False
		self.socket = socket.socket()
		# other vars
		self._msg_id = 0
		self.command_complete_id = -1
		self.fail_count = 0
		self.hv_cli = False
		self.readout_version = 5.3

	@marsthread.takesLock
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

		try:
			self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
		except Exception as msg:
			self.marslog.print_log("Exception on setting NODELAY ", msg, level="error", method="connect")
		try:
			self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256000)
		except Exception as msg:
			self.marslog.print_log("Exception on setting socket send buffer size to 256 K ", msg, level="error", method="connect")
		try:
			self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256000)
		except Exception as msg:
			self.marslog.print_log("Exception on setting socket receive buffer size to 256 K ", msg, level="error", method="connect")

		return self.connected

	@marsthread.takesLock
	def disconnect(self, flush_queue=False):
		"""
		Disconnect the socket, and set the flags to close the networking threads.
		"""
		self.connected = False

		self.socket.close()

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

	@takesSocketLock
	def _recv(self):
		self.socket.settimeout(5e-4)
		ch = self.socket.recv(1000)
		return ch

	@marsthread.takesLock
	@marsthread.modifier
	def recv_command(self, command, msg_id=None, timeout=5.0):
		"""
		No longer its own thread, we run for timeout time, looking for a specific command.
		Any commands that aren't specific commands get added to the queue. As soon as we find
		specific command in the queue we stop receiving (maybe before we even receive anything
		on the socket).

		No messages should be coming back on the socket unless we've asked for them, so should be safe
		to only check it at the right times.

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
		"""
		start_time = time.time()
		ret = self.get_from_rx_buffer_queue(command, msg_id)
		while self.connected and ret is None and time.time() - start_time < timeout:
			try:
				ch = self._recv()
			except socket.timeout:
				time.sleep(5e-4)
			else:
				if self.print_receive:
					print ch
				self.rx_raw_buffer += ch
				if "!" in ch:
					packet_start_time = time.time()
				if "\n" in ch:
					try:
						messages = self.rx_raw_buffer.split('\n')
						try:
							for mesg in messages[:-1]:
								if "!" in mesg and "#" in mesg:
									start_i = mesg.index("!")
									end_i = mesg.index("#")
									_mesg = mesg[start_i:end_i+1]
									try:
										self.add_msg_to_buffer(_mesg) ### This is 32 ms..
									except Exception as msg:
										self.marslog.print_log("Readout communication error. Returned message:", msg, mesg[mesg.index("!"):], level="error", method="recv_command")
								else:
									self.marslog.print_log("Invalid data back from readout:", mesg, level="warning", method="recv_command")
						finally:
							self.rx_raw_buffer = messages[-1]
							ret = self.get_from_rx_buffer_queue(command, msg_id)
					except Exception as msg:
						self.marslog.print_log("Error during receive:", msg, level="error", method="recv_command")
						self.rx_raw_buffer = ""
		return ret

	@marsthread.modifier
	def get_from_rx_buffer_queue(self, command_name, msg_id):
		"""
		Search the list of received messages for the one we want.

		Clear any messages that are greater than 20.0 seconds old without having been asked for,
		and provide a warning.
		"""
		ret = None
		timeout_messages = []
		old_timeout_messages = []

		for message in self.old_rx_buffer:
			if time.time() - message.msg_time > 40.0:
				old_timeout_messages.append(message)

		for message in old_timeout_messages:
			if message is not ret:
				self.old_rx_buffer.remove(message)

		for message in self.rx_buffer:
			if time.time() - message.msg_time > 20.0:
				found = False
				for _m in self.old_rx_buffer:
					if _m.message_name == message.message_name and _m.msg_id == message.msg_id:
						found = True
						break
				if found is False:
					self.marslog.print_log("message", message.message_name + "-" + str(message.msg_id), "has not been cleared after", time.time() - message.msg_time, "seconds", self.address, level="warning", method="get_from_rx_buffer_queue")
					timeout_messages.append(message)
				else:
					#print "received repeat message that wasn't cleared", message.message_name, message.msg_id
					timeout_messages.append(message)
			elif ret is None and command_name == message.message_name:
				if command_name is "commandComplete":
					if (msg_id is None or msg_id == int(message.params[0])) and message.msg_id > self.command_complete_id:
						self.command_complete_id = message.msg_id
						ret = message
				else:
					if msg_id is None or msg_id == message.msg_id:
						ret = message

		if ret is not None:
			self.rx_buffer.remove(ret)
			self.old_rx_buffer.append(ret)


		for message in timeout_messages:
			if message is not ret:
				self.rx_buffer.remove(message)


		return ret

	@marsthread.modifier
	def add_msg_to_buffer(self, data_message):
		"""
		Called by the receive thread. Put received frame into a buffer.

		We have to send back ack messages to valid non-ack messages received.
		"""

		message = marsCameraMessage(data_message)
		#if message.message_name == "commandComplete":
		#	print time.time(), "received", self.address, data_message[0:50]
		if "commandFailed" in message.message_name:
			self.marslog.print_log(self.address, "-" + message.message_name + " " + ",".join([str(m) for m in message.params]), level="warning", method="add_msg_to_buffer")
		if message.message_name != 'ack':
			try:
				self.possibly_send_ack(message)
			except Exception as msg:
				pass
		if message.message_name != "progress":
			self.rx_buffer.append(message)
		else:
			print "progress is", message.params

	def possibly_send_ack(self, message):
		"""
		Send an ack to a received message, but only if its a valid message (e.g. checksum is correct)
		"""
		if message.check_cs():
			msg_id = message.msg_id
			new_message = marsCameraMessage("ack", msg_id=msg_id, params=[])
			self.tx_command(new_message.build_message())
		else:
			raise IOError("Checksum on received message from v5 camera failed")

	@takesSocketLock
	def tx_command(self, message):
		"""
		Actually transmit the message to the readout.

		Message must be in the correct format for the v5 camera server.
		"""
		self.socket.settimeout(1.0)
		# print time.time(), "sending command on ", self.address,  message[0:50],
		if self.connected:
			self.socket.sendall(message + "\r\n")

	@marsthread.takesLock
	def async_send_command(self, message):
		start_time = time.time()

		self.tx_command(message.build_message())

		return start_time, message

	@marsthread.takesLock
	def finalise_send_command(self, message, start_time, tries=0, ret_msg=None, ret_msg_timeout=5.0, allow_retries=True):
		TIMEOUT = 1.0*2
		if allow_retries:
			MAX_TRIES = 5
		else:
			MAX_TRIES = 1
		ack = None
		ret = None

		while ack is None and time.time() - start_time < TIMEOUT:
			ack = self.recv_command(command="ack", msg_id=message.msg_id, timeout=5e-3)

		if time.time() - start_time >= TIMEOUT or ack.check_cs() is False:
			if ack is None:
				failure_reason = "timeout"
			else:
				failure_reason = "bad checksum"
			if (tries < MAX_TRIES) and self.connected:
				if not (message.message_name == "hvOn" and tries == 0):
					self.marslog.print_log("Retrying after failure (%s):"%(failure_reason), self.address, message.build_message()[:100], level="warning", method="finalise_send_command")
				ret = self.send_command(message, tries = tries + 1)

			else:
				self.marslog.print_log("Failed too many times. Aborting.", self.address, message.build_message()[:100], level="warning", method="finalise_send_command")
				self.fail_count += 1
				if self.fail_count >= 3:
					self.marslog.print_log("Aborted too many commands due to failure. Disconnecting connection.", level="error", method="finalise_send_command")
					self.disconnect()
				return None

		command_resp_time = time.time()

		#search for commandComplete or commandFailed
		tmp = None
		while self.hv_cli is False and self.readout_version >= 5.2 and tmp is None and time.time() - command_resp_time < ret_msg_timeout:
			if not self.connected:
				return None

			tmp = self.recv_command(command="commandFailed", msg_id=None, timeout=5e-3)
			if tmp is None:
				if message.message_name == "test":
					tmp = self.recv_command(command="commandComplete", msg_id=None, timeout=5e-3)
				else:
					tmp = self.recv_command(command="commandComplete", msg_id=message.msg_id, timeout=5e-3)
			else:
				self.marslog.print_log("Received commandFailed in response to", self.address, message.build_message()[:30], level="warning", method="finalise_send_command")

		if self.hv_cli is False and tmp is None and self.readout_version >= 5.2:
			self.marslog.print_log("Failed to get commandComplete or Failed within timeout for command", message.build_message()[:30], level="error", method="finalise_send_command")
		#if self.hv_cli is False and tmp is not None:
		#	self.marslog.print_log("Received commandComplete/Failed with msg_id", self.address, tmp.msg_id, method="finalise_send_command")


		ack_start_time = time.time()


		if ret_msg is not None and ret is None:
			if not self.connected:
				return None

			ret = None

			while ret is None and time.time() - ack_start_time < ret_msg_timeout:
				ret = self.recv_command(command=ret_msg, msg_id=None, timeout=5e-3)
			if ret is None:
				self.marslog.print_log("Failed to get response", ret_msg, "in time from v5 camera. Aborting.", level="warning", method="send_command")

		if (self.hv_cli == True or self.readout_version < 5.2) and ret_msg is None and ret is None:
			ret = ack

		return ret


	@marsthread.takesLock
	def send_command(self, message, tries=0, ret_msg=None, ret_msg_timeout=5.0, allow_retries=True):
		"""
		Send a network message to the readout.

		Check for ACKs to confirm it was sent, and retry on a fail.

		Retries by recursively calling itself, labelling how many tries it is up to.
		"""
		start_time, message = self.async_send_command(message)
		return self.finalise_send_command(message=message, start_time=start_time, tries=tries, ret_msg=ret_msg, ret_msg_timeout=ret_msg_timeout, allow_retries=allow_retries)


	def destroy(self, *args):
		"""
		A destroy/disconnect method.. only used for the ifdef==main stuff below.
		"""
		self.marslog.print_log("Initiate close now.", level="debug", method="destroy")
		self.disconnect()
		time.sleep(1.0)

	def run_command(self, name, params=[], ret_msg=None, ret_msg_timeout=5.0, allow_retries=True):
		"""
		Build and push a command to the readout to be processed.

		Arguments:
			name - the name of the command you wish to run on the v5camera
			params - a list of each parameter you want to send with the command

		This is the command you use for generic testing via a console.
		"""
		message = marsCameraMessage(name, self.msg_id, params)
		return self.send_command(message, ret_msg=ret_msg, ret_msg_timeout=ret_msg_timeout, allow_retries=allow_retries)

	def async_run_command(self, name, params=[]):
		message = marsCameraMessage(name, self.msg_id, params)
		return self.async_send_command(message)

marsthread.create_RWlock_variable(marsReadoutProtocol, "_rx_buffer", "rx_buffer")
marsthread.create_RWlock_variable(marsReadoutProtocol, "_rx_raw_buffer", "rx_raw_buffer")
marsthread.create_RWlock_variable(marsReadoutProtocol, "_connected", "connected")
marsthread.create_RWlock_variable(marsReadoutProtocol, "_fail_count", "fail_count")

class marsHVClient(marsReadoutProtocol):

	def __init__(self):
		marsReadoutProtocol.__init__(self)
		self.hv_cli = True
		self.marslog = ml.logger(source_module="pyMarsCamera", group_module="camera", class_name="marsHVClient", object_id=self)
		self.hv = 0
		self.read_hv_v = 0
		self.read_hv_a = 0

	@marsthread.takesLock
	def set_hv(self, hv, allow_retries=True):
		"""
		Set the high voltage.

		If 0, turn it off, if > 0, turn it on.

		TODO: Use the correct polarity for the chip,
		instead of assuming its a CdTe/CZT/GaAs.
		"""
		if self.connected:
			self.hv = hv
			ret1 = self.run_command("hvSetPolarity", ["-"], allow_retries=allow_retries)
			if hv > 0:
				self.marslog.print_log("turning HV on with val", hv, level="info", method="set_hv")
				time.sleep(500e-3)
				ret2 = self.run_command("hvSetVolts", [str(hv)], allow_retries=allow_retries)
				time.sleep(1500e-3)
				ret3 = self.run_command("hvOn", [], allow_retries=allow_retries)
			else:
				self.marslog.print_log("Turning hv off", level="info", method="set_hv")
				time.sleep(1500e-3)
				ret2 = True
				ret3 = self.run_command("hvOff", [], allow_retries=allow_retries)
			if ret1 is None or ret2 is None or ret3 is None:
				raise IOError("Exception setting bias voltage")

	def get_hv(self):
		"""
		Get the HV currently being set to the chip.
		"""
		return self.hv

	@marsthread.takesLock
	def read_hv(self):
		"""
		Read the HV from the ADC monitor.
		"""
		if self.connected:
			ret = self.run_command("hvReadVolts", ret_msg="hvVolts")
			self.read_hv_v = ret.params[0]
			return self.read_hv_v
		else:
			return 0.0

	@marsthread.takesLock
	def read_hv_current(self):
		"""
		Read the leakage current of the HV circuit.
		"""
		if self.connected:
			ret = self.run_command("hvReadCurrent", ret_msg="hvCurrent")
			self.read_hv_a = ret.params[0]
			return self.read_hv_a
		else:
			return 0.0


marsthread.create_RWlock_variable(marsHVClient, "_read_hv_a", "read_hv_a")
marsthread.create_RWlock_variable(marsHVClient, "_read_hv_v", "read_hv_v")
marsthread.create_RWlock_variable(marsHVClient, "_hv", "hv")

class marsCameraClient(marsReadoutProtocol):
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

	Ideally the objects should be created via the UDPListener class, which
		will automatically identify the servers that can be connected to
		and create the correct marsCameraClient for it.
	"""
	MULTIFRAME_IDLE, MULTIFRAME_RUNNING, MULTIFRAME_TRIGGER_ABORT, MULTIFRAME_ABORTING, MULTIFRAME_FAILED_START = range(5)

	def __init__(self):

		marsReadoutProtocol.__init__(self)
		self.marslog = ml.logger(source_module="pyMarsCamera", group_module="camera", class_name="marsCameraClient", object_id=self)
		# multiframe related
		self.mf_port = get_mf_port()
		self.mf_sock = None
		self.multi_frames = []
		self.mf_state = 0
		self.mf_sync_mode = "standalone"
		self.multiframe_timeout = 1.0

		# persistent settings
		self.OMR = {}
		self.dacvals = {}

		# retrieved variables
		self.ADC_val = [0 for i in range(2, 14)]
		self.DAC_scan_vals = []
		self.fuses = 0
		self.temperature = 0
		# actual image related

		self.frame = [[],[]]
		self.imageframe = [[],[]]
		self.configurationmatrix = [[],[]]
		self.downloaded_image = [[],[]]

		# other (non-lock checked)
		self.extdac = 2048
		self.extbg = 2048
		self.use_global_config = True   # we are being run from an external entity, so we should get info from the MarsConfig environment


	# We've never written a read_dacs command, but still maybe useful for future?
	# triggers off "DACs"
	# def read_dacs(self, message_components):
	# 	"""
	# 	A callback for setting the current dacvals to what is on the sensor from a read_dacs command.
	# 	"""
	# 	val = base64.b64decode(message_components[2])
	# 	bitarr = bitstring.BitArray(bytes=val)[::-1]
	# 	for i,dacname in enumerate(DACNAMES):
	# 		bitval = bitarr[DACBITVALS[i]:DACBITVALS[i+1]][::-1]
	# 		self.dacvals[dacname] = int(bitval.uint)


	@marsthread.takesLock
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
				bitarr[DACBITVALS[i]:DACBITVALS[i+1]] = 0


		bbytes = bitarr[::-1].bytes

		self.run_command("writeDACs", [str(extdac), str(extbg), bbytes])

	@marsthread.takesLock
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

		self.run_command("setOMRconfig", ["%04x"%(tmp0), "%04x"%(tmp1), "%04x"%(tmp2)])
		# The "setOMRconfig" doesn't actually write the OMR, just leaves it so the next command sets it off.
		# Write the current DACs to allow the OMR to be processed.
		self.write_DACs({})

	@marsthread.takesLock
	def acquire(self, exptime_ms):
		"""
		Expose the shutter manually, and wait for the acquire to finish.
		"""
		self.run_command("openShutter", [str(int(exptime_ms))], ret_msg="shutterDone", ret_msg_timeout=(2.0 * exptime_ms + 150) * 1e-3)

	@marsthread.reader
	def get_multiframes(self):
		return self.multi_frames

	@marsthread.takesLock
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
			self.mf_sock.bind(('0.0.0.0', self.mf_port))
			self.mf_sock.settimeout(1.0)
		self.multiframe_timeout = (exptime_ms + 225.0) / 1000.0 * frame_count * 2.0 + 2.0
		if frame_count > 1:
			self.marslog.print_log("multiframe_acquire called with exptime, frame_count, spacing_time, counters, sync_mode, parallel:", exptime_ms, frame_count, spacing_time_ms, counters, sync_mode, parallelisation, level="debug", method="multiframe_acquire")

		self.mf_start_time = time.time()
		self.mf_sock.listen(1)

		# todo set the ip address here from lookup rather than assuming its 192.168.0.1
		self.mf_sync_mode = sync_mode
		mr_start_time, mr_message = self.async_run_command("multipleReadout", [frame_count, exptime_ms, '192.168.0.1', self.mf_port,
						     sync_mode, spacing_time_ms, counters, parallelisation])

		try:
			tmp_sock, tmp_addr = self.mf_sock.accept()
		except Exception as msg:
			self.marslog.print_log(self.address, mr_message.msg_id, "Exception accepting mf socket", msg, sync_mode, level="error", method="multiframe_acquire")
			tmp_sock = None
		if tmp_sock is not None:
			#print sync_mode, " primed", self.address
			self.mf_state = MULTIFRAME_RUNNING

			try:
				tmp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024000)
			except Exception as msg:
				self.marslog.print_log("Exception on setting socket send buffer size to 1024 K ", msg, level="error", method="multiframe_acquire")
			try:
				bufsize = min(frame_count * 2 * 1024000, 500*1024000)
				tmp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, bufsize)
			except Exception as msg:
				self.marslog.print_log("Exception on setting socket receive buffer size to large amount ", msg, level="error", method="multiframe_acquire")


		self.finalise_send_command(mr_message, mr_start_time)

		if tmp_sock is not None:
			self.read_multiframe(tmp_sock, frame_count, counters, while_waiting_fn=while_waiting_fn)

			if self.readout_version >= 5.3:
				self.recv_command(command="multipleReadoutComplete", timeout=2.0)
			return self.multi_frames
		else:
			self.mf_state = MULTIFRAME_FAILED_START

	def cancel_multiframe(self):
		"""
		Cancel the multiple readout command currently running on
			the v5 camera.
		"""
		if self.mf_state in [MULTIFRAME_RUNNING]:
			if self.mf_sync_mode in ["master", "standalone"]:
				self.marslog.print_log("Multiframe abort manually triggered", level="info", method="cancel_multiframe")
			self.mf_state = MULTIFRAME_TRIGGER_ABORT # this triggers a cancel from the multiframe acquire command

	@marsthread.takesLock
	def read_multiframe(self, sock, frame_count, counters, while_waiting_fn=None):
		"""
		Read the multiple acquire data on a socket, and
			automatically parse it into a list of
			marsCameraImage objects.

		Parses the data in to marsCameraImages as it is
			downloading currently.

		TODO: A notify every X frames to allow the GUI to
			update the image as the multiple acquire is
			running.

		Note: Aborting multiframes can be triggered either by the
		while_waiting_fn causing an error, or somebody
		calling the cancel_multiframe command.
		"""
		# header, timestamp, shutter, frameindex, framecount, counter, unused, sensorId, pixelmatrix

		self.multi_frames = []
		try:
			FRAMELENGTH = 4 + 4 + 4 + 4 + 4 + 1 + 3 + 32 + 98304
			frame_data = ""
			ctr = 0
			read_amount = 0
			current_timeout = 5.0
			sock.settimeout(current_timeout)
			if counters == "both":
				total_len = FRAMELENGTH * frame_count * 2
				total_frames = frame_count * 2
			else:
				total_len = FRAMELENGTH * frame_count
				total_frames = frame_count
			ctr_timeout = time.time()
			while len(frame_data) < total_len:
				if self.mf_state in [MULTIFRAME_RUNNING]:
					try:
						if while_waiting_fn is not None:
							while_waiting_fn()
					except Exception as msg:
						#trigger the abort
						if self.mf_sync_mode in ["master", "standalone"]:
							self.marslog.print_log("Exception in while_waiting_fn is triggering an abort in multiframe acquire", level="debug", method="read_multiframe")
						self.mf_state = MULTIFRAME_TRIGGER_ABORT
				if self.mf_state == MULTIFRAME_TRIGGER_ABORT:
					abort_start_time, abort_message = self.async_run_command("abortMultipleReadout", [])
					self.mf_state = MULTIFRAME_ABORTING
					current_timeout = 0.5
					sock.settimeout(current_timeout)

				# read the data (we need to keep doing this even if its during an abort)
				try:
					ch = sock.recv(102400)
				except socket.timeout:
					if self.mf_state != MULTIFRAME_ABORTING:
						self.marslog.print_log("Socket timeout on multiframe read", ctr, level="error", method="read_multiframe")
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
						self.marslog.print_log("Exception in frame header at frame_index=", ctr-1, level="warning", method="read_multiframe")
					self.multi_frames.append(frame)
					if ctr%20 == 0: # show progress
						sys.stdout.write('.')
						sys.stdout.flush()
				if time.time() - ctr_timeout > current_timeout:
					if self.mf_state != MULTIFRAME_ABORTING:
						self.marslog.print_log("Timeout since last image read (images read = %d)"%(ctr), level="error", method="read_multiframe")
					break

			# if self.mf_state == MULTIFRAME_RUNNING:
			# 	while True:
			# 		try:
			# 			data = sock.recv(1024)
			# 			if len(data) == 0:
			# 				break
			# 			if len(data) >= 1:
			# 				self.marslog.print_log("Got non-shutdown response from socket", len(data), level="error", method="read_multiframe")
			# 				break
			# 		except socket.timeout:
			# 			break

			if frame_count > 1 and self.mf_state in [MULTIFRAME_RUNNING]:
				self.marslog.print_log("finished receiving frames at:", time.time(), time.time() - self.mf_start_time, ctr, level="debug", method="read_multiframe")
			if self.mf_state == MULTIFRAME_ABORTING:
				self.marslog.print_log("finished aborting frames at:", time.time(), time.time() - self.mf_start_time, ctr, level="debug", method="read_multiframe")
				self.finalise_send_command(abort_message, abort_start_time)

			return self.multi_frames

		finally:
			#print self.address, "finished mf reading with total read = ", len(frame_data), "\n"
			if self.mf_state in [MULTIFRAME_RUNNING, MULTIFRAME_ABORTING]:
				self.mf_state = MULTIFRAME_IDLE

	@marsthread.takesLock
	def download_image(self, counter):
		"""
		Manually download a counter from the Medipix into local memory.
		"""
		MATRIX_READ_TIMEOUT = 10.0
		if counter == 0:
			p_counter = "L"
		else:
			p_counter = "H"

		ret = self.run_command("readMatrix", [p_counter], ret_msg="pixelMatrix", ret_msg_timeout=MATRIX_READ_TIMEOUT)
		self.set_frame(ret, counter)


	@marsthread.takesLock
	def get_id(self):
		"""
		Get the fuses from the Medipix chip.
		"""
		ID_READ_TIMEOUT = 5.0
		ret = self.run_command("getID", [], ret_msg="id", ret_msg_timeout="ID_READ_TIMEOUT")

		IH_bytes = ret.params[0]
		infoheader = marsCameraInfoHeader(IH_bytes)
		self.fuses = infoheader.fuse

		return self.fuses

	@marsthread.takesLock
	def get_temperature(self):
		"""
		Read the temperature off the MitySOM FPGA sensor.
		"""
		TEMPERATURE_READ_TIMEOUT = 1.0
		ret = self.run_command("getTemperature", [], ret_msg="temperature", ret_msg_timeout=TEMPERATURE_READ_TIMEOUT)
		self.temperature = float(ret.params[0])

		return float(self.temperature)

	@marsthread.takesLock
	def get_sense_dac_temperatures(self):
		"""
		Read the temperature ADCs from the detector.
		"""
		p = self.read_ADC(dac_name='band_gap_output')
		q = self.read_ADC(dac_name='band_gap_temperature')
		return p, q

	@marsthread.takesLock
	def read_ADC(self, dac_name=None):
		"""
		Read all the ADCs. Store them all into ADC_val, but only return
			the ADC for the SenseDAC on the Medipix chip.

		Has the support to set the SenseDAC for the requested DAC,
			however	if dac_name is None then it will use the
			currently configured SenseDAC.
		"""
		ADC_READ_TIMEOUT = 5.0

		if dac_name is not None:
			for key in RX3_SCAN.keys():
				if dac_name.lower() == key.lower():
					self.write_OMR({"SenseDAC":RX3_SCAN[key]})
					break

		time.sleep(50e-3)

		ret = self.run_command("getADCs", [], ret_msg="ADCs", ret_msg_timeout=ADC_READ_TIMEOUT)

		time.sleep(50e-3)

		self.ADC_val = [int(p) for p in ret.params]

		return self.ADC_val[11]

	@marsthread.takesLock
	def run_dac_scan(self, dac_name):
		"""
		The V5 camera readout has a DAC scan command, for iterating a single DAC
			over its full range of values and receiving the ADC results for
			each value. Due to the lack of network latency this is considerably
			faster than doing it by hand.

		This method runs that DAC scan on the camera.
		"""
		DAC_SCAN_TIMEOUT = 15.0

		for key in RX3_SCAN.keys():
			if dac_name.lower() == key.lower():
				break

		dac_index = RX3_SCAN[key]
		ret = self.run_command("test", ["dacScan", "linear", dac_index], ret_msg="testResults", ret_msg_timeout=5.0)

		datalines = (",".join(ret.params).strip('\r\n').strip(',').split('\r\n'))
		tmp = [[int(data) for data in dataline.split(',')] for dataline in datalines[1:]]
		self.DAC_scan_vals = [[tmp[j][i] for j in range(len(tmp))] for i in range(len(tmp[0]))] #transpose

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

	@marsthread.takesLock
	def upload_image(self, imageframe, counter=1):
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

	@marsthread.modifier
	def set_frame(self, message, counter):
		"""
		This is only valid for the original "acquire" command, not the multiple Acquire command.
		"""
		framedata = message.params[0]
		imageframe = marsCameraImage(framedata, deshift=False)
		self.downloaded_image[counter] = imageframe.get_image()

		self.frame[counter] = imageframe.get_image(deshift=True)

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
			self.marslog.print_log("Threshold 0 set to", val, "with frame mean", self.frame[0].mean(), "and pixels counting:", (self.frame[0] > 10).sum(), level="info", method="noisefloor_scan")

		self.last_frames = numpy.array(frames)


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
			self.marslog.print_log("Error: ", total, " pixels did not match from mask read write test.", level="warning", method="test_mask_read_write")
			print "quadrants are:"
			for i in range(8):
				for j in range(8):
					quadrant = (image[i*32:(i+1)*32,j*32:(j+1)*32] != mask[i*32:(i+1)*32,j*32:(j+1)*32]).sum()
					if quadrant != 0:
						sys.stdout.write('x')
					else:
						sys.stdout.write('.')

				print ""
		return total < 20000


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
			self.write_DACs({dacname:top_val})
			v_top_val = self.read_ADC(dacname)
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
		if self.readout_version < 5.2:
			self.run_command("writeMedipixSupplies", ["1510", "1510", "2510"])   #Set voltage for VDD, VDDA, VDDD, which should be measured at 1.5, 1.5 and 2.5
		self.write_OMR({"GainMode":0, "Disc_CSM_SPM":0, "Polarity":0, "Equalization":0, "ColourMode":0, "CSM_SPM":0, "ExtDAC":0, "ExtBGSel":0, "EnableTP":0, "SenseDAC":0})
		self.run_command("hardwareReadout", ["T"])
		if self.use_global_config:
			# Can we skip this if we always use marscamera from here on out?
			from marsct import marssystem
			cfg = marssystem.MarsSystem().get_camera_config()
			#print "======================1========================"
			def_mode = cfg.software_config.default_mode
			#print "======================2========================"
			dacs = cfg.software_config.modes[def_mode]["dacs"][0]
			self.write_DACs(dacs, extdac=0, extbg=0)
		else:
			self.write_DACs({'V_Rpz': 255, 'V_Tp_refA': 50, 'V_Tp_refB': 255, 'I_Shaper_test': 100, 'I_DAC_test': 100, 'V_Cas': 174, 'V_Tp_ref': 120, 'I_Ikrum': 30, 'Threshold1': 50, 'Threshold0': 30, 'Threshold3': 50, 'Threshold2': 50, 'Threshold5': 50, 'Threshold4': 300, 'Threshold7': 300, 'Threshold6': 300, 'I_Disc_LS': 100, 'I_DAC_DiscL': 64, 'I_DAC_DiscH': 69, 'I_Disc': 125, 'I_Shaper': 200, 'I_TP_BufferOut': 4, 'V_Fbk': 177, 'V_Gnd': 135, 'I_TP_BufferIn': 128, 'I_Delay': 30, 'I_Preamp': 150}, extdac=0, extbg=0)
                #c by vt
		#self.upload_image(numpy.zeros([256, 256]), counter=0)
		#self.upload_image(numpy.zeros([256, 256]), counter=1)


	def test_matrix_read_write(self):
		"""
		A full suite of read/write tests on the pixel matrix of the v5 camera.

		Tests that a) the read and write are workign correctly and b) that row/columns are being done correctly.

		pixel_patterns is what gets applied to the tested pixels.
		Spacings determines the gaps between pixels that are active for the test.
		"""
		pixel_patterns = [0x0, 0xfff, 0xa5a, 0x5a5]
		spacings = [1, 2, (7,11)]
		counters = [0, 1]
		success = 0
		total = 0
		exc_count = 0
		img_count = 0
		for spacing in spacings:
			for pix_pat in pixel_patterns:
				self.marslog.print_log("Testing pix_pat: 0x%03X"%(pix_pat), "spacing: ", spacing, level="info", method="test_matrix_read_write")
				mask = self.generate_mask(pix_pat, spacing)
				for counter in counters:
					try:
						if self.test_mask_read_write(mask, counter):
							success += 1
						img_count = img_count + 1
						if img_count - success > 5:
							self.marslog.print_log("Pixel matrix read write test: Aborting due to too many failures", level="error", method="test_matrix_read_write")
							return False
					except Exception as msg:
						exc_count = exc_count + 1
						if exc_count > 5:
							raise
						self.marslog.print_log("Exception in testing the individual mask:", msg, level="error", method="test_matrix_read_write")
					total += 1

		self.marslog.print_log("%0.1f%% of read write tests passed (%d / %d)"%((100.0 * float(success) / float(total)), success, total), level="info", method="test_matrix_read_write")
		return success == total

	def test_adc_tuning(self):
		"""
		Test the key dacs that require tuning actually meet their required values.
		"""
		final_voltages = {"V_Cas":850, "V_Gnd":650, "V_Fbk":850}
		self.marslog.print_log("V5 camera tuning voltages to:", final_voltages, level="info", method="test_adc_tuning")
		self.tune_DACs_from_ADCs(final_voltages)
		total_error = 0.0
		for dacname in final_voltages.keys():
			dacval = self.dacvals[dacname]
			v_dacval = self.read_ADC(dacname)
			error = abs(float(v_dacval) - float(final_voltages[dacname])) / float(final_voltages[dacname]) * 100
			self.marslog.print_log(dacname, " measured at ", v_dacval, "mV (error = %0.1f%%)"%(error), level="info", method="test_adc_tuning")
			total_error += error

		self.marslog.print_log("Average error was: %0.1f%%"%(total_error / len(final_voltages)), level="info", method="test_adc_tuning")

		return (error / len(final_voltages)) < 1.0
              
        def upload_mask(self, name, code):
                try:
                    mask = numpy.load(name)
                except:
                    print "numpy load problem"
                    pass
                if "mask" not in locals():
                    print "failed to read mask file", name
                else:
                    print "read in mask file", name
                    #print self.test_mask_read_write(mask & code)
                    #if self.test_mask_read_write(mask & code):
                        #print "the mask is successfully read"

# automatic locks for making variables thread-safe.
# multiframe related
marsthread.create_RWlock_variable(marsCameraClient, "_multiframe_timeout", "multiframe_timeout")
marsthread.create_RWlock_variable(marsCameraClient, "_mf_state", "mf_state")
marsthread.create_RWlock_variable(marsCameraClient, "_mf_sock", "mf_sock")
marsthread.create_RWlock_variable(marsCameraClient, "_mf_port", "mf_port")
marsthread.create_RWlock_variable(marsCameraClient, "_multi_frames", "multi_frames")
# various variables to be stored after retrieval
marsthread.create_RWlock_variable(marsCameraClient, "_temperature", "temperature")
marsthread.create_RWlock_variable(marsCameraClient, "_fuses", "fuses")
marsthread.create_RWlock_variable(marsCameraClient, "_DAC_scan_vals", "DAC_scan_vals")
marsthread.create_RWlock_variable(marsCameraClient, "_ADC_val", "ADC_val")
# actual image related
marsthread.create_RWlock_variable(marsCameraClient, "_downloaded_image", "downloaded_image")
marsthread.create_RWlock_variable(marsCameraClient, "_configurationmatrix", "configurationmatrix")
marsthread.create_RWlock_variable(marsCameraClient, "_imageframe", "imageframe")
marsthread.create_RWlock_variable(marsCameraClient, "_frame", "frame")
# settings we send but need to be persistent
marsthread.create_RWlock_variable(marsCameraClient, "_dacvals", "dacvals")
marsthread.create_RWlock_variable(marsCameraClient, "_OMR", "OMR")



if __name__ == "__main__":
	pass
