# @package marsct.marscamera
# MARS-CT3 Medipix camera controller module.
# The module includes an abstract class for common camera methods and
# MXR and MP3 classes for methods specific to those camera types.

__author__ = "jpr, Michael Walsh, Alex Opie"
import pygtk
pygtk.require('2.0')
import gtk
import gobject

import Image
import libmarscamera_py
import numpy
import os
import os.path
import sys
import socket
import grp
import multiprocessing

from marsct import os_accounts
from time import time, sleep
from marsct.marsobservable import marsObservable
from marsct import marslog as ml
marslog = ml.logger(source_module="marscamera", group_module="camera")
#from marsct import marsguiref # THIS DOES NOT WORK !!!
import marsguiref
from marsct import marsdicomref
from marsct import marsprivatetags
REPORT_TIMEOUT = False
try:
	import UDPListener
	from marsct import pyMarsCamera
except Exception as msg:
	marslog.print_log("Failed to import pyMarsCamera from MARS-camera tree " + str(msg), level="error")

import marsct.marsconfig
from DAC_config import RX3_TYPE, RX3_REVISION, RX3_SCAN, RX3_RANGE, RX3_VALUES, MXR_TYPE, MXR_SCAN, MXR_RANGE, MXR_VALUES, MP3_TYPE, MP3_REVISION, MP3p1_REVISION, MP3_SCAN, MP3_RANGE, MP3_VALUES, RANGES
from marsct.network_tools import get_mac_address

#from optparse import OptionParser
verbose = False

import marsthread

IMAGE_ROTATION = {90: Image.ROTATE_90, -90: Image.ROTATE_270, 180: Image.ROTATE_180, 270: Image.ROTATE_270,
                  90.0: Image.ROTATE_90, -90.0: Image.ROTATE_270, 180.0: Image.ROTATE_180, 270.0: Image.ROTATE_270}

IMAGE_ROTATE_FILTER = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}

HV_OFF, HV_WARMUP, HV_ON = range(3)

MAX_HISTORY_COUNT = 1000


class WrongConfigException(Exception):
	pass

class libmarsError(Exception):
	pass


class Message:
# better to declare this here than hide it in an imported module.

   def __init__(self, **kwargs):
        self.__dict__ = kwargs
        self.pid = os.getpid()

#
# TODO:
# - Handle library error codes ... when they become available!
# - Include module level logging
#


def noncounting_pixel_mask(frame, threshold=0):
	if threshold == 0:
		mask = (frame == 0) * 1
	elif threshold > 0:
		mask = (frame <= 0) * 1
	else:
		marslog.print_log("marscamera.noncounting_pixel_mask(): WARNING invalid threshold. Using threshold = 0", level="warning")
		mask = (frame == 0) * 1
	return mask


def bright_pixel_mask(frame, threshold=4095):
	mask = (frame >= threshold) * 1
	return mask


def combine_masks(mask1, mask2):
	mask = ((mask1 == 1) + (mask2 == 1)) * 1
	return mask


def getMarsCamera():
	print "#### getMarsCamera"
        print "#### marsguiref.CONFIG:"
        print " ", marsguiref.CONFIG
        print "#### marsguiref.CONFIG[\"camera\"]:"
        print marsguiref.CONFIG["camera"]
        print sys.argv
        #return 0
	if "camera_testing" in sys.argv:
                print "#### \"camera_testing\" in sys.argv"
		driver = "testing"
	else:
                print "#### \"camera_testing\" not in sys.argv"
		driver = marsguiref.CONFIG["camera"].get_attr(["driver"])
	print "#### driver =", driver
	camera_type = None
	if driver == "libmarscamera":
		try:
			libmarscamera_py.find()
			camera_type = libmarscamera_py.getChipAttribute(0, 0, "type")
			revision_type = libmarscamera_py.getAttribute(0, "revision")
		except Exception as msg:
			marslog.print_log("getMarsCamera:", msg, level="error")
			camera_type = None
			#		raise
	if camera_type is None:
		camera_type = marsguiref.CONFIG["camera"].get_attr(["software_config", "type"])
		if camera_type == RX3_TYPE:
			camera_type = MP3_TYPE
			revision_type = RX3_REVISION
		else:
			revision_type = MP3_REVISION
	if camera_type == MXR_TYPE:
                print "#### camera_type == MXR_TYPE"
		return marsCameraMXR()
	elif camera_type == MP3_TYPE:
		if revision_type == MP3_REVISION or revision_type == MP3p1_REVISION:
                        print "#### camera_type == MP3_TYPE"
			return marsCameraMP3()
		elif revision_type == RX3_REVISION:
                        print "#### camera_type == RX3_TYPE"
			return marsCamera3RX()
	else:
		raise EnvironmentError("Unknown/unsupported camera type")


# Has both locks. Use hasLock for libmars calls and reader/modifier as required for internal data.
@marsthread.hasRWLock
@marsthread.hasLock
class marsCamera(marsObservable):

	"""
	Base class to represent the MARS/Medipix camera family.

	NOTE: This inherits from marsObservable, which should work now.
	A function can be told to be called after many of the functions here complete.

	E.g. A function to redraw the image to the screen can be set as an observer, then when an acquire
	is completed, the image will be redrawn to the screen automatically.

	Unfortunately there is no way to distinguish which events to observe, so images will also
	be redrawn to the screen whenever the DACs are updated etc.
	"""
	STATUS_NORMAL, FUSEID_MISMATCH = range(2)
	@marsthread.takesLock
	def libmars_caller(self, fn, *args):
		check = 0
		if "MARSDBG_LIBMARSCALLER" in os.environ:
			self.marslog.print_log("libmarscaller:", fn, args, level="debug", method="libmars_caller")
			if not hasattr(self, "f"):
				self.f = open(os.path.join(os.getenv("HOME"), "libmarscaller.txt"), 'a')
			self.f.write("libmarscaller:%s, %s\n" % (str(fn), str(args)))

		if self.camera_found == False and fn != libmarscamera_py.find:
			return

		while check < 3:
			try:
				if check > 0 and fn == libmarscamera_py.downloadImage:
					# sleep(50.0/1000.0)  						#sleep 50 ms
					return libmarscamera_py.acquire(self.devid, self.chip, self.exptime / 1000.0)
				else:
					return fn(*args)
			except Exception as msg:
				self.marslog.print_log(msg, level="debug", method="libmars_caller")
				if check < 10:
					check += 1
				else:
					self.marslog.print_log("libmarscaller: ", fn, args, level="error", method="libmars_caller")
					self.marslog.print_log(msg, level="error", method="libmars_caller")
					raise

	def __del__(self):
		if "MARSDBG_LIBMARSCALLER" in os.environ:
			self.f.close()
		self.running = False

	def check_config_loaded(self):
		try:
			if self.config is None:
				self.config = marsguiref.CONFIG["camera"]
		except Exception as msg:
			pass

		assert self.config is not None, "Failed to load marscamera config"
		assert issubclass(type(self.config), marsct.marsconfig.cameraConfig), "mars-camera configs must be a cameraConfig" + str(type(self.config))

	@marsthread.modifier
	@marsthread.takesLock
	def __init__(self, masks=None, dacvals=None):
		"""
		Instantiates a Medipix camera object and sets the pixel configuration mask
		and dac-values if supplied.

		Sets up default internal variables and initial state of the camera.
		"""
		marsObservable.__init__(self)
		self.marslog = ml.logger(source_module="marscamera", group_module="camera", class_name="marsCamera", object_id=self)

		self.hv_mac_addr = ""
		if "camera_testing" in sys.argv:
			self.driver = "testing"
		else:
			self.driver = marsguiref.CONFIG["camera"].get_attr(["driver"], default="testing")

		self.slave_timeout_count = 0
		self.status = self.STATUS_NORMAL
		self.has_dac_lock = False
		self.inhibit_bit = 0
		self.adjustment_bits = [0]
		self.active_high = 0
		self.full_counter = False
		self.scanning = False
		self._camera_found = False
		self.devid = 0
		self._chipid = 0
		self._read_hv = 0.0
		self._last_good_hv = 100000 			# used for handling errors in measurenemts returned by different versions of the bias_voltage boards.
		self._read_temp = 0.0
		self._dacvals = None
		self._frames = []
		self._frame_order = [0]
		self._frame_type = ["SPM"]
		self._noncounting_mask = []
		self._active_high = False
		self._inhibit_bit = 0
		self._adjust_bits = []
		self._bright_mask = []
		self._openframes = []
		self._exptime = 0.1
		self._polarity = 1
		self._hv = 0
		self._base_hv = 200
		self.read_bias_current = 0.0
		self.hv_fixed_output_count = 0
		self.hv_enabled = True
		self.hv_state = HV_OFF
		self.hv_idle_control = False
		self.hv_idle_timeout_start = 0.0
		self.hv_idle_timeout = 600.0
		self.hv_warmup_start = 0.0
		self.hv_warmup_time = 30.0
		self._image = None
		self._lm35_temp = 25.0
		self._config = None
		self._config_loaded = 0
		self._chipcount = 0
		self._type = 0
		self._has_OpenBeam = False
		self._pixelsize = 0.055  # in mm  (changes to 0.11 in MP3 colourmode)
		self.counters = 1
		self.set_counters = 1
		self.csm_counters = 0
		self._mode = None
		self._energy_calibration = None
		self.maxcount = 4096
		self.pseudo_attributes = ["pixelsize", "counters"]
		self.csm = False
		self.colourmode = False
		self.running = True
		self._temperature_history = []
		self._hv_history = []
		self.start_time = time()

		self.modes = {}
		self.mode_map = {}

		self.set_masks(numpy.zeros([256, 256]), chipid="all")

		try:
			if self.driver == "libmarscamera":
				self.libmars_caller(libmarscamera_py.find)
				self.camera_found = True
				self.hv_ip_addr = "192.168.0.44"
				self.hv_mac_addr = get_mac_address([self.hv_ip_addr])[self.hv_ip_addr]
			elif self.driver == "v5camera":
                                print "#### marsCamera.__init__, self.driver == ", self.driver
				if "main" in marsguiref.WINDOW.keys():
					update_string = "Listening for camera discovery."
					gobject.idle_add(marsguiref.WINDOW["main"].update_status_bar, update_string)
				try:
					self.listener = listener = UDPListener.UDPListener()
					listener.listen()
                                        print "#### listener.queue_sizes: ", listener.queue_sizes
				except socket.timeout:
					self.marslog.print_log("Timed out getting marscamera client UDP broadcast packets", level="warning", method="__init__")
					self.camera_driver = [pyMarsCamera.marsCameraClient()]
					self.camera_driver[0].connect("192.168.0.13")
					self.chipcount = 1
					self.hv_driver = pyMarsCamera.marsCameraClient()
					self.camera_found = True
				else:
					self.chipcount = listener.queue_sizes["MarsCam"]
                                        print "#### chipcount", chipcount
					if "main" in marsguiref.WINDOW.keys():
						update_string = "%d camera sensor modules found. Connecting and testing."%(self.chipcount)
						gobject.idle_add(marsguiref.WINDOW["main"].update_status_bar, update_string)
					self.marslog.print_log("V5 camera found. Chipcount is:", self.chipcount, level="info", method="__init__")
					self.camera_driver = []
					to_remove = []
					for i in range(self.chipcount):
						try:
							self.camera_driver.append(listener.pop_marscamera_client())
						except Exception as msg:
							self.marslog.print_log("Exception in getting marscamera client", msg, level="error", method="__init__")
						else:
							#self.camera_driver[i].run_command("writeDigitalPots", ["59", "64", "65"])
							#self.camera_driver[i].write_DACs({"I_TP_BufferOut":250})
							try:
								self.marslog.print_log("Testing camera:", self.camera_driver[i].socket.getpeername(), level="info", method="__init__")
								#self.camera_driver[i].test_adc_tuning()
								if not self.camera_driver[i].test_matrix_read_write():
									to_remove.append(i)
							except Exception as msg:
								 self.marslog.print_log("Exception testing camera", msg, level="error")
								 to_remove.append(i)
					for key in to_remove[::-1]:
						marslog.print_log("Removing camera ", key, "after it failed to pass startup tests.", level="warning")
						self.camera_driver.pop(key)
						self.chipcount = self.chipcount - 1
					if self.chipcount > 0:
						self.camera_found = True
					else:
						self.camera_found = False
						self.marslog.print_log("Removed all readouts due to testing faults. Likely to encounter problems now.", level="error", method="__init__")
					try:
						self.hv_driver = listener.pop_hv_client()
						self.hv_ip_addr = self.hv_driver.socket.getpeername()[0]
						self.hv_mac_addr = get_mac_address([self.hv_ip_addr])[self.hv_ip_addr]

					except Exception as msg:
						self.marslog.print_log("Exception in getting hv client", msg, level="error", method="__init__")
			elif self.driver == "testing":
				self.camera_found = True
				pass
			self.reset()  # force a reset upon creation to set default DAC-vals
		except Exception as msg:
			self.marslog.print_log("marscamera.__init__:", msg, level="error", method="__init__")
			self.camera_found = False
#			raise

		if dacvals is not None:
			self.set_dacvals(dacvals)
		self.chipcount = self.get_attribute("chips")
		if self.chipcount is None:
			self.chipcount = 1
		self.medipix_type = self.get_chip_attribute("type")
		self.masks = numpy.zeros([self.chipcount, 256, 256], dtype='uint32')
		self._frames = numpy.zeros([self.chipcount, 256, 256], dtype='uint32')

		# Temperature monitor offset and gradient - from Medipix3.0 manual v19
		self._k_temp = [88.75 for chipid in range(self.chipcount)]
		self._m_temp = [-.6073 for chipid in range(self.chipcount)]

		if masks is not None:
			self.masks = masks
		self.load_config()
		try:
			self.set_attribute("peltier", 1500)  # STB: set peltier regulator to 1.5v based on P.Hilton temperature measurements
		except Exception as msg:
			self.marslog.print_log("marscamera.__init__():\tWarning: Peltier could not be set.", msg, level="warning", method="__init__")

		self.poll_thread = marsthread.new_thread(self.get_polling)
		
        def getChipCount(self):
                print "#### chip count is ",self._chipcount
                return self._chipcount
                
        def test(self):
                print  "marssystem test"

	def map_chipid(self):
		"""
		Enumerate and reorder the chipids based on the fuseid of the medipix chips
		and the same fuseid in the config file, allowing for arbitrary ordering
		by the hardware to be corrected.


		"""
		self.check_config_loaded()
		if self.config.driver == "testing" or "camera_testing" in sys.argv:
			self.chipmap = {}
			for i in range(self.chipcount):
				self.chipmap[i] = i
		else:
			fuseids = self.config.get_attr(["software_config", "fuseid"])
			config_mac_addresses = [self.config.get_attr(["software_config", "network", i, "mac_addr"]) for i in range(len(fuseids))]
			if self.driver == "v5camera":
				real_mac_addresses = get_mac_address([cd.address for cd in self.camera_driver])
				real_mac_addresses = [real_mac_addresses[cd.address] for cd in self.camera_driver]
			else:
				real_mac_addresses = get_mac_address(["192.168.0.44" for i in range(fuseids)]).values()
			reverse_chipmap = {}
			chipmap = {}
			mac_chipmap = {}
			mac_reverse_chipmap = {}
			for i in range(self.chipcount):
				for j in range(len(fuseids)):
					if int(fuseids[j], 16) == self.get_chip_attribute("fuses", chipid=i):
						chipmap[i] = j
						reverse_chipmap[j] = i
						continue
				for j in range(len(config_mac_addresses)):
					if real_mac_addresses[i] == config_mac_addresses[j]:
						mac_chipmap[i] = j
						mac_reverse_chipmap[j] = i
						continue

			mismatch = False

			missing_chipids = []
			missing_fuseids = []
			missing_mac_addresses = []
			# print when fuseids are not present
			for i in range(self.chipcount):
				if i not in chipmap.keys():
					missing_chipids.append(i)
					self.marslog.print_log("marscamera.map_chipid(): Fuse %s not found in config"%(hex(self.get_chip_attribute("fuses", chipid=i))),
							       level="warning", method="map_chipid")
					if i in mac_chipmap.keys():
						self.marslog.print_log("Able to substitute with mac address", real_mac_addresses[i], level="warning", method="map_chipid")
						chipmap[i] = mac_chipmap[i]
						reverse_chipmap[chipmap[i]] = i
					else:
						mismatch = True


			# print when config fuses are not used
			for i in range(len(fuseids)):
				if i not in reverse_chipmap.keys():
					missing_fuseids.append(i)
					self.marslog.print_log("marscamera.map_chipid(): Fuse %s in config but not used."%(fuseids[i]),
							       level="warning", method="map_chipid")


			if mismatch:
				self.marslog.print_log("Not all chips in camera have fuseids in config. Service is required.", level="warning", method="map_chipid")
				self.marslog.print_log("Proceeding with mapping", level="info", method="map_chipid")
				self.status = self.FUSEID_MISMATCH
				while len(missing_chipids) > 0:
					chipid = missing_chipids.pop(0)
					if len(missing_fuseids) > 0:
						# If there is an ID in the config that is unused, use this first.
						fuseid_index = missing_fuseids.pop(0)
						chipmap[chipid] = fuseid_index
						reverse_chipmap[fuseid_index] = chipid
						self.marslog.print_log("Mapping chipid %d with fuse %s to config entry %d with fuse %s"
								       %(chipid, hex(self.get_chip_attribute("fuses", chipid=chipid)), fuseid_index, fuseids[fuseid_index]),
								       level="warning", method="map_chipid")
					else:
						# If there is no ID in the config, then clone the config of the first fuseid and use that.
						fuseid_index = len(fuseids)
						fuseids[fuseid_index] = hex(self.get_chip_attribute("fuses", chipid=i))
						chipmap[chipid] = fuseid_index
						reverse_chipmap[fuseid_index] = chipid
						self.marslog.print_log("Copying the config from config ID %d with fuseid %s for chip %d with fuseid %s"
								       %(0, fuseids[0], chipid, hex(self.get_chip_attribute("fuses", chipid=chipid))), level="warning", method="map_chipid")
				if self.mode is not None:
					# actually update the config
					modes_dict = self.config.software_config.modes[self.mode]
					self.config.software_config.fuseid[fuseid_index] = fuseids[fuseid_index]
					keys = ["dacs", "energy_calibration", "masks", "mask_files"]
					for key in keys:
						modes_dict[key][fuseid_index] = modes_dict[key][0]
			else:
				self.status = self.STATUS_NORMAL


			if self.driver == "v5camera":
				# reorder the driver IDs to be in the same order as the fuseid in the config.
				self.chipmap = {}
				camera_driver_list = []

				for i in range(len(reverse_chipmap)):
					old_chip_pos = reverse_chipmap[sorted(reverse_chipmap.keys())[i]]
					camera_driver_list.append(self.camera_driver[old_chip_pos])
					self.chipmap[i] = chipmap[old_chip_pos]

				if self.chipcount != len(camera_driver_list):
					self.marslog.print_log("Map chipid has encountered a bug where it has reduced the chipcount.", level="error", method="map_chipid")
					for cd in self.camera_driver:
						if cd not in camera_driver_list:
							camera_driver_list.append(cd)
				self.camera_driver = camera_driver_list

				self.chipcount = len(self.camera_driver)
				print "mapchipid done:", self.camera_driver, self.chipcount
			else:
				self.chipmap = chipmap

	def update_dicom_mechanical_tags(self):
		self.check_config_loaded()

		try:
			chipx = numpy.array([self.config.get_attr(["mechanical_config", "chipx_pixels"])[self._get_chipid(i)] for i in range(self.chipcount)])
			chipy = numpy.array([self.config.get_attr(["mechanical_config", "chipy_pixels"])[self._get_chipid(i)] for i in range(self.chipcount)])
			chipz = numpy.array([self.config.get_attr(["mechanical_config", "chipz_pos"])[self._get_chipid(i)] for i in range(self.chipcount)])
			chipr = numpy.array([self.config.get_attr(["mechanical_config", "chipr_deg"])[self._get_chipid(i)] for i in range(self.chipcount)])
		except Exception as msg:
			self.marslog.print_log("exception reading camera chiparray geometry from camera mechanical config", level="warning", method="update_dicom_mechanical_tags")
			chipx = [0]
			chipy = [0]
			chipz = [0.0]
			chipr = [0.0]

		try:
			pixelxy = self.config.get_attr(["mechanical_config", "pixel_xy_dimension"], default=0.055)
		except Exception as msg:
			self.marslog.print_log("Could not get the xy dimension size of pixels.", msg, level="error", method="update_dicom_mechanical_tags")
			pixelxy = 0.055

		try:
			in_use_pixelxy = self.get_chip_attribute("pixelsize", chipid = 0)
		except Exception as msg:
			in_use_pixelxy = 0.110
			self.marslog.print_log("exception while getting 'in use' pixelsize from camera (setting to -> 110 um): ", msg, level="error", method="update_dicom_mechanical_tags")

		chipxy_multiplier = pixelxy / in_use_pixelxy

		try:
			camera_xcentre_mm = self.config.get_attr(["mechanical_config", "image_origin", 0])
			camera_ycentre_mm = self.config.get_attr(["mechanical_config", "image_origin", 1])
			if camera_xcentre_mm is None or camera_ycentre_mm is None:
				raise RuntimeError()  # Just so we try the next form of calculation.
		except Exception as msg:
			self.marslog.print_log("could not get the image_origin for the camera", msg, level="error", method="update_dicom_mechanical_tags")
			try:
				camera_xcentre_mm = pixelxy * (max(chipx) + 256 - min(chipx)) / 2.0
				camera_ycentre_mm = pixelxy * (max(chipy) + 256 - min(chipy)) / 2.0
			except Exception as msg:
				self.marslog.print_log("fail to calculate image origin from chip geometry data", msg, level="error", method="update_dicom_mechanical_tags")
				try:
					camera_xcentre_mm = 128 * pixelxy
					camera_ycentre_mm = 128 * pixelxy
				except Exception as msg:
					camera_xcentre_mm = 128 * 0.055
					camera_ycentre_mm = 128 * 0.055
				self.config.set_attr(["mechanical_config", "image_origin"], [float(camera_xcentre_mm), float(camera_ycentre_mm)])
		chipx_pixel00_to_center = (128 * 0.055)
		chipy_pixel00_to_center = (128 * 0.055)
		try:
			for i in range(len(chipx)):
				# working with real data so need multiplier (Argh! X and Y position are integers in pixel units for the frame. TODO: need physical positions in mm)
				marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "Chip_%d_frameX"%(i + 1), int( int(chipx[i]) * chipxy_multiplier)) #	int truncation is intentional. TODO: There should be a common method used both here and by build frame.
				marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "Chip_%d_XPosition_mm"%(i + 1), float(chipx[i] * pixelxy - camera_xcentre_mm) + chipx_pixel00_to_center)

			for i in range(len(chipy)):
				marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "Chip_%d_frameY"%(i + 1), int( int(chipy[i]) * chipxy_multiplier) )
				marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "Chip_%d_YPosition_mm"%(i + 1), float(chipy[i] * pixelxy - camera_ycentre_mm) + chipy_pixel00_to_center)

			for i in range(len(chipr)):
				marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "Chip_%d_RotationAngle"%(i + 1), float(chipr[i]))

			for i in range(len(chipx)):
				# no multiplier needed - working entirely in mechanical config coordinates
				marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "Chip_%d_XCenter_inmm"%(i + 1), float((chipx[i] + 128) * pixelxy - camera_xcentre_mm))

			for i in range(len(chipy)):
				marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "Chip_%d_YCenter_inmm"%(i + 1), float((chipy[i] + 128) * pixelxy - camera_ycentre_mm))

			#chipz = [0,0,0]
			for i in range(len(chipz)):
				marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "Chip_%d_ZPosition_mm"%(i + 1), float(chipz[i]))

		except Exception as msg:
			self.marslog.print_log("failed to write camera geometry to dicom tags.", msg, level="error", method="update_dicom_mechanical_tags")


	def _get_chipid(self, _chipid):
		return self.chipmap[_chipid]

	def reconnect(self):
		try:
			if self.driver == "libmarscamera":
				self.libmars_caller(libmarscamera_py.find)
				self.camera_found = True
			elif self.driver == "v5camera":
				self.connect('192.168.0.11')
			elif self.driver == "testing":
				pass
			self.reset_and_rebuild()  # This method does too exist. pylint: disable=E
		except Exception as msg:
			self.marslog.print_log("marsCamera.reconnect:", msg, level="error", method="reconnect")
			self.camera_found = False

	def hv_refresh(self):
		"""
		refreshes the HV supply.

		If it is on, leave it on. If its off, start the warmup phase.

		If warming up, check that the warmup time has finished and declare it as on.
		In all cases, if the idle control is on, then refresh the timeout timer.

		the warmup check is also done in hv_timer_check, and it might be ideal to turn it off here.
		I don't think it will cause any harm though.
		"""
		if self.hv_enabled == True:
			c_time = time()
			if self.hv_idle_control == True:
				self.hv_idle_timeout_start = c_time
			if self.hv_state == HV_WARMUP:
				if c_time - self.hv_warmup_start > self.hv_warmup_time:
					self.marslog.print_log("HV warmup complete.", method="hv_refresh")
					self.hv_state = HV_ON
					self.notify(key="hv")
			if self.hv_state == HV_OFF:
				marsthread.new_thread(self.set_hv, self.hv)
				self.hv_state = HV_WARMUP
				self.hv_warmup_start = c_time
				self.notify(key="hv")

	def set_hv_enable(self, hv_enable):
		"""
		Sets the hv_enabled flag. When True, the hv is enabled, although can
		still be turned off when idle. If it is disabled, it should always be off.
		"""
		if hv_enable == True and self.hv_enabled == False:  # turn hv_enabled on
			self.hv_enabled = True
			self.hv_refresh()  # This will start the warmup process with the right
		elif hv_enable == False and self.hv_enabled == True:  # turn hv_enabled off
			self.hv_enabled = False
			self.set_hv_off()

	def set_hv_off(self):
		"""
		Turn the HV off, assuming its valid to do so.
		"""
		if not (self.hv_idle_control == False and self.hv_enabled == True):
			##
			# When idle control is off, but the hv is on,
			# the hv should always be on. Ignore this function if so.
			#
			old_hv = self.hv
			self.set_hv(0)
			self.hv = old_hv
			self.hv_state = HV_OFF
			self.notify(key="hv")

	def hv_timer_check(self):
		"""
		To be called routinely from a threaded function.

		Change HV_WARMUP to HV_ON after the appropriate amount of time.

		turn off the HV if the idle_timer has timed out.
		"""
		if self.hv_enabled == True:
			c_time = time()
			if self.hv_state == HV_ON:
				if self.hv_idle_control:
					if c_time - self.hv_idle_timeout_start > self.hv_idle_timeout:
						self.marslog.print_log("HV idle timeout occurred. Turning HV off.", method="hv_timer_check")
						self.set_hv_off()
			elif self.hv_state == HV_WARMUP:
				if c_time - self.hv_warmup_start > self.hv_warmup_time:
					self.marslog.print_log("HV warmup complete.", method="hv_timer_check")
					self.hv_state = HV_ON
					self.notify(key="hv")

	def hv_wait_warmup(self, hang=True):
		if hang:
			self.marslog.print_log("Waiting for HV warmup.", method="hv_wait_warmup")
			while self.hv_wait_warmup(hang == False):
				sleep(0.2)
		else:
			if self.hv_state == HV_ON:
				return False
			else:
				self.hv_refresh()
				return True

	def set_hv_idle_control(self, control):
		"""
		Sets whether the idle control is enabled or not.

		NOTE: This still changes even when the hv_enabled flag is False.
		This is so when the hv_enabled flag is set to True, the system
		knows whether the idle control should be on or not.
		"""
		if control == True and self.hv_idle_control == False:
			self.hv_idle_control = True
		elif control == False and self.hv_idle_control == True:
			self.hv_idle_control = False
		self.hv_refresh()

	def load_default_dacvals(self):
		raise NotImplementedError("load_default_dacvals not implemented for this camera type")

	def are_DACs_initialised(self):
		return self.dacvals is not None

	def _check_exptime_bounds(self):
		"""
		Corrects any bad exposure times on the camera. exptime measured in ms
		"""
		if self.exptime < 1:
			self.exptime = 1

	def _check_chip_bounds(self):
		"""
		Corrects any bad chip identifiers for the camera.
		"""
		if self.chip < 0:
			self.chip = 0
		elif self.chip > self.chipcount - 1:
			self.chip = self.chipcount - 1

	# Member interaction methods
	"""
	These methods create properties, which act like variables but call functions when used.
	e.g. self.exptime = 0.2 actually calls a function exptime.fset(0.2), which we have set to
	be thread-safe.

	Several of the methods have been left the way they are, because they supply additional arguments
	to the function, which the property attribute doesn't allow. E.g. the chipid.
	"""

	@property
	@marsthread.reader
	def dacidxs(self):
		"""
		Filler method to be replaced in relevant classes
		"""
		return {}

	@property
	def dacranges(self):
		"""
		Filler method to be replaced in relevant classes
		"""
		return

	@property
	@marsthread.reader
	def chip(self):  # Properties don't get overwritten. pylint: disable=E
		return self._chipid

	@chip.setter  # Valid call. pylint: disable=E
	@marsthread.modifier
	def chip(self, chipId):  # Valid function declaration. pylint: disable=E
		self._chipid = int(chipId)
		self._check_chip_bounds()

	@property
	@marsthread.reader
	def polarity(self):  # Properties don't get overwritten. pylint: disable=E
		return self._polarity

	@polarity.setter  # Valid call. pylint: disable=E
	@marsthread.modifier
	def polarity(self, polarity):  # Valid function declaration. pylint: disable=E
		self._polarity = min(1, max(0, int(polarity)))

	@property
	@marsthread.reader
	def masks(self):  # Properties don't get overwritten. pylint: disable=E
		return self._masks

	@masks.setter  # Valid call. pylint: disable=E
	@marsthread.modifier
	def masks(self, masks):  # Valid function declaration. pylint: disable=E
		self._masks = numpy.array(masks)

	def _get_mask(self, chipid=None):
		if chipid is None:
			chipid = self.chip

		return self.masks[chipid]

	@marsthread.modifier
	def _set_mask(self, mask, chipid=None):
		if chipid is None:
			chipid = self.chip
		if chipid == "all":
			for i in range(self.chipcount):
				self._set_mask(mask, i)
			return
		else:
			self._masks[chipid] = mask

	@marsthread.reader
	def check_dac_name(self, dacname):
		for tmp in self.dacvals[0].keys():
			if tmp.lower() == dacname.lower():
				return tmp

	@marsthread.reader
	def _get_dac(self, dacname, chipid=None):
		if chipid is None:
			chipid = self.chip
		dacname = self.check_dac_name(dacname)
		return self.dacvals[chipid][dacname]

	@marsthread.modifier
	def _set_dac(self, dacname, dacval, chipid=None, ignore_lock=False):
		if self.has_dac_lock and not ignore_lock:
			self.marslog.print_log("marscamera._set_dac() called while dac lock active. Aborting.", level="error", method="_set_dac")
			import traceback
			traceback.print_stack()
			return

		if chipid is None:
			chipid = self.chip
		if chipid == "all":
			for i in range(self.chipcount):
				self._set_dac(dacname, dacval, i, ignore_lock=ignore_lock)
			return

		dacname = self.check_dac_name(dacname)
		if dacval >= RANGES[self.medipix_type][dacname]:
			dacval = RANGES[self.medipix_type][dacname] - 1
		elif dacval < 0:
			dacval = 0
		self._dacvals[chipid][dacname] = dacval

	@property
	@marsthread.reader
	def exptime(self):  # Properties don't get overwritten. pylint: disable=E
		return self._exptime

	@exptime.setter  # Valid call. pylint: disable=E
	@marsthread.modifier
	def exptime(self, exptime):  # Valid function declaration. pylint: disable=E
		"""
		sets exptime in ms
		"""
		self._exptime = float(exptime)
		self._check_exptime_bounds()

	@marsthread.reader
	def _get_frame(self, chipid=None, counter=0):
		if chipid is None:
			chipid = self.chip
		return self._frames[chipid]

	@marsthread.modifier
	def _set_frame(self, frame, chipid=None):
		if chipid is None:
			chipid = self.chip
		if chipid == "all":
			for i in range(self.chipcount):
				self._set_frame(frame, i)
			return
		self._frames[chipid] = frame

	def get_frame(self, chipid, counter=0, frame=None):
		frame = self._get_frame(chipid=chipid, counter=counter)
		return frame

#	@marsthread.reader
	def get_image(self, counter=0, OpenBeamCorrection=True, CreatePixelMasks=True, image_rotatefilter=None, image_expand=False):
		""" Returns a composite NxM image from the camera.
			valid kwargs:
				OpenBeamCorrection: Boolean
				CreatePixelMasks: Boolean
				NOTE: OpenBeamCorrection = True ,  implies CreatePixelMasks = True
				image_rotatefilter:  NEAREST, BILINEAR, BICUBIC
		"""
#		if OpenBeanCorrection: CreatePixelMasks=True
		self.marslog.print_log("This method is depreciated. Please report this error message to the development team.", level="error", method="get_image")
		self.check_config_loaded()
		try:
			N = max(self.config.get_attr(["mechanical_config", "chipx_pixels"]).values()) + 256
			M = max(self.config.get_attr(["mechanical_config", "chipy_pixels"]).values()) + 256
			self._image = Image.new('L', N, M)
			for chip in range(self.chipcount):
				x = self.config.get_attr(["mechanical_config", "chipx_pixels", chip])
				y = self.config.get_attr(["mechanical_config", "chipy_pixels", chip])
				rot = self.config.get_attr(["mechanical_config", "chipr_deg", chip])

				img = Image.new('L', [256, 256])
				# NOTE: base class get_frame accepts a counter argument but does nothing with it.
				b = self._get_frame(chipid=chip, counter=counter).astype('double')  # .clip(min,max) - min) / (max - min)
				if CreatePixelMasks == True:
					self._noncounting_mask[chip] = noncounting_pixel_mask(b, threshold=0.0)
					self._bright_mask[chip] = bright_pixel_mask(b, threshold=float(4096))

				if OpenBeamCorrection:
					pass
					# b = b / self.open_frames[chip]
				img.putdata(b.flatten(), scale=256)
				if rot in IMAGE_ROTATION.keys():				# Fast rotation for multiples of 90 deg
					img.transpose(IMAGE_ROTATION[rot])
				elif (float(rot) != 0.0):						# Rotation for other angles
					ROT_args = {}
					if image_rotatefilter.lower() in IMAGE_ROTATE_FILTER.keys():
						ROT_args["filter"] = IMAGE_ROTATE_FILTER[image_rotatefilter.lower()]
					if image_expand:
						ROT_args["expand"] = True

					img.rotate(rot, ROT_args)

				self._image.paste(img, (x, y))
		except Exception as msg:
			self._image = None
			self.marslog.print_log("camera._get_image(): error reading camera_mechanical config", msg, level="error", method="get_image")
		return self._image  # self._frames[chipid=0]

	def set_inhibits(self, inh_mask, chipid=None):
		"""
		Sets the inhibit mask on the chip.
		"""
		if chipid == "all":
			for i in range(self.chipcount):
				self.set_inhibits(inh_mask, i)
			return

		inh_mask = inh_mask.astype('int32')

		mask = self._get_mask(chipid)
		mask = mask & ~(self.inhibit_bit)
		if self.active_high:
			mask = mask + inh_mask * self.inhibit_bit
		else:
			mask = mask + (inh_mask == 0).astype('int32') * self.inhibit_bit
		self._set_mask(mask, chipid)

		self.load_config(chipid=chipid)

	def get_inhibits(self, chipid=None):
		"""
		Gets the inhibit mask of the chip.
		"""
		if chipid == "all":
			return [self.get_inhibits(i) for i in range(self.chipcount)]
		mask = self._get_mask(chipid)
		if self.active_high:
			inh_mask = ((mask & self.inhibit_bit) != 0).astype('int32')
		else:
			inh_mask = ((mask & self.inhibit_bit) == 0).astype('int32')
		return inh_mask

	def build_multiframe(self, frames=None, enabled_chips="all"):  # not fully implemented..
		"""
		Stitch images from different chips, and concatenate energies

		e.g.

		+------------+------------+
		| Energy 1                |
		+============+============+
		|            |            |
		|  Chip 4    |  Chip 3    |
		|            |            |
		+------------+------------+
		|            |            |
		|  Chip 2    |  Chip 1    |
		|            |            |
		+------------+------------+

		+------------+------------+
		| Energy 2                |
		+============+============+
		|            |            |
		|  Chip 4    |  Chip 3    |
		|            |            |
		+------------+------------+
		|            |            |
		|  Chip 2    |  Chip 1    |
		|            |            |
		+------------+------------+

		"""
		if frames is None:
			frames = self._get_frames()

		if enabled_chips == "all":
			enabled_chips = [True for i in range(self.chipcount)]

		first_frame = self.frame_order[0]

		frame = self.build_frame(counter=first_frame, frames=frames, enabled_chips=enabled_chips)

		imagesize = numpy.array(frame).squeeze().shape
		ret_img = numpy.zeros([self.counters * (imagesize[0]), imagesize[1]])
		ret_img[0:imagesize[0], 0:imagesize[1]] = frame

		for counter in range(1, self.counters):
#		for counter in self.frame_order: # oops, not ready for this yet
			ret_img[counter * imagesize[0]:(counter + 1) * imagesize[0], 0:imagesize[1]] = self.build_frame(counter=counter, frames=frames, enabled_chips=enabled_chips)

		return ret_img

	def unpack_multiframe(self, multiframe):
		"""
		Unpacks a multiframe as built by "build_multiframe"
		to the same format as expected from "_get_frames"
		"""
		frames = []
		for counter in range(self.counters):
			imagesize = [multiframe.shape[0] / self.counters, multiframe.shape[1]]
			frames.append(self.unpack_frame(multiframe[imagesize[0] * counter:imagesize[0] * (counter + 1), 0:imagesize[1]], counter))
		frames = numpy.array(frames)
		return frames.sum(0)
		#frames0 = frames[::2,:,:].sum(0)
		#frames1 = frames[1::2,:,:].sum(0)
		#return numpy.array([frames0, frames1])


	def build_frame(self, counter=0, frames=None, enabled_chips="all"):
		"""
		Stitch images from different chips

		e.g.

		+-----------+----------+
		|           |          |
		|  Chip 4   |  Chip 3  |
		|           |          |
		+-----------+----------+
		|           |          |
		|  Chip 2   |  Chip 1  |
		|           |          |
		+-----------+----------+
		"""
		self.check_config_loaded()
		if frames is None:
			frames = self._get_frames()

		if enabled_chips == "all":
			enabled_chips = [True for i in range(self.chipcount)]

		posxChips = numpy.array([self.config.get_attr(["mechanical_config", "chipx_pixels"])[self._get_chipid(i)] for i in range(self.chipcount)])
		posyChips = numpy.array([self.config.get_attr(["mechanical_config", "chipy_pixels"])[self._get_chipid(i)] for i in range(self.chipcount)])

		try:
			if str(self.__class__.__name__) in ["marsCameraMP3", "marsCamera3RX"]:
				if self.colourmode:
					posxChips = posxChips / 2
					posyChips = posyChips / 2
		except Exception as msg:
			pass

		maxposx = 0
		maxposy = 0
		minposx = 200000000
		minposy = 200000000
		for chip in range(self.chipcount):
			if enabled_chips[chip]:
				maxposx = max(maxposx, posxChips[chip])
				minposx = min(minposx, posxChips[chip])
				maxposy = max(maxposy, posyChips[chip])
				minposy = min(minposy, posyChips[chip])

		if minposx == 200000000:
			minposx = 0
		if minposy == 200000000:
			minposy = 0
		imagesize = numpy.array(self.get_frame(0, counter, frames[0])).squeeze().shape
                print "#### numpy.zeros", imagesize[0] + maxposy - minposy, imagesize[1] + maxposx - minposx
		ret_img = numpy.zeros([imagesize[0] + maxposy - minposy, imagesize[1] + maxposx - minposx])

		for chipid in range(self.chipcount):
			if enabled_chips[chipid]:
				chipRot = self.config.get_attr(["mechanical_config", "chipr_deg", self._get_chipid(chipid)])
				rotation = lambda angle: int((angle + 45) % 360) / 90
				if rotation(chipRot) != 0:
					ret_img[posyChips[chipid] - minposy:posyChips[chipid] + imagesize[0] - minposy, posxChips[chipid] - minposx:posxChips[chipid] + imagesize[1] - minposx] = numpy.array([numpy.rot90(self.get_frame(chipid, counter, frames[chipid]), rotation(chipRot))]).squeeze()
				else:
					ret_img[posyChips[chipid] - minposy:posyChips[chipid] + imagesize[0] - minposy, posxChips[chipid] - minposx:posxChips[chipid] + imagesize[1] - minposx] = numpy.array([self.get_frame(chipid, counter, frames[chipid])]).squeeze()

		return ret_img

	def unpack_frame(self, frame, counter):
		"""
		Converts a frame built by "build_frame" back to the
		same format as expected from "_get_frames"
		"""
		frame = frame.astype('uint32')
		posxChips = numpy.array([self.config.get_attr(["mechanical_config", "chipx_pixels"])[self._get_chipid(i)] for i in range(self.chipcount)])
		posyChips = numpy.array([self.config.get_attr(["mechanical_config", "chipy_pixels"])[self._get_chipid(i)] for i in range(self.chipcount)])

		imagesize = [256, 256]
		try:
			if str(self.__class__.__name__) in ["marsCameraMP3", "marsCamera3RX"]:
				if self.colourmode:
					posxChips = posxChips / 2
					posyChips = posyChips / 2
					imagesize = [128, 128]
		except Exception as msg:
			pass

		maxposx = 0
		maxposy = 0
		minposx = 200000000
		minposy = 200000000
		for chip in range(self.chipcount):
			maxposx = max(maxposx, posxChips[chip])
			minposx = min(minposx, posxChips[chip])
			maxposy = max(maxposy, posyChips[chip])
			minposy = min(minposy, posyChips[chip])

		if minposx == 200000000:
			minposx = 0
		if minposy == 200000000:
			minposy = 0

		frames = []

		for chipid in range(self.chipcount):
			chipRot = self.config.get_attr(["mechanical_config", "chipr_deg", self._get_chipid(chipid)])
			rotation = lambda angle: int((360 - angle + 45) % 360) / 90
			frame_to_rot = frame[posyChips[chipid] - minposy:posyChips[chipid] + imagesize[0] - minposy,
					       posxChips[chipid] - minposx:posxChips[chipid] + imagesize[1] - minposx]
			if rotation(chipRot) != 0:
				frames.append(self.remux_frame(numpy.array(numpy.rot90(frame_to_rot, rotation(chipRot))).squeeze(), counter=counter))
			else:
				frames.append(self.remux_frame(numpy.array(frame_to_rot).squeeze(), counter=counter))


		return numpy.array(frames).astype('uint32')


	def remux_frame(self, frame, counter=0):
		"""
		Multiplexes the frame back into the bits and
		pixels of the 256x256 24-bit image as libmars
		treats them.
		"""
		frame = frame.astype('uint32')

		counter = self.parse_csm_counter(counter)

		if not self.colourmode:
			if counter == 0 and self.equalizethh == False or counter == 1 and self.equalizethh:
				return frame & 0x000FFF
			elif counter == 1:
				return (frame << 12) & 0xFFF000
			else:
				return frame
		else:
			if counter in [0, 1, 4, 5]:
				i = 0
			else:
				i = 1
			if counter in [0, 1, 2, 3]:
				j = 1
			else:
				j = 0
			ret = numpy.zeros([256, 256]).astype('uint32')
			if (counter in [0, 2, 4, 6] and self.equalizethh == False) or (counter == [1, 3, 5, 7] and self.equalizethh):
				_tempFrames = frame & 0x000FFF
				ret[i::2, j::2] =  _tempFrames
				return ret
			elif counter in [1, 3, 5, 7]:
				_tempFrames = (frame << 12) & 0xFFF000
				ret[i::2, j::2] =  _tempFrames
				return ret
			else:
				return frame

	@marsthread.reader
	def _get_frames(self, counter=0):
		return self._frames

	@marsthread.modifier
	def _set_frames(self, frames):
		self._frames = frames

	@property
	@marsthread.reader
	def config(self):  # Properties don't get overwritten. pylint: disable=E
		return self._config

	@config.setter  # Valid call. pylint: disable=E
	@marsthread.modifier
	def config(self, config):  # Valid function declaration. pylint: disable=E
		self._config = config
		self._config_loaded = time()

	@marsthread.reader
	def _get_time_since_config_load(self):
		return time() - self._config_loaded

	@marsthread.modifier
	def _set_mode(self, mode_name, mode_dict):
		self._modes[mode_name] = mode_dict  # This does exist. pylint: disable=E

	@marsthread.reader
	def _get_mode(self, mode_name):
		return self._modes[mode_name]  # This does exist. pylint: disable=E

	@marsthread.modifier
	def _set_mode_map(self, mode_name, m):
		self._mode_map[mode_name] = m  # This does exist. pylint: disable=E

	@marsthread.reader
	def _get_mode_map(self, mode_name):
		# mode_name may not exist in _mode_map when a protocol is loaded for an old system configuration. (mode name got changed?)
		return self._mode_map[mode_name]  # This does exist. pylint: disable=E

	# Full camera commands (applied over all chips or chip-independant).

	def set_colourMode(self, is_cm):
		"""
		Dummy method
		"""
		pass

	def set_csm(self, is_csm):
		"""
		Dummy method
		"""
		pass

	def set_full_counter(self, fc):
		"""
		Dummy method
		"""
		pass

	def set_mode(self, mode_args):
		try:
			if self.config is None:
				self.marslog.print_log("camera config not loaded", level="warning", method="set_mode")
#				camera_config = marsconfig.get("camera_config")
#				self.update_config(camera_config.software_config)
				self.check_config_loaded()
				self.update_config(self.config)
			try:
				if isinstance(mode_args, dict):
					try:
						mode = self._get_mode_map(mode_args["mode"])
					except:
						# This exception can be triggered when the mode (name) originates from a saved protocol or image and is
						# no longer present in the current configuration.
						self.marslog.print_log("marscamera.set_mode: failed to get mode ->", mode_args["mode"], " loading default mode instead.", level="warning", method="set_mode")
						mode = self.config.get_attr(("software_config","default_mode"), default="colour_csm_full")
						mode_name = self.config.get_attr(("software_config","modes", mode, "name"))
						self.marslog.print_log("marscamera.set_mode: attempting to load ->", mode_name,":",str(mode), level="warning", method="set_mode")
				else:
					mode = mode_args

				self.marslog.print_log("got mode:", mode, level="debug", method="set_mode")
				self.mode = mode
				mode_dict = self._get_mode(mode)

				for key in mode_dict.keys():
					self.marslog.print_log(key, mode_dict[key], level="debug", method="set_mode")


				if ("energy_calibration" in mode_dict.keys()):
					if self.energy_calibration is None:
						self.energy_calibration = {}
					for chipid in range(self.chipcount):
						_chipid = self._get_chipid(chipid)
						if _chipid in mode_dict["energy_calibration"].keys():
							self.energy_calibration[chipid] = mode_dict["energy_calibration"][_chipid]
				#=============== ColourMode switching ===============================================================
				if ("features" in mode_dict.keys()) and ('colourmode' in mode_dict["features"].keys()):
					if self._type in [MP3_TYPE, RX3_TYPE]:
						# only need to switch colourmode on/off in non CSM modes
						self.set_colourMode(is_cm=bool(mode_dict["features"]['colourmode']))
				#=============== CSM switching ======================================================================
				# always needs to load a new mask if csm mode is switched
				if ("features" in mode_dict.keys()) and ('csm' in mode_dict["features"].keys()):
					if self._type in [MP3_TYPE, RX3_TYPE]:
						if self.csm != bool(mode_dict["features"]['csm']):  # if csm mode is being switched
							self.set_csm(is_csm=bool(mode_dict["features"]['csm']))  # set csm  on/off state
				if ("features" in mode_dict.keys()) and ('fullcounter' in mode_dict["features"].keys()):
					if self._type in [MP3_TYPE, RX3_TYPE]:
						if self.full_counter != bool(mode_dict["features"]["fullcounter"]):
							self.set_full_counter(bool(mode_dict["features"]["fullcounter"]))
				if ("masks" in mode_dict.keys()):
					for chipid in range(self.chipcount):
						_chipid = self._get_chipid(chipid) # the chipid of the config
						try: # This exception gets triggered when first setting up masks in a new protocol.
							if _chipid in mode_dict["masks"].keys():
								self._set_mask(mode_dict["masks"][_chipid], chipid=chipid)
						except Exception as msg:
							self.marslog.print_log("error accessing mode_dict[masks].keys(): ", mode_dict, msg, level="debug", method="set_mode")


					self.load_config()
				if ("dacs" in mode_dict.keys()):
					if self.dacvals is None:
						self.dacvals = {}
					for chipid in range(self.chipcount):
						_chipid = self._get_chipid(chipid) # the chipid of the config
						if _chipid in mode_dict["dacs"].keys():
							if chipid not in self.dacvals.keys():
								self.dacvals[chipid] = {}
							for dac in mode_dict["dacs"][_chipid].keys():
								if dac not in self.dacvals[_chipid].keys():
									self.dacvals[chipid][dac] = mode_dict["dacs"][_chipid][dac]
								self.set_dac(dac, mode_dict["dacs"][_chipid][dac], chipid=chipid)
			except Exception as msg:
				self.marslog.print_log("marscamera: exception handling the camera mode change", msg, level="error", method="set_mode")
			else:
				self.notify(Message(mode=self.mode), key="mode_change")

			self.update_dicom_mechanical_tags()  # image size and chip position (in pixel units) in the pseudo stitched images depends on operating mode.

		except Exception as msg:
			self.marslog.print_log("exception in camera mode change", msg, level="error", method="set_mode")

		return

	# First do commands that call libmars
	def set_attribute(self, key, val, allow_retries=True):
		"""
		Set the <val> of the camera <key> attribute.
		(See Medipix manual.)
		Returns None.
		"""
		try:
			if self.driver == "libmarscamera":
				self.libmars_caller(libmarscamera_py.setAttribute, self.devid, key, val)
			elif self.driver == "v5camera":
				if key.lower() == "hv":
					self.hv_driver.set_hv(int(val), allow_retries=allow_retries)
				elif key.lower() == "peltier":
					pass
				elif key in self.camera_driver[0].OMR:
					for cd in self.camera_driver:
						cd.write_OMR({key:val})
			elif self.driver == "testing":
				pass
			self.notify(key="attribute")
		except Exception as msg:
			self.marslog.print_log("marscamera.set_attribute:%s" % (str(msg)), level="error", method="set_attribute")
			raise

	@marsthread.modifier
	def _set_downloaded_mask(self, key, val):
		if not hasattr(self, "downloaded_mask"):
			self.downloaded_mask = {}
		self.downloaded_mask[key] = val

	@marsthread.reader
	def _get_downloaded_mask(self, key):
		if hasattr(self, "downloaded_mask"):
			return self.downloaded_mask[key]
		else:
			return None

	def load_config(self, chipid="all"):
		"""
		Helper to load config equalisation mask array onto the chip.

		Try up to 10 times per chip before giving up.

		Stores the redownloaded mask in an array called downloadedMask,
		to check the uploads are working correctly.
		"""
		if chipid == "all":
			for i in range(self.chipcount):
				self.load_config(i)
			return
		else:
			mask = self._get_mask(chipid).flatten().astype('uint32')
			if self.driver == "libmarscamera":
				self.libmars_caller(libmarscamera_py.loadConfig, self.devid, chipid, mask)
				self.libmars_caller(libmarscamera_py.downloadMask, self.devid, chipid)
				self._set_downloaded_mask(chipid, numpy.array(self.libmars_caller(libmarscamera_py.getImage, self.devid, chipid)))
			elif self.driver == "v5camera":
				self.camera_driver[chipid].upload_image((mask & 0xFFF).reshape([256, 256]), 0)
				self.camera_driver[chipid].upload_image(((mask & 0xFFF000).reshape([256, 256]) >> 12), 1)
				#self.camera_driver.upload_image(numpy.zeros([256, 256]), 0)
				#self.camera_driver.upload_image(numpy.zeros([256, 256]), 1)
				self._set_downloaded_mask(chipid, mask)
				try:
					self.camera_driver[chipid].acquire(1)
					self.camera_driver[chipid].acquire(1)
					self.camera_driver[chipid].download_image(0)
					self.camera_driver[chipid].download_image(1)
				except Exception as msg:
					self.marslog.print_log("Exception in acquiring images after mask upload", msg, level="error", method="load_config")
			elif self.driver == "testing":
				self._set_downloaded_mask(chipid, mask)
			if (self._get_downloaded_mask(chipid).flatten() != self._get_mask(chipid).flatten()).any():
				self.marslog.print_log("marscamera.load_config: Mask downloaded different to mask uploaded.", level="debug", method="load_config")

	def load_dacs(self):
		"""
		Sets the camera dacs according to dacvals dictionary.
		"""
		exc = None
		dacvals = self.dacvals
		for i in range(self.chipcount):
			for k in dacvals[i].keys():
				try:
					if verbose:
						self.marslog.print_log("Setting DAC", k, " to ", dacvals[i][k], level="info", method="load_dacs")
					dac_value = dacvals[i][k]
					self._set_dac(k, dac_value,  chipid=i)
					dac_value = self._get_dac(k, chipid=i)
					if self.driver == "libmarscamera" and (not self.has_dac_lock):
						self.libmars_caller(libmarscamera_py.setDac, self.devid, i, k, int(dac_value))
					elif self.driver == "testing":
						pass
				except Exception as e:
					self.marslog.print_log("marscamera.load_dacs:%s" % (str(e)), level="error", method="load_dacs")
					exc = e

		if exc is not None:
			raise exc  # Only raising the exception. pylint: disable=E
		else:
			self.notify(key="dac")

	def get_attribute(self, key):
		"""
		Return the value of the camera <key> attribute.
		(See Medipix manual.)
		"""
		if self.driver == "libmarscamera":
			if key == "hv":
				self.check_config_loaded()
				if not hasattr(self, "hv_adc_slope"):
					self.hv_adc_slope = slope = self.config.get_attr(["software_config", "hv_corrections", self.hv_mac_addr, "adc_slope"], default=1.0)
				else:
					slope = self.hv_adc_slope
				if not hasattr(self, "hv_adc_offset"):
					self.hv_adc_offset = offset = self.config.get_attr(["software_config", "hv_corrections", self.hv_mac_addr, "adc_offset"], default=0.0)
				else:
					offset = self.hv_adc_offset
				if self.hv_fixed_output_count >= 5:
					return self.config.get_attr(["software_config", "hv_v"], default=65535)
				good_hv = False
				for i in range(1):  # number of tries
					ret = self.libmars_caller(libmarscamera_py.getAttribute, self.devid, key)
					if isinstance(ret, (int, float)) and ret < 2000 and ret >= 0:
						if ret > 0:
							ret = float(ret - offset) / slope
							self.last_good_hv = ret
						good_hv = True
						break
				if not good_hv:  # Invalid return value from libmars_caller, set the ret to something useful.
					self.marslog.print_log("marscamera.get_attribute: Could not retrieve a valid HV reading from board: value=", ret, level="debug", method="get_attribute")
					if self.last_good_hv < 2000:
						bias_V = self.last_good_hv
						self.marslog.print_log("marscamera HV: Assuming a communication error. Using last measured value: ", bias_V, level="debug", method="get_attribute")
					else:
						bias_V = self.config.get_attr(["software_config", "hv_v"], default=ret)
						self.marslog.print_log("marscamera HV: Assuming a fixed output bias voltage board with no ADC. Using configured value: ", bias_V, level="debug", method="get_attribute")
						self.hv_fixed_output_count = self.hv_fixed_output_count + 1
						if self.hv_fixed_output_count >= 5:
							self.marslog.print_log("marsCamera HV: Unable to read any good high voltage measurements. Assuming fixed bias voltage output.", level="warning", method="get_attribute")

					ret = bias_V

				return ret

			elif key == "hv_a":
				return 0.0 # pass
			else:
				return self.libmars_caller(libmarscamera_py.getAttribute, self.devid, key)
		elif self.driver == "v5camera":
			if key in self.camera_driver[0].OMR:
				return self.camera_driver[0].OMR[key]
			elif key == "chips":
				return len(self.camera_driver)
			elif key == "revision":
				return RX3_REVISION
			elif key == "temp":
				return self.camera_driver[0].get_temperature()
			elif key == "hv":
				self.check_config_loaded()
				if not hasattr(self, "hv_adc_slope"):
					self.hv_adc_slope = slope = self.config.get_attr(["software_config", "hv_corrections", self.hv_mac_addr, "adc_slope"], default=1.57)
				else:
					slope = self.hv_adc_slope
				if not hasattr(self, "hv_adc_offset"):
					self.hv_adc_offset = offset = self.config.get_attr(["software_config", "hv_corrections", self.hv_mac_addr, "adc_offset"], default=0.0)
				offset = self.hv_adc_offset
				return round((float(self.hv_driver.read_hv()) * slope + offset), 2)
			elif key == "hv_a":
				return round((float(self.hv_driver.read_hv_current())), 2)
			elif key == "peltier":
				return 1500
			else:
				return 0
		elif self.driver == "testing":
			if key == "chips":
				if self.config is not None:
					return self.config.get_attr(["mechanical_config", "chip_count"], default=1)
				else:
					return 1
			else:
				return 0

	def reset(self):
		"""
		Reset the chip.
		"""
		try:
			if self.driver == "libmarscamera":
				pass
				#self.libmars_caller(libmarscamera_py.resetChip, self.devid, self.chip)
			elif self.driver == "testing":
				pass
			self.notify(key="reset")
		except Exception as msg:
			self.marslog.print_log("marscamera.reset:%s" % (str(msg)), level="error", method="reset")
			raise

	@marsthread.hasRWLock
	def acquire(self, exptime=None, chipid="all", attempts=0):
		"""
		Start camera acquisiton for <exptime> milliseconds.
		Retrieve the data and set the frame array.
		"""
		okay = True
		self.hv_refresh()
		if exptime is None:
			exptime = self.exptime
		if chipid is None:
			chipid = self.chip
		if self.driver == "libmarscamera":
			self.libmars_caller(libmarscamera_py.acquire, self.devid, self.chip, float(exptime) / 1000)
		elif self.driver == "v5camera":
			if chipid == "all":
				camera_driver_list = self.camera_driver
			else:
				camera_driver_list = [self.camera_driver[chipid]]
			threads = []
			if len(camera_driver_list) > 1:
				#print "starting multiframe acquire on slaves"
				for cd in camera_driver_list[1:]:
					threads.append(marsthread.new_thread(cd.multiframe_acquire, exptime_ms=exptime, frame_count=1, sync_mode="slave"))
				_t = time()
				for cd in camera_driver_list[1:]:
					while cd.mf_state not in [cd.MULTIFRAME_RUNNING, cd.MULTIFRAME_FAILED_START]:
						if time() - _t > 5.0:
							print "\n\n******Timeout waiting for slave to start", cd.address, "*****\n\n"
							break
						sleep(5e-3)
			try:
				#print "starting multiframe acquire on master"
				camera_driver_list[0].multiframe_acquire(exptime_ms=exptime, frame_count=1, sync_mode="master")
				#print "multiframe acquire finished on master"
			except Exception as msg:
				self.marslog.print_log("Exception when calling multiframe acquire on master", msg, level="warning", method="acquire")
				okay = False

			for thr in threads:
				thr.join()

			if okay is True:
				for cd in camera_driver_list:
					if cd.mf_state == cd.MULTIFRAME_FAILED_START:
						okay = False
						break
		elif self.driver == "testing":
			self.check_config_loaded()
			readout_delay = self.config.get_attr(["software_config", "testing_readout_delay_ms"], default=0.1)
			sleep((float(exptime) + float(readout_delay)) / 1000.0)
		if chipid == "all":
			chips = range(self.chipcount)
		else:
			chips = [chipid]
		if okay:
			for i in chips:
				if self.driver == "libmarscamera":
					self._set_frame(self.libmars_caller(libmarscamera_py.getImage, self.devid, i), i)
				elif self.driver == "v5camera":
					attempts = 0
					frames = self.camera_driver[i].get_multiframes() # this is two frames, counterL and counterH.
					while len(frames) < 2 and attempts < 15:
						self.marslog.print_log("Failed to finish a multiframe acquire. Retrying.", level="debug", method="acquire")
						attempts += 1
						try:
							# [ORIGINAL bug line]frames = cd.multiframe_acquire(exptime, 1)
							# [FIXME] patch to re-acquire from just the failing chip. [Assume re-acquire is needed here, not just re-getting the frames]
							self.camera_driver[i].multiframe_acquire(exptime_ms=exptime, frame_count=1, sync_mode="master")
							frames = self.camera_driver[i].get_multiframes()
						except Exception as msg:
							self.marslog.print_log("Unable to finish acquire command", msg, level="warning", method="acquire")
							sleep(0.010)
					if attempts >= 15:# or len(frames) < 2:
						self.marslog.print_log("Frame acquisition failed 15 times in a row. Aborting current frame, expect lost data.", level="error", method="acquire")
						return
					elif len(frames) < 2:
						self.marslog.print_log("Frame acquisition failed to return two frames. Aborting current frame, expect lost data.", level="error", method="acquire")
						return
					frames = sorted(frames, key=lambda x: x.counter) # Make sure the counter order is fixed.
					try:
						self._set_frame(frames[0].get_image().astype('uint32') + (frames[1].get_image().astype('uint32') << 12), i)
					except Exception as msg:
						self.marslog.print_log("marscamera: unable to set image frames", level="warning", method="acquire")
						self.marslog.print_log("marscamera._set_frame:",msg, level="debug", method="acquire")
				elif self.driver == "testing":
					self._set_frame(numpy.zeros([256, 256]), i)
		else:
			if attempts < 5:
				self.marslog.print_log("Acquire failed.. retrying", level="warning", method="acquire")
				sleep(2.0)
				marsCamera.acquire(self, exptime, chipid, attempts+1)
			else:
				self.marslog.print_log("Acquire failed but too many failures in a row..", level="error", method="acquire")
				raise Exception("Acquire failed too many failures in a row")
		self.notify(key="image")

	def abort_multiple_acquire(self, chipid="all"):
		if self.driver == "v5camera":
			if chipid == "all":
				camera_driver_list = self.camera_driver
			else:
				camera_driver_list = [self.camera_driver[chipid]]
			for cd in camera_driver_list:
				cd.cancel_multiframe()

	def multiple_acquire(self, num_frames, exptime=None, spacing_time=0.0, chipid="all", while_waiting_fn=None, attempts=0):
		"""
		Does a multiframe acquire, getting num_frames images with a shutter of exptime (ms) and
		a frame spacing/interval of spacing_time (ms).

		Returns the resulting frames back in an 3-d list (chipid, counter, frame_index), with
		each element holding a marsCameraImage object, holding timestamp, frame_index, exposure time,
		frame id info and the actual Medipix image, which can be retrieved with "get_image"
		"""
		if exptime is None:
			exptime = self.exptime
		if self.driver not in ["v5camera"]:
			self.marslog.print_log("Multiple acquire mode not accessible without a v5 camera", level="error", method="multiple_acquire")
			assert False
		if self.driver == "v5camera":
			if chipid == "all":
				camera_driver_list = self.camera_driver
			else:
				camera_driver_list = [self.camera_driver[chipid]]

			okay = True
			threads = []
			if len(camera_driver_list) > 1:
				for cd in camera_driver_list[1:]:
					threads.append(marsthread.new_thread(cd.multiframe_acquire, exptime_ms=exptime, frame_count=num_frames, spacing_time_ms=spacing_time, sync_mode="slave", while_waiting_fn=while_waiting_fn))
				_t = time()
				for cd in camera_driver_list[1:]:
					while cd.mf_state != cd.MULTIFRAME_RUNNING:
						if time() - _t > 5.0:
							if REPORT_TIMEOUT:
								print "\n\n******Timeout waiting for slave to start", cd.address, "*****\n\n"
							self.slave_timeout_count = self.slave_timeout_count + 1
							break
						sleep(5e-4)

			try:
				camera_driver_list[0].multiframe_acquire(exptime_ms=exptime, frame_count=num_frames, spacing_time_ms=spacing_time, sync_mode="master", while_waiting_fn=while_waiting_fn)
			except Exception as msg:
				okay = False

			for thr in threads:
				while thr.is_alive():
					sleep(10e-4)
					thr.join(10e-4)


			if okay is True:
				for cd in camera_driver_list:
					if cd.mf_state == cd.MULTIFRAME_FAILED_START:
						okay = False
						break
			if okay is False:
				for cd in camera_driver_list:
					cd.cancel_multiframe()
			if okay:
				ret_frames = []
				for cd in camera_driver_list:
					frames = cd.get_multiframes()
					frames = sorted(frames, key=lambda x: x.counter)
					frames2 = [sorted(frames[0:num_frames], key=lambda x: x.frameindex), #c0
						   sorted(frames[num_frames:], key=lambda x: x.frameindex)] #c1
					ret_frames.append(frames2)
				return ret_frames # [chipid, counter, frame_index]

			else:
				if attempts < 5:
					self.marslog.print_log("Multiple acquire failed.. retrying", level="warning", method="multiple_acquire")
					sleep(2.0)
					return marsCamera.multiple_acquire(self, num_frames, exptime, spacing_time, chipid, while_waiting_fn, attempts+1)
				else:
					self.marslog.print_log("Multiple acquire failed but too many failures in a row..", level="error", method="multiple_acquire")
					raise Exception("Multiple acquire failed too many failures in a row")


	def expose(self, exptime=None):
		"""
		Start the camera exposure for <exptime> milliseconds.
		"""
		if exptime is None:
			exptime = self.exptime

		if self.driver == "libmarscamera":
			self.libmars_caller(libmarscamera_py.expose, self.devid, self.chip, float(exptime) / 1000)
		elif self.driver == "v5camera":
			for cd in self.camera_driver:
				cd.acquire(exptime)

	def download_images(self):
		"""
		Retrieve the data from the MARS readout camera and set the frame array.
		"""
		if self.driver == "libmarscamera":
			self.libmars_caller(libmarscamera_py.downloadImage, self.devid, self.chip)
		elif self.driver == "v5camera":
			for cd in self.camera_driver:
				cd.download_image(0)
				cd.download_image(1)
		for i in range(self.chipcount):
			if self.driver == "libmarscamera":
				self._set_frame(self.libmars_caller(libmarscamera_py.getImage, self.devid, i), i)
			elif self.driver == "v5camera":
				self._set_frame(self.camera_driver[i].frame[0].astype('uint32') + (self.camera_driver[i].frame[1].astype('uint32') << 12), i)
			elif self.driver == "testing":
				self._set_frame(numpy.zeros([256, 256]), i)
		self.notify(key="image")

	# And then do commands that do not call libmars directly, but still apply over the whole camera

	def set_masks(self, masks, chipid=None):
		"""
		Set the configuration matrix (adjustment mask) array and load the mask
		onto the chip. Convert the mask to uint32 if necessary.
		(See Medipix manual.)
		Returns None.
		"""
		if chipid is None:
			chipid = self.chip
		if len(masks.shape) == 3:
			self.masks = masks
			chipid = "all"
		elif len(masks.shape) == 2:
			self._set_mask(masks, chipid)
		self.load_config(chipid=chipid)

	def set_dacvals(self, dacvals):
		"""
		Set the dacvals dictionary variable and load the values onto the chip.
		(See Medipix manual.)
		"""
		self.dacvals = dacvals
		self.load_dacs()

	def set_polarity(self, polarity=1):
		"""
		Set ASIC input polarity (default=1 set for Si, 0 for CdTe and GaAs).
		"""
		self.polarity = polarity
		self.marslog.print_log("set polarity:", polarity, level="debug", method="set_polarity")
		self.set_chip_attribute("Polarity", polarity, chipid="all")
		self.notify(key="polarity")

	def update_get_hv(self):
		"""
		Update the high voltage, but always return False (for gtk idle loops)
		"""
		self.get_hv()
		return False

	def get_bias_current(self):
		if self.camera_found == False:
			return -1.0
		bias_a = self.get_attribute("hv_a")

		self.read_bias_current = bias_a
		return bias_a

	def get_hv(self):
		if self.camera_found == False:
			return -1.0
		try:
			bias_v = self.get_attribute("hv")
			if type(bias_v) in [type(0),type(0.0)]:
				ret = bias_v + self.base_hv
			else:
				self.marslog.print_log("sensor bias voltage read failed. ", level="warning", method="get_hv")
				ret = self.base_hv
		except Exception as msg:
			self.marslog.print_log("exception reading sensor bias voltage. ",msg, level="error", method="get_hv")
			ret = 0.0
		self._append_hv((time() - self.start_time, ret))
		self.read_hv = ret
		return ret

	def set_hv(self, hv=1, hv_step=100, hv_tolerance=None, force_adjustable_off=False, settling_time=2.0, allow_retries = True):
		"""
		Sets the sensor layer high voltage bias.

		Repeat three times to ensure it gets sent through (due to problems previously had with the HV control)
		"""
		# Note the minimum output of the programable supply is ~100V if it is on, when off the ADC returns ~12V.
		# The adjustable supply can be turned off by either setting hv to the base_hv voltage or setting force_adjustable_off to True.

		self.check_config_loaded()

		if not hasattr(self, "hv_dac_slope"):
			if self.driver == "v5camera":
				_default = 0.8
			else:
				_default = 1.0
			self.hv_dac_slope = slope = self.config.get_attr(["software_config", "hv_corrections", self.hv_mac_addr, "dac_slope"], default=_default)
		else:
			slope = self.hv_dac_slope
		if not hasattr(self, "hv_dac_offset"):
			self.hv_dac_offset = offset = self.config.get_attr(["software_config", "hv_corrections", self.hv_mac_addr, "dac_offset"], default=0.0)
		else:
			offset = self.hv_dac_offset


		if self.driver == "testing" or self.camera_found == False:
			return
		if self.hv_enabled == False:
			self.set_attribute("hv", 0, allow_retries=allow_retries)
			return
		if self.driver == "v5camera":
			new_hv = float(hv - offset) / slope
			self.hv = hv
			if hv > 0:
				hv = new_hv
			self.set_attribute("hv", hv, allow_retries=allow_retries)

			return

		new_hv = float(hv - offset) / slope
		if hv > 0:
			hv = new_hv

		if hv_tolerance is None:
			hv_tolerance = self.config.get_attr(["software_config", "hv_tolerance"], default=5)
		if hv_tolerance < 1:
			hv_tolerance = 1
		if (hv != 1) and (hv > self.base_hv) and (hv - self.base_hv < 100) and not force_adjustable_off:
			self.marslog.print_log("marscamera.set_hv(): Error - the minimum output of the programmable hv supply is 100V", level="error", method="set_hv")
			self.marslog.print_log("WARNING: requested bias is " + str(hv) + " ... setting HV bias to " + str(self.base_hv + 100), level="error", method="set_hv")
			hv = self.base_hv + 100
			if hv_tolerance < 30:
				hv_tolerance = 30
		if force_adjustable_off:
			self.hv = int(self.base_hv)
		else:
			if self.hv != int(hv):
				self.hv = int(hv)
				self.hv_state = HV_WARMUP
		self.set_attribute("hv", 0, allow_retries=allow_retries)
		sleep(settling_time)  # allow output voltage to stabilise
		if self.hv > self.base_hv:
			self.set_attribute("hv", 1000, allow_retries=allow_retries)
			sleep(settling_time)  # allow output voltage to stabilise
		HV_RETRIES = 50
		bias_v = self.get_attribute("hv")
		if type(bias_v) in [type(0),type(0.0)]:
			current_hv = int(bias_v) + self.base_hv
		else:
			current_hv = self.base_hv
		set_hv = 100
		hv_set_count = 0

		difference = 0

		if hv <= self.base_hv:
			self.set_attribute("hv", 0, allow_retries=allow_retries)
			return

		while (abs(current_hv - self.hv) > hv_tolerance) and (hv_set_count < HV_RETRIES):
			if abs(current_hv - self.hv) < hv_step / 2.0 and hv_step > hv_tolerance:
				hv_step = int(hv_step / 2)
			if (abs(set_hv + self.base_hv - current_hv) > 100) or (current_hv < 1) or (current_hv > 2000):

				try:
					hv_monitored = int(self.get_attribute("hv"))
				except Exception as msg:
					hv_monitored = 0						# will force continued adjustment
					self.marslog.print_log("Camera: failed to read sensor bias voltage -> ", level="warning", method="set_hv")
				current_hv = hv_monitored + self.base_hv
				self.marslog.print_log("Camera: high_voltage -> ", str(current_hv), "(reread)", level="info", method="set_hv")
				if (abs(set_hv + self.base_hv - current_hv) > 100):
					if not force_adjustable_off and (self.hv > self.base_hv):
						current_hv = 100 + self.base_hv
						set_hv = 100
					else:
						current_hv = self.base_hv
						set_hv = 0
				elif (current_hv < 1) or (current_hv > 2000):
					if not force_adjustable_off and (self.hv > self.base_hv):
						current_hv = 100 + self.base_hv
						set_hv = 100
					else:
						current_hv = self.base_hv
						set_hv = 0

#			difference = min(abs(current_hv - self.base_hv - set_hv), 50)
#			set_hv = min(self.hv-self.base_hv + difference, set_hv + hv_step + difference)
			difference = int((difference + current_hv - (set_hv + self.base_hv)) / 2)  # Provide some damping on difference
			if set_hv == 0 and hv_tolerance < difference:
				hv_tolerance = abs(difference)
			self.marslog.print_log("Difference, set_hv, self.hv, current_hv", difference, set_hv, self.hv, current_hv, level="debug", method="set_hv")
			if abs(difference > 10.0):
				self.marslog.print_log("marscamera.set_hv(): Warning - the difference between the programmed and measured HV bias is greater than 10V.", level="info", method="set_hv")

			set_hv = min(self.hv - self.base_hv, current_hv - self.base_hv + hv_step) - difference
			if (set_hv < 0) or force_adjustable_off or (self.hv == self.base_hv):
				set_hv = 0
			hv_set_count += 1
			self.set_attribute("hv", set_hv * 1000, allow_retries=allow_retries)
			sleep(settling_time)  # allow output voltage to stabilise
			if "debug" in ml.global_display_level:
				 extra = "difference:" + str(difference)
			else:
				extra = ""
			self.marslog.print_log("Setting  high voltage on camera to", set_hv + self.base_hv, "(" + str(set_hv) + " + " + str(self.base_hv) + ")" + extra, level="info", method="set_hv")

			hv_monitored = self.get_attribute("hv")
			if type(hv_monitored) in [type(0),type(0.0)]:
				hv_monitored = int(hv_monitored)
			else:
				self.marslog.print_log("Camera: failed to read sensor bias voltage -> ", level="warning", method="set_hv")
				hv_monitored = 0
			current_hv = hv_monitored + self.base_hv
			self.marslog.print_log("Camera: high_voltage -> ", str(current_hv), level="info", method="set_hv")
			self.notify(key="hv")
		self.marslog.print_log("Camera: set high voltage to -> ", str(current_hv), level="info", method="set_hv")

			#if abs(set_hv + self.base_hv - current_hv > 50): current_hv = 100 +self.base_hv

	def save_frames(self, filename):
		"""
		Save the numpy frame array data to <filename>.
		"""
		numpy.save(filename, self._get_frames())

	def read_dacs(self):
		"""
		Read the on-pixel DACs using the SenseDAC function.
		Returns a dictionary of dacids and dacvals (millivolts).
		"""
		dacs = [{} for i in range(self.chipcount)]

		L = self.dacidxs  # [(id, idx)]
		for id in L.keys():
			for i in range(self.chipcount):
				dacs[i][id] = self.sense_dac(id, i)
		return dacs

	def load_mode_masks(self, mode):
		return

	@marsthread.modifier
	def _append_temperature(self, val):
		self._temperature_history.append(list(val)[:])  # by applying the list casting, we should create a copy
		while len(self._temperature_history) > MAX_HISTORY_COUNT:
			self._temperature_history.pop(0)

	@marsthread.modifier
	def _append_hv(self, val):
		self._hv_history.append(val)
		while len(self._hv_history) > MAX_HISTORY_COUNT:
			self._hv_history.pop(0)

	def get_asic_temperatures(self):
		temps = [0.0 for i in range(self.chipcount)]
		for chipid in range(self.chipcount):
			temps[chipid] = self.read_temperature(chipid=chipid) # read_temperature always return a float.
		return temps

	def run_polled_temperature_read(self):
		# camera lm35 temperature
		if self.config:
			has_lm35 = self.config.get_attr(['software_config', 'lm35_temperature', 'enabled'], default=True)
		else:
			has_lm35 = False
		chipcount = self.chipcount
		if has_lm35 == False:
			templen = chipcount
		elif has_lm35 == True:
			templen = chipcount + 1

		temps = [0.0 for i in range(templen)]

		if has_lm35:
			lm35_temp = self.read_lm35_temperature()
			temps[-1] = lm35_temp

		# ASIC bandgap temperatures
#		sleep(0.0001) # hold time to separate from previous acquires.
		temps = self.get_asic_temperatures()
		for chipid in range(chipcount):
			marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "Chip_"+str(chipid+1)+"_ASICTemperature", temps[chipid]) # temps[chipid] must always be a float.

		if has_lm35:
			self.read_temp = temps[-1]
		else:
			self.read_temp = numpy.array(temps).mean()

		marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "CameraTemperatureMonitor", self.read_temp)
		self._append_temperature((time() - self.start_time, temps))
		self.notify(Message(sender="get_temperatures", action="temperature", value=temps), key="temperature")



	def get_polling(self):
		"""
		Actually just the generic poll thread.
		"""
		temps = [0.0 for i in range(self.chipcount + 1)]
		last_time = 0.0
		last_time2 = 0.0
		last_time3 = 0.0
		self.check_config_loaded()
		while self.running:
			if time() > last_time + 20.0 and self.scanning is False and self.config != None and self.driver in ["libmarscamera", "testing", "v5camera"]:
				self.run_polled_temperature_read()
				last_time = time()

			if time() > last_time2 + 0.2:
				self.hv_timer_check()
				last_time2 = time()

			if time() > last_time3 + 10.0 and self.scanning is False and self.driver in ["testing", "libmarscamera"]:
				self.get_hv()
				self.notify(Message(sender="get_hv", action="hv"), key="hv")

				last_time3 = time()

			sleep(0.01)
		self.marslog.print_log("Exiting get_temperatures thread on marscamera", level="debug", method="get_polling")

	def read_temperature(self, chipid=None):
		return -1.0

	def read_lm35_temperature(self):
		self.check_config_loaded()
		try:   # if self.config is None here the software error should be logged!!
			slope = self.config.get_attr(("software_config", "lm35_temperature", "slope"), default=0.1)
			offset = self.config.get_attr(("software_config", "lm35_temperature", "offset"), default=0.0)
		except Exception as msg:
			self.marslog.print_log("marscamera.read_lm35_temperature: error reading lm35 temperature configuration data", msg, level="error", method="read_lm35_temperature")

		try:
			temp = self.get_attribute("temp")
			if type(temp) in [type(0.0), type(0)]:
				lm35_temp = float(temp * slope + offset)
				if (lm35_temp < 500.0) and (lm35_temp > -50.0):	# this test is to distinguish between numeric data that is a temperature, and the 'bogus' data returned at times bu V3 Readout boards.
					self.lm35_temp = lm35_temp
				else:
					self.marslog.print_log("marscamera.read_lm35_temperature: got invalid temperature: ", lm35_temp, "  ... skipping", level="warning", method="read_lm35_temperature")
			else:
				self.lm35_temp = float(0.0)
				self.marslog.print_log("marscamera.read_lm35_temperature: (invalid temperature, setting 0.0)", self.lm35_temp, level="debug", method="read_lm35_temperature")
		except Exception as msg:
			self.marslog.print_log("marscamera.read_lm35_temperature: error reading lm35 temperature", self.lm35_temp, msg, level="error", method="read_lm35_temperature")

		self.marslog.print_log("marscamera.read_lm35_temperature: ", self.lm35_temp, level="debug", method="read_lm35_temperature")
		return float(self.lm35_temp)

	@marsthread.modifier
	def _set_ktemp(self, i, val):
		self._k_temp[i] = val

	@marsthread.reader
	def _get_ktemp(self, i):
		return self._k_temp[i]

	@marsthread.modifier
	def _set_mtemp(self, i, val):
		self._m_temp[i] = val

	@marsthread.reader
	def _get_mtemp(self, i):
		return self._m_temp[i]

	def check_config_id(self, config):
		"""
		We have already validated the config by this point. But we may need to load a different one if the fuses don't match.

		If no id has been supplied for these fuses, then we will create one for the current camera id.
		"""
		return

	def update_config(self, config=None):
		"""
		Loads the config into the camera memory
		"""
		assert config is not None, "Invalid config supplied to marsCamera"
		assert issubclass(type(config), marsct.marsconfig.cameraConfig), "MarsCamera configs must be camera_configs." + str(type(config))
		self.check_config_id(config)

		self.config = config

		# exptime
		self.exptime = config.get_attr(["software_config", "default_exposure_ms"], default=self.exptime)

		##
		# We use the mode for dacvals and masks from now on
		#

		# masks
		#masks = numpy.zeros([self.chipcount, 256, 256], dtype='uint32')
		#for i in config.get_attr(["software_config", "masks"]).keys():
		#	i = int(i)
		#	if i < self.chipcount:
		#		masks[i] = config.get_attr(["software_config", "masks", i]).copy().reshape([256, 256])
		#
		#self.masks = masks
		#self.load_config()
		# dacs
		#dacvals = {}
		#for i in config.get_attr(["software_config", "dacs"]).keys():
		#	i = int(i)
		#	dacvals[i] = config.get_attr(["software_config", "dacs", i]).copy_to_dict()
		#self.dacvals = dacvals
		#self.load_dacs()

		# check fuses (and automatically fill up config as needed)
		fuseids = config.get_attr(["software_config","fuseid"], default=["0x00000"])
		for i in range(self.chipcount):
			if config.driver == "testing" or "camera_testing" in sys.argv:
				do_get_fuseid = False
			else:
				do_get_fuseid = False
				if i < len(fuseids):
					self.marslog.print_log("update_config: fuseid dict", fuseids, feature='camera_config', level='debug', method="update_config")
					try:
						self.marslog.print_log("update_config: read fuseid ", int(fuseids[i], 16), feature='camera_config', level='debug', method="update_config")
					except Exception as msg:
						pass
					try:
						if int(fuseids[i], 16) < 1:
							do_get_fuseid = True
					except Exception as msg:
						do_get_fuseid = True
				else:
					do_get_fuseid = True
			if "OVERWRITE_FUSEID" in sys.argv:
				do_get_fuseid = True
			if do_get_fuseid:
				self.marslog.print_log("update_config: do_get_fuseid is true." ,feature='camera_config', level='info', method="update_config")
				try:
					fusedata = self.get_chip_attribute("fuses", chipid=i)  # Assume readout 0.
				except Exception as msg:
					self.marslog.print_log("update_config: read fuses", msg, level="error", method="update_config")
					fusedata = 0

				if "fuseid" not in config.get_attr(["software_config"]).keys():
					config["software_config"]["fuseid"] = {}

				config.set_attr(["software_config","fuseid",i], str(hex(fusedata & 0xffffffff)))

				try:
					filename =  config.software_config.filename
					directory = config.software_config.directory
					filename =  os.path.join(directory, filename)
					config.software_config.write_config(filename)
					self.marslog.print_log("update_config: config filename", filename ,feature='camera_config', level='info', method="update_config")
				except Exception as msg:
					self.marslog.print_log("marscamera.update_config: error checking fuseid" ,i ,msg ,level='error', method="update_config")

				self.marslog.print_log("update_config: write fuseid  - " ,config.software_config.fuseid[i], " chip", int(i), feature='camera_config', level='info', method="update_config")

				# if fuse id is newly updated, check module mac_address.
				try:
					ip_addrlist  = [ config.software_config.network["ip_addr"] ]
				except Exception as msg:
					ip_addrlist  = ["192.168.0.44"]
				mac_table = get_mac_address(ip_addresses=ip_addrlist)
				for ip_addr in mac_table.keys():

					if i == 0: # updates when chip 0 in the software_config is set.
						config.network["ip_addr"] = ip_addr
						config.network["mac_addr"] = mac_table[ip_addr]
						try:
							cfg_filename =  config.filename
							cfg_directory = config.directory
							cfg_filename =  os.path.join(cfg_directory,cfg_filename)
							config.write_config(cfg_filename)
						except Exception as msg:
							self.marslog.print_log("marscamera.update_config: error updating camera mac_address", msg ,level='error', method="update_config")

					do_sw_cfg_update=False
					if not config.software_config.network.has_key(i):
						config.software_config.network[i]={}
						config.software_config.network[i]["ip_addr"] = ip_addr
						do_sw_cfg_update=True
					if ip_addr == config.software_config.network[i]["ip_addr"]:
						do_sw_cfg_update=True
						config.software_config.network[i]["mac_addr"] = mac_table[ip_addr]
					if do_sw_cfg_update:
						try:
							config.software_config.write_config(filename)
						except Exception as msg:
							self.marslog.print_log("marscamera.update_config: error updating camera mac_address" ,i ,msg ,level='error', method="update_config")

		self.map_chipid()
		self.update_dicom_mechanical_tags()

		# polarity
		self.set_polarity(config.get_attr(["software_config", "polarity"], default=self.polarity))

		##
		# Also use mode for energy calibration

		#self.energy_calibration = config.get_attr(["software_config", "energy_calibration"]).copy_to_dict()
		for i in range(self.chipcount):
			_i = self._get_chipid(i)
			self._set_mtemp(i, config.get_attr(["software_config", "temperatures", _i, "slope"]))
			self._set_ktemp(i, config.get_attr(["software_config", "temperatures", _i, "offset"]))
		# hv
		try:
			self.base_hv = int(config.get_attr(["software_config", "hv_base_v"], default=self.base_hv))
		except Exception as msg:
			self.base_hv = 0
			self.marslog.print_log("Camera: could not load base HV from camera software config", msg, level="error", method="update_config")

		if self.driver == "v5camera":
			_default = 1.57
		else:
			_default = 1.0
		self.hv_adc_slope = config.get_attr(["software_config", "hv_corrections", self.hv_mac_addr, "adc_slope"], default=_default)
		self.hv_adc_offset = config.get_attr(["software_config", "hv_corrections", self.hv_mac_addr, "adc_offset"], default=0.0)
		if self.driver == "v5camera":
			_default = 0.8
		else:
			_default = 1.0
		self.hv_dac_slope = config.get_attr(["software_config", "hv_corrections", self.hv_mac_addr, "dac_slope"], default=_default)
		self.hv_dac_offset = config.get_attr(["software_config", "hv_corrections", self.hv_mac_addr, "dac_offset"], default=0.0)

		self.hv_warmup_time = int(config.get_attr(["software_config", "hv_warmup_time"], default=self.hv_warmup_time))
		self.hv_idle_timeout = int(config.get_attr(["software_config", "hv_idle_timeout"], default=self.hv_idle_timeout))

		try:
			self.set_hv(config.get_attr(["software_config", "hv_v"], default=self.hv))
		except Exception as msg:
			self.marslog.print_log("Failure to set the bias voltage during config read", msg, level="error", method="update_config")
		gobject.timeout_add(5000, self.update_get_hv)
		self.hv_refresh()

		# modes
		#try:
		modes = config.get_attr(["software_config", "modes"])
		for mode in modes.keys():
			self._set_mode_map(modes[mode]["name"], mode)  # usage: modedata=self.modes[self.mode_map[text_name]]
			self._set_mode(mode, modes[mode])
		#except Exception as msg:
		#	self.marslog.print_log("Exception setting the modes from config", msg, level="error")
		#else:
		self.set_mode(config.get_attr(["software_config", "default_mode"], default="colour_csm_full"))

	# for chip specific commands

	# First do methods that call libmars directly
	def set_dac(self, dac_name, dac_value, chipid=None, ignore_lock=False):
		"""
		Sets just one DAC in the marsCamera
		"""
		if self.has_dac_lock and ignore_lock is False:
			self.marslog.print_log("Set dac called while dac lock active. Aborting.", level="error", method="set_dac")
			import traceback
			traceback.print_stack()
			return
		if chipid is None:
			chipid = self.chip

		if chipid == "all":
			for i in range(self.chipcount):
				self.set_dac(dac_name, dac_value, chipid=i, ignore_lock=ignore_lock)
			return
		dac_name = self.check_dac_name(dac_name)
		if dac_name in self.dacvals[chipid].keys():
			self._set_dac(dac_name, dac_value, chipid=chipid, ignore_lock=ignore_lock)
			dac_value = self._get_dac(dac_name, chipid=chipid)
			try:
				marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "MP3_" + str(chipid+1) + "_" + dac_name, dac_value)
			except Exception as msg:
				self.marslog.print_log("Unable to set private tag for dac", dac_name, msg, level="warning", method="set_dac")
		if verbose:
			self.marslog.print_log("Setting DAC", dac_name, level="info", method="set_dac")
		if self.driver == "libmarscamera":
			self.libmars_caller(libmarscamera_py.setDac, self.devid, chipid, dac_name, int(dac_value))
		elif self.driver == "v5camera":
			self.camera_driver[chipid].write_DACs({dac_name:int(dac_value)})
		self.notify(key="dac")

	def get_chip_attribute(self, key, chipid=None):
		"""
		Return the value of the camera-chip <key> attribute.
		"""
		if chipid is None:
			chipid = self.chip
		if key not in self.pseudo_attributes:
			if self.driver == "libmarscamera":
				return self.libmars_caller(libmarscamera_py.getChipAttribute, self.devid, chipid, key)
			elif self.driver == "v5camera":
				if key in self.camera_driver[chipid].OMR:
					return self.camera_driver[chipid].OMR[key]
				elif key in ["adc"]:
					return self.camera_driver[chipid].read_ADC()
				elif key.lower() == "fuses":
					return self.camera_driver[chipid].get_id()
				elif key.lower() == "type":
					return 3
				else:
					return 0
			elif self.driver == "testing":
				if key == "type":
					return 3
				if key == "adc":
					return 400.0
				if key == "fuses":
					return chipid
				else:
					return 0
		else:
			return getattr(self, key)

	def get_counter_type(self, counter=0 ):
		medipix_counter = self.parse_csm_counter(counter)
		frame_index = self.frame_order.index(medipix_counter)
		frame_type = self.frame_type[frame_index]
		return frame_type

	def parse_csm_counter(self, counter):
		"""
		Converts a virtual counter index to the index the Medipix ASIC uses.
		This method should be overriden by any camera subclass that alters the counter order.
		"""
		return counter


	def set_chip_attribute(self, key, val, chipid=None):
		"""
		Set the <val> of the camera-chip <key> attribute.
		"""
		if chipid is None:
			chipid = self.chip
		if chipid == "all":
			for i in range(self.chipcount):
				self.set_chip_attribute(key, val, i)
			return
		try:
			if self.driver == "libmarscamera":
				self.libmars_caller(libmarscamera_py.setChipAttribute, self.devid, chipid, key, val)
			elif self.driver == "v5camera":
				self.camera_driver[chipid].write_OMR({key:val})
		except Exception as msg:
			self.marslog.print_log("marscamera.set_chip_attribute:%s" % (str(msg)), level="error", method="set_chip_attribute")
			raise
		else:
			self.notify(key="attribute")

	# Then do methods that don't call libmars directly

	def _sense_dac(self, dacidx, chipid=None):
		"""
		Filler method to be overwritten by relevant classes
		"""
		pass

	@marsthread.takesLock
	def sense_dac(self, dacname, chipid=None):
		"""
		Gets the ADC value of a dac by its dacname
		"""
		for k in self.dacidxs.keys():
			if k.lower() == dacname.lower():
				return self._sense_dac(self.dacidxs[k], chipid)
		else:
			raise KeyError(dacname)

	def set_threshold(self, threshold, chipid=None):
		"""
		Filler method to be overwritten by relevant classes
		"""
		pass

	def get_threshold(self, chipid=None):
		"""
		Filler method to be overwritten by relevant classes.
		"""
		pass

	def set_energy(self, energy, chipid=None, counter=0):
		"""
		Filler method to be overwritten by relevant classes
		"""
		pass

	def get_energy(self, chipid=None, counter=0):
		"""
		Filler method to be overwritten by relevant classes
		"""
		pass


@marsthread.hasRWLock
@marsthread.hasLock
class marsCameraMP3(marsCamera):

	"""
	Represents the MARS/Medipix3 camera.
	Sub-classed from the marsCamera (base) class.
	"""
	# THA<0-4> bit values for first threshold.
	THA0 = 1 << 17
	THA1 = 1 << 5
	THA2 = 1 << 6
	THA3 = 1 << 19
	THA4 = 1 << 7

	# THB<0-4> bit values for second threshold.
	THB0 = 1 << 18
	THB1 = 1 << 22
	THB2 = 1 << 20
	THB3 = 1 << 16
	THB4 = 1 << 4

	# High-gain/low-gain bit. (Set to 0 for high-gain.)
	GAIN = 1 << 8

	# Pixel logic inhibit mask bit. (Set to 1 to inhibit the pixel.)
	INHIBIT = 1 << 21

	@marsthread.modifier
	@marsthread.takesLock
	def __init__(self, masks=None, dacvals=None, is_csm=False, is_eq=False, is_cm=False):

		"""
		Instantiates a MP3 Medipix camera object and sets the configuration mask
		and DAC-values if supplied.
		"""
		self.marslog = ml.logger(source_module="marscamera", group_module="camera", class_name="marsCameraMP3", object_id=self)
		self._config = None
		self._devid = 0
		self._camera_found = False
		if "camera_testing" in sys.argv:
			self.driver = "testing"
		else:
			self.driver = marsguiref.CONFIG["camera"].get_attr(["driver"], default="testing")
		self.chipcount = self.get_attribute("chips")
		if self.chipcount is None:
			self.chipcount = 1
		self.attributes = {}
		self.chip_attributes = [{} for i in range(self.chipcount)]
		self._counters = 2
		self._set_counters = 2
		self._csm_counters = 0
		self._cm = False
		self._csm = False
		self._eq = False
		self._extdac = 0
		self._extdacvalue = 0
		self._cas_adc = [[] for chipid in range(self.chipcount)]

		marsCamera.__init__(self, masks=masks, dacvals=dacvals)

		self.marslog = ml.logger(source_module="marscamera", group_module="camera", class_name="marsCameraMP3", object_id=self)

		self.chipcount = self.get_attribute("chips")
		if self.chipcount is None:
			self.chipcount = 1

		while len(self.chip_attributes) < self.chipcount:
			self.chip_attributes.append({})
		while len(self._cas_adc) < self.chipcount:
			self._cas_adc.append([])

		self.adjust_bits = [[self.THA0, self.THA1, self.THA2, self.THA3, self.THA4],
                      [self.THB0, self.THB1, self.THB2, self.THB3, self.THB4]]
		self.inhibit_bit = self.INHIBIT
		self.gain_bit = self.GAIN
		self.active_high = True

		# marsthread.new_thread(self.check_cas_adc)

		# if is_csm != None:
		self.set_csm(is_csm)
		self.set_equalizeTHH(is_eq)
		self.set_colourMode(is_cm)

	def load_default_dacvals(self):
		dacvals = {}
		for i in range(self.chipcount):
			dacvals[i] = MP3_VALUES.copy()
		self.set_dacvals(dacvals)

	def check_cas_adc(self):
		if self.medipix_type == MP3_TYPE:
			cas_str = "Cas"
		elif self.medipix_type == RX3_TYPE:
			cas_str = "V_Cas"
		else:
			return
		while self.running:
			for chipid in range(self.chipcount):
				tmp_val = self.sense_dac(cas_str, chipid=chipid)
				self.cas_adc[chipid].append(tmp_val)
				if max(self.cas_adc[chipid]) > min(self.cas_adc[chipid]) * 1.1:
					self.marslog.print_log("Warning: Cas ADC output on chip %d varied by over 10\% with values %d and %d" % (chipid, max(self.cas_adc[chipid]), min(self.cas_adc[chipid])), level="warning", method="check_cas_adc")  # Its a \%, not a format character. pylint: disable=E
			sleep(5)

	# Member interaction methods
	@property
	@marsthread.reader
	def dacidxs(self):
		return MP3_SCAN

	@property
	@marsthread.reader
	def dacranges(self):
		return MP3_RANGE

	@marsthread.reader
	def _get_frame(self, chipid):
		return self._frames[chipid]

	def get_frame(self, chipid, counter=0, frame=None):
		if frame is None:
			frame = self._get_frame(chipid)
		if not self.colourmode:
			if counter == 0 and self.equalizethh == False or counter == 1 and self.equalizethh:
				return frame & 0x000FFF
			elif counter == 1:
				return (frame & 0xFFF000) >> 12
			else:
				return frame
		else:
			if counter in [0, 1, 4, 5]:
				i = 0
			else:
				i = 1
			if counter in [0, 1, 2, 3]:
				j = 1
			else:
				j = 0
			if (counter in [0, 2, 4, 6] and self.equalizethh == False) or (counter == [1, 3, 5, 7] and self.equalizethh):
				_tempFrames = frame & 0x000FFF
				return _tempFrames[i::2, j::2]
			elif counter in [1, 3, 5, 7]:
				_tempFrames = (frame & 0xFFF000) >> 12
				return _tempFrames[i::2, j::2]
			else:
				return frame

	@marsthread.reader
	def _get_frames(self, counter="both"):
		if counter == 0 and self.equalizethh == False or counter == 1 and self.equalizethh:
			return self._frames & 0x000FFF
		elif counter == 1:
			return (self._frames & 0xFFF000) >> 12
		else:
			return self._frames

	# Full camera methods (applied over all chips or chip-independant).

	def load_dacs(self):
		old_cas = [self._get_dac("Cas", chipid=chipid) for chipid in range(self.chipcount)]
		#old_fbk = [self._get_dac("FBK", chipid=chipid) for chipid in range(self.chipcount)]
		#old_gnd = [self._get_dac("GND", chipid=chipid) for chipid in range(self.chipcount)]
		#self.set_dac("Cas", 0, chipid="all")
		#self._set_dac("FBK", 0, chipid="all")
		#self._set_dac("GND", 0, chipid="all")
		marsCamera.load_dacs(self)
		for chipid in range(self.chipcount):
			self.set_dac("Cas", old_cas[chipid], chipid=chipid)
			#self._set_dac("FBK", old_fbk[chipid], chipid=chipid)
			#self._set_dac("GND", old_gnd[chipid], chipid=chipid)
		# marsCamera.load_dacs(self)

	# first do methods that call libmars directly
	@marsthread.modifier
	def _set_attribute(self, key, val):
		self.attributes[key] = val

	@marsthread.reader
	def _get_attribute(self, key):
		return self.attributes[key]

	def set_attribute(self, key, val, allow_retries=True):
		marsCamera.set_attribute(self, key, val, allow_retries=allow_retries)
		self._set_attribute(key, val)

	# And then do methods that don't call libmars directly

	def set_csm(self, is_csm=True):
		"""
		Set Charge Summing Mode (EnablePixelCom).
		"""
		self.csm = is_csm
		self.set_chip_attribute("EnablePixelCom", is_csm, "all")

	def set_equalizeTHH(self, is_eq=True):
		"""
		Set EqualizeTHH (puts Threshold1 results into Counter0 for equalisation purposes)
		"""
		self.equalizethh = is_eq
		self.set_chip_attribute("EqualizeTHH", is_eq, "all")

	def set_colourMode(self, is_cm=True):
		"""
		Set colourMode

		The scan scripts use self.pixelsize (determines pixel size of pseudo-stitched frames), the GUI uses self.counters
		NOTE:

		* ColourMode can be set by calling set_chip_attribute without calling set_ColourMode
		* the system may also have cameras with some chips operating in spm and some in colourmode
			so pixelsize and counters are chip level attributes [TODO]

		TODO: we should allow setting colourmode, pixelsize and counters for selected chips.
		"""
		self.colourmode = is_cm

		if self.colourmode:
			self.pixelsize = 0.11
			self.set_chip_attribute("pixelsize", 0.11, "all")
			self.set_chip_attribute("counters", 8, "all")
		else:
			self.pixelsize = 0.055
			self.set_chip_attribute("pixelsize", 0.055, "all")
			self.set_chip_attribute("counters", 2, "all")
		self.set_chip_attribute("ColourMode", is_cm, "all")

	def save_frames(self, filename, counter="both"):
		"""
		Saves the frames to file.
		"""
		numpy.save(filename, self._get_frames(counter))

	def update_config(self, config=None):
		"""
		Updates from the configuration file.
		"""
		if config.get_attr(["software_config", "type"]) != MP3_TYPE:
			raise LookupError("Config is for incorrect Medipix type")
		marsCamera.update_config(self, config)
		for i in range(self.chipcount):
			for name in config.get_attr(["software_config", "omr"]).keys():
				self.set_chip_attribute(name, config.get_attr(["software_config", "omr", name]), i)

	# Chip-specific methods

	# first do methods that call libmars directly
	@marsthread.modifier
	def _set_chip_attribute(self, key, val, chipid=None):
		if chipid is None:
			chipid = self.chip
		self.chip_attributes[chipid][key] = val

	@marsthread.reader
	def _get_chip_attribute(self, key, chipid=None):
		if chipid is None:
			chipid = self.chip
		return self.chip_attributes[chipid][key]

	def set_chip_attribute(self, key, val, chipid=None):
		"""
		Set the <val> of the camera-chip <key> attribute.
		"""
		if chipid is None:
			chipid = self.chip
		if chipid == "all":
			for i in range(self.chipcount):
				self.set_chip_attribute(key, val, i)
			return
		if key.lower() == "enablepixelcom":
			key = "EnablePixelCom"
			self.csm = val
		if key.lower() == "equalizethh":
			key = "EqualizeTHH"
			self.equalizethh = val
		if key.lower() == "colourmode":
			key = "ColorMode"
			self.colourmode = val
			if val:
				self.counters = 8
				self.pixelsize = 0.11
				if not self.csm:
					self.set_counters = 8
					self.csm_counters = 0
					self.frame_order=[0, 1, 2, 3, 4, 5, 6, 7]
					self.frame_type=["SPM1","SPM2","SPM3","SPM4","SPM5","SPM6","SPM7","SPM8"]
				else:
					self.set_counters = 7
					self.csm_counters = 4
					self.frame_order=[0, 1, 3, 5, 7, 2, 4, 6]
					self.frame_type=["ARB","CSM1","CSM2","CSM3","CSM4","SPM1","SPM2","SPM3"]

				self.set_chip_attribute("counters", 8, chipid)
				self.set_chip_attribute("pixelsize", 0.11, chipid)
			else:
				self.counters = 2
				self.set_counters = 2
				self.csm_counters = 0
				self.frame_order=[0, 1]
				if not self.csm:
					self.frame_type=["SPM1","SPM2"]
				else:
					self.frame_type=["ARB","CSM1"]

				self.pixelsize = 0.055
				self.set_chip_attribute("counters", 2, chipid)
				self.set_chip_attribute("pixelsize", 0.055, chipid)
		if key.lower() == "extdac":
			key = "ExtDAC"
			self.extdac = val

		try:			# only do this for real chip attributes (i.e. don't send pixelsize)
			if key not in ["counters", "pixelsize"]:  # TODO: change this to a test for valid chipAttribute keys..
				if self.driver == "libmarscamera":
					self.libmars_caller(libmarscamera_py.setChipAttribute, self.devid, chipid, key, val)
				elif self.driver == "v5camera":
					self.camera_driver[chipid].write_OMR({key:val})
		except Exception as msg:
			self.marslog.print_log("marscamera.set_chip_attribute:: Error: %s" % (str(msg)), level="error", method="set_chip_attribute")
#			raise
		else:
			self._set_chip_attribute(key, val, chipid=chipid)
			self.notify(key="attribute")

	def set_dac(self, dac_name, dac_value, chipid=None, ignore_lock=False):
		"""
		Sets just one DAC in the marsCamera
		"""
		if self.has_dac_lock and ignore_lock is False:
			self.marslog.print_log("Set dac called while dac lock active. Aborting.", level="error", method="set_dac")
			import traceback
			traceback.print_stack()
			return
		if chipid is None:
			chipid = self.chip
		if chipid == "all":
			for i in range(self.chipcount):
				self.set_dac(dac_name, dac_value, chipid=i, ignore_lock=ignore_lock)
			return
		dac_name = self.check_dac_name(dac_name)
		if dac_name in self.dacvals[chipid].keys():
			self._set_dac(dac_name, dac_value, chipid=chipid, ignore_lock=ignore_lock)
			dac_value = self._get_dac(dac_name, chipid=chipid)
			try:
				marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "MP3_" + str(chipid+1) + "_" + dac_name, dac_value)
			except Exception as msg:
				self.marslog.print_log("Unable to set private tag for dac", dac_name, msg, level="warning", method="set_dac")

		if verbose:
			self.marslog.print_log("Setting DAC", dac_name, level="info", method="set_dac")
		if dac_name == "Cas":
			self.set_chip_attribute("dac", dac_value, chipid)
			self.cas_adc = [[] for chipid in range(self.chipcount)]
		else:
			if self.driver == "libmarscamera":
				self.libmars_caller(libmarscamera_py.setDac, self.devid, chipid, dac_name, int(dac_value))
			elif self.driver == "v5camera":
				self.camera_driver[chipid].write_DACs({dac_name:int(dac_value)})

		self.notify(key="dac")

	# And then do methods that don't call libmars directly
	def set_threshold(self, threshold, counter=0, chipid=None):
		"""
		Sets the threshold of the chip independant of the dacname
		"""
		self.set_dac(("Threshold%d" % (counter)), int(threshold), chipid)

	def get_threshold(self, counter=0, chipid=None):
		"""
		Gets the threshold of the chip independant of the dacname
		"""
		return self._get_dac(("Threshold%d" % (counter)), chipid)

	def _sense_dac(self, dacid=0, chipid=None):
		"""
		Read DAC <dacid> using SenseDAC OMR function.
		NB MP3 has 25 on pixel DACs (0 - 24).
		"""
		self.set_chip_attribute('SenseDAC', dacid, chipid)
		sleep(10e-3)
		return self.get_chip_attribute('adc', chipid=chipid)

	def set_energy(self, energy, counter=0, chipid=None):
		"""
		Sets the threshold of the chip to an energy.
		"""
		if chipid is None:
			chipid = self.chip
		elif chipid == "all":
			for i in range(self.chipcount):
				self.set_energy(energy, counter, i)
			return

		energy_calibration = self.energy_calibration

		if chipid < len(energy_calibration) and counter < len(energy_calibration[chipid]):
			slope = energy_calibration[chipid][counter]['slope']
			offset = energy_calibration[chipid][counter]['offset']
			threshold = int(round(energy * slope + offset))
		else:
			self.marslog.print_log("Warning there is no energy calibration data for this configuration:   chip->", chipid, "  counter->", counter, level="warning", method="set_energy")
			threshold = 511
#			slope = energy_calibration[chipid]['slope']
#			offset = energy_calibration[chipid]['offset']
		self.set_threshold(threshold, counter, chipid)

	def get_energy(self, counter=0, chipid=None):
		"""
		Get the energy value of the current chip threshold
		"""
		if chipid is None:
			chipid = self.chip

		energy_calibration = self.energy_calibration

		if chipid < len(energy_calibration) and counter < len(energy_calibration[chipid]):
			slope = energy_calibration[chipid][counter]['slope']
			offset = energy_calibration[chipid][counter]['offset']
			threshold = self.get_threshold(counter, chipid)
			try:
				energy = float(threshold - offset) / float(slope)
			except Exception as msg:
				self.marslog.print_log("Energy calibration error: (zero slope?)  chip,counter,slope,offset->", chipid, counter, slope, offset, msg, level="error", method="get_energy")
				energy = 0.0
		else:
			self.marslog.print_log("Warning there is no energy calibration data for this configuration:   chip->", chipid, "  counter->", counter, level="warning", method="get_energy")
			energy = 0.0

		return float(energy)

	def read_temperature(self, chipid=None):
		"""
		Read band_gap_output and band_gap_temperature monitor signals.
		Calculate the temperature using k_temp and m_temp instance variables.
		Return the result in degrees C.
		"""
		if chipid is None:
			chipid = self.chip
		if self.camera_found == False:
			return -1.0
		if self.driver == "v5camera":
			p, q = self.camera_driver[chipid].get_sense_dac_temperatures()
		else:
			p = self.sense_dac('band_gap_output', chipid)
			q = self.sense_dac('band_gap_temperature', chipid)

		try:
			temp = self._get_ktemp(chipid) + self._get_mtemp(chipid) * (q - p)
		except Exception as msg:
			self.marslog.print_log("Failed to calculate temperature for Chipid ", chipid, msg, level="debug", method="read_temperature")
			temp = 0.0
		#self.marslog.print_log("Chipid ", chipid, "'s temperature is therefore ", temp, level="debug")
		return temp

	# def build_frame(self, counter=0, frames = None):
		#"""
		# Stitch images from different chips
		#"""
		# if frames == None:
			#frames = self._get_frames()

		#posxChips = numpy.array(self.config.mechanical_config.chipx_pixels)
		#posyChips = numpy.array(self.config.mechanical_config.chipy_pixels)

		# if self.colourmode:
			#posxChips = posxChips/2
			#posyChips = posyChips/2
		#imagesize = numpy.array(self.get_frame(0, counter, frames[0]) ).squeeze().shape;
		#ret_img = numpy.zeros([imagesize[1] + max(posyChips), imagesize[0] + max(posxChips)])
		# for chipid in range(self.chipcount):
			#chipRot = self.config.mechanical_config.chipr_deg[chipid]
			#rotation = lambda angle: int((angle +45 )%  360)/90
			# if rotation(chipRot) <> 0:
				#ret_img[ posyChips[chipid]:posyChips[chipid]+imagesize[0], posxChips[chipid]:posxChips[chipid]+imagesize[1] ] = numpy.array( [numpy.rot90(self.get_frame(chipid, counter, frames[chipid]),rotation(chipRot))] ).squeeze()
			# else:
				#ret_img[ posyChips[chipid]:posyChips[chipid]+imagesize[0], posxChips[chipid]:posxChips[chipid]+imagesize[1] ] = numpy.array( [self.get_frame(chipid, counter, frames[chipid])] ).squeeze()
		# return ret_img

	@marsthread.reader
	def reset_and_rebuild(self):
		self.reset()
		self.load_dacs()
		self.load_config()
		for key in self.attributes.keys():
			self.set_attribute(key, self.attributes[key])
		for chipid in range(self.chipcount):
			for key in self.chip_attributes[chipid].keys():
				self.set_chip_attribute(key, self.chip_attributes[chipid][key])

	def finalise(self):
		"""
		The system is shutting down. Do what needs to be done prior to close.
		Watch mutexes - don't put a lock on this method. It could break things.
		"""
		self.set_hv(0)
		self.running = False
		if self.driver == "v5camera":
			try:
				if self.slave_timeout_count > 0:
					self.marslog.print_log("Slave timeout count is:", self.slave_timeout_count, level="error", method="finalise")
				for cd in self.camera_driver:
					cd.disconnect()
				self.hv_driver.disconnect()
				self.listener.finalise()
			except Exception as msg:
				self.marslog.print_log("Disconnect on V5 camera failed " + str(msg), level="info", method="finalise")

		self.poll_thread.join()

class marsCameraMXR(marsCamera):

	"""
	Represents the MARS/Medipix2-MXR camera.
	Sub-classed from the marsCamera (abstract) class.
	"""
	# THA<0-4> bit values for first threshold.
	THL0 = 1 << 7
	THL1 = 1 << 6
	THL2 = 1 << 8

	# THB<0-4> bit values for second threshold.
	THH0 = 1 << 12
	THH1 = 1 << 10
	THH2 = 1 << 11

	# Pixel logic inhibit mask bit. (Set to 1 to inhibit the pixel.)
	INHIBIT = 1 << 0

	def __init__(self, masks=None, dacvals=None):
		"""
		Instantiates a Medipix-MXR camera object and sets the configuration mask
		and DAC-values if supplied.
		"""
		marsCamera.__init__(self, masks=masks, dacvals=dacvals)

		self.marslog = ml.logger(source_module="marscamera", group_module="camera", class_name="marsCameraMXR", object_id=self)
		# MXR specific settings
		self.adjust_bits = [[self.THL0, self.THL1, self.THL2]]
		self.inhibit_bit = self.INHIBIT
		self.active_high = False

		self.set_attribute("polarity", 1)  # This assumed sensor type (CdTe or GaAs) by default!
		self.set_attribute("hv", 1)
		self.perm_mask = {}
		self.perm_THL_adj = {}
		self.perm_THH_adj = {}
		self.perm_test = {}
		self.maxcount = 11810
		self._counters = 1
		self._set_counters = 1
		self._csm_counters = 0
		for i in range(2):  # flush noisey frames
			self.acquire()

	def set_polarity(self, polarity=1):
		"""
		Set ASIC input polarity (default=1 set for Si, 0 for CdTe and GaAs).
		"""
		self.polarity = polarity
		self.marslog.print_log("set polarity:", polarity, level="debug", method="set_polarity")
		self.set_attribute("polarity", polarity)
		self.notify(key="polarity")

	@property
	@marsthread.reader
	def dacidxs(self):
		return MXR_SCAN

	@property
	@marsthread.reader
	def dacranges(self):
		return MXR_RANGE

	@marsthread.modifier
	def load_default_dacvals(self):
		dacvals = {}
		for i in range(self.chipcount):
			dacvals[i] = MXR_VALUES.copy()
		self.set_dacvals(dacvals)

	# Member interaction methods
	@marsthread.reader
	def _get_dacidxs(self):
		return MXR_SCAN

	@marsthread.reader
	def _get_dacranges(self):
		return MXR_RANGE

	# Methods that apply to the whole camera (or are chip independant)

	# First we do methods that call libmars directly
	@marsthread.modifier
	def reset(self):
		"""
		Reset the chip. Set the MXR configuration matrix to 'ones'.
		Returns None.
		"""
		try:
			masks = numpy.ones([256, 256, self.chipcount], dtype='uint32')
			if self.driver == "libmarscamera":
				self.libmars_caller(libmarscamera_py.resetChip, self.devid, self.chip)
			self.set_masks(masks)
		except Exception as msg:
			self.marslog.print_log("marsCameraMXR.reset:%s" % (str(msg)), level="error", method="reset")
			raise
		self.notify(key="reset")

	# And then we do methods that don't call libmars directly

	def update_config(self, config=None):
		"""
		Updates from the configuration file.
		"""
		if config.get_attr(["software_config", "type"]) != MXR_TYPE:
			raise LookupError("Config is for incorrect Medipix type")
		marsCamera.update_config(self, config)

	# Methods that are chip specific

	# First we do methonds that call libmars directly

	# And then we do methods that don't call libmars directly
	def set_threshold(self, threshold, counter=0, chipid=None):
		"""
		Sets the threshold on the MXR chip.
		"""
		self.set_dac("THLFine", int(threshold), chipid)

	def get_threshold(self, counter=0, chipid=None):
		"""
		Gets the threshold of the MXR chip using the generic interface
		"""
		return self._get_dac("THLFine", chipid)

	def set_energy(self, energy, counter=0, chipid=None):
		"""
		Sets the threshold of the chip to an energy.
		"""
		if chipid is None:
			chipid = self.chip
		if chipid == "all":
			for i in range(self.chipcount):
				self.set_energy(energy, chipid=i, counter=counter)
		else:
			energy_calibration = self.energy_calibration
			try:
				slope = energy_calibration[chipid][counter]['slope']
				offset = energy_calibration[chipid][counter]['offset']
			except Exception as msg:
				self.marslog.print_log("exception calculating threshold for chip", chipid, counter, msg, level="error", method="set_energy")
				self.marslog.print_log(energy_calibration[chipid], level="error", method="set_energy")
				slope = 1.2  # get from software_settings
				offset = 14.0
			threshold = int(round(float(energy) * slope + offset))
			self.set_threshold(threshold, chipid=chipid)
			self.marslog.print_log("Set energy  ", energy, "keV  ", "THL:", threshold, level="debug", method="set_energy")

	def get_energy(self, chipid=None, counter=0):
		"""
		Get the energy value of the current chip threshold
		"""
		if chipid is None:
			chipid = self.chip

		energy_cal = self.energy_calibration

		try:
			slope = energy_cal[chipid][counter]['slope']
			offset = energy_cal[chipid][counter]['offset']
			threshold = self.get_threshold(chipid)
			return float(threshold - offset) / float(slope)
		except Exception as msg:
			self.marslog.print_log("Exception getting energy on", chipid, counter, msg, level="error", method="get_energy")
			return 0.0

	def _sense_dac(self, dacid=0, chipid=None):
		"""
		Read DAC <dacid> using SenseDAC DAC
		"""
		self.set_dac('SenseDAC', 1, chipid)
		self.set_dac('DacCode', dacid, chipid)
		return self.get_chip_attribute('adc', chipid=chipid)

	@marsthread.modifier
	def parse_config(self, mask=[], THL_adj=[], THH_adj=[], test=[]):
		"""
		Places bits in the right place for MXR pixel configurations.
		"""
		# define the size of the config matrix
		self.acquire(0.01)
		chip_size = numpy.array(self._frames).shape
		buf_size = numpy.uint32(chip_size[0] * chip_size[1])  # this is the number of pixels

		# Parse agruments
		if mask == 'reset':
			reset = True
		else:
			reset = False

		if len(mask) == 0 and self._chipid in self.perm_mask.keys():
			mask = self.perm_mask[self._chipid]
		if len(THL_adj) == 0 and self._chipid in self.perm_THL_adj.keys():
			THL_adj = self.perm_THL_adj[self._chipid]
		if len(THH_adj) == 0 and self._chipid in self.perm_THH_adj.keys():
			THH_adj = self.perm_THH_adj[self._chipid]
		if len(test) == 0 and self._chipid in self.perm_test.keys():
			test = self.perm_test[self._chipid]

		# set defaults if needed
		if len(mask) == 0 or reset == True:
			mask = numpy.ones(chip_size)  # default is all pixels on
		if len(THL_adj) == 0 or reset == True:
			THL_adj = numpy.zeros(chip_size)  # default is all zero
		if len(THH_adj) == 0 or reset == True:
			THH_adj = numpy.zeros(chip_size)  # default is all zero
		if len(test) == 0 or reset == True:
			test = numpy.zeros(chip_size)  # default is all zero

		self.perm_mask[self._chipid] = mask
		self.perm_THL_adj[self._chipid] = THL_adj
		self.perm_THH_adj[self._chipid] = THH_adj
		self.perm_test[self._chipid] = test

		# these are the values for each bit in the full config matrix.
		mxrPixelOn = 1  # 1 << 0
		mxrTest = 512  # 1 << 9
		mxrLow0 = 128  # 1 << 7
		mxrLow1 = 64  # 1 << 6
		mxrLow2 = 256  # 1 << 8
		mxrHigh0 = 4096  # 1 << 12
		mxrHigh1 = 1024  # 1 << 10
		mxrHigh2 = 2048  # 1 << 11
		# see page 24/25 of MXR manual describing the pixel config register
		# I could have used bitset here, but this works ok.

		# Assemble the config matrix
		config = mxrPixelOn * mask + \
                    mxrTest * test + \
                    mxrLow0 * (~numpy.uint32(THL_adj) >> 0 & 1) + \
                    mxrLow1 * (~numpy.uint32(THL_adj) >> 1 & 1) + \
                    mxrLow2 * (~numpy.uint32(THL_adj) >> 2 & 1) + \
                    mxrHigh0 * (~numpy.uint32(THH_adj) >> 0 & 1) + \
                    mxrHigh1 * (~numpy.uint32(THH_adj) >> 1 & 1) + \
                    mxrHigh2 * (~numpy.uint32(THH_adj) >> 2 & 1)
		# NB threshold adjustments are "active low", hence the "~" for these bits
		# Pixel is active with high
		# testpulse is active high
		return config


class marsCamera3RX(marsCameraMP3):
	# THA<0-4> bit values for first threshold.
	THA0 = 1 << 13
	THA1 = 1 << 14
	THA2 = 1 << 15
	THA3 = 1 << 16
	THA4 = 1 << 17

	# THB<0-4> bit values for second threshold.
	THB0 = 1 << 18
	THB1 = 1 << 19
	THB2 = 1 << 20
	THB3 = 1 << 21
	THB4 = 1 << 22

	# Pixel logic inhibit mask bit. (Set to 1 to inhibit the pixel.)
	INHIBIT = 1 << 12

	TEST_BIT = 1 << 23

	def __init__(self, masks=None, dacvals=None, is_csm=False, is_eq=False, is_cm=False):

		"""
		Setup the default values. Override type to 3RX type (its only revision normally)
		"""
		self.marslog = ml.logger(source_module="marscamera", group_module="camera", class_name="marsCamera3RX", object_id=self)
		try:
			self.driver = marsguiref.CONFIG["camera"].get_attr(["driver"], default="testing")
		except Exception as msg:
			self.marslog.print_log("marsCamera3RX - error getting camera driver; setting to 'testing'.", msg, level="error", method="__init__")
			self.driver = "testing"

		if self.driver in ["v5camera"]:
			self.camera_driver = [pyMarsCamera.marsCameraClient()]

		self._camera_found = False
		self._disc_csm_spm = 0
		self._gainmode = 0
		self._ctpr = 0
		self._enable_tp = 0
		marsCameraMP3.__init__(self, masks=masks, dacvals=dacvals, is_csm=is_csm, is_eq=is_eq, is_cm=is_cm)

		self.marslog = ml.logger(source_module="marscamera", group_module="camera", class_name="marsCamera3RX", object_id=self)
		self.adjust_bits = [[self.THA0, self.THA1, self.THA2, self.THA3, self.THA4],
                      [self.THB0, self.THB1, self.THB2, self.THB3, self.THB4]]
		self.inhibit_bit = self.INHIBIT
		self.test_bit = self.TEST_BIT
		self.active_high = True

		self.medipix_type = RX3_TYPE

	def load_default_dacvals(self):
		"""
		Default dacvals are RX3_VALUES from DAC_Config.
		"""
		dacvals = {}
		for i in range(self.chipcount):
			dacvals[i] = RX3_VALUES.copy()

		self.set_dacvals(dacvals)



	def parse_csm_counter(self, counter):
		"""
		Converts a virtual counter index to the index the Medipix3RX uses.

		The virtual index organises them based on our preference for keeping/changing them.

		E.g. CSM > SPM > SPMa

		We should never change SPMa in a scan, so it has lower priority than SPM counters.
		"""
		if self.csm:
			if self.colourmode:
				if counter < 4 and counter >= 0:  # CSM counters are 0, 1, 2, and 3, to 1, 3, 5 and 7
					counter = counter * 2 + 1
				else:
					if self.full_counter:
						if counter >= 4 and counter < 7:  # SPM counters are 4, 5, and 6, to 2, 4, and 6
							counter = (counter - 3) * 2
						elif counter == 7:  # SPMa counter is 7 to 0.
							counter = 0
					else:
						if counter == 4:
							return 0

			else:
				if counter == 1:  # SPMa counter is 1 to 0
					counter = 0
				elif counter == 0:  # CSM counter is 0 to 1
					counter = 1
		return counter



	##
	# Getter and Setter methods.
	#
	@property
	@marsthread.reader
	def dacidxs(self):
		return RX3_SCAN

	@property
	@marsthread.reader
	def dacranges(self):
		return RX3_RANGE

	def set_energy(self, energy, counter=0, chipid=None):
		"""
		Sets the threshold of the chip to an energy.
		"""
		if chipid is None:
			chipid = self.chip
		elif chipid == "all":
			for i in range(self.chipcount):
				self.set_energy(energy, counter, i)
			return

		energy_calibration = self.energy_calibration

		if chipid < len(energy_calibration) and counter < len(energy_calibration[chipid]):
			slope = energy_calibration[chipid][counter]['slope']
			offset = energy_calibration[chipid][counter]['offset']
		else:
			slope = energy_calibration[chipid]['slope']
			offset = energy_calibration[chipid]['offset']
		threshold = int(round(energy * slope + offset))
		self.set_threshold(threshold, counter, chipid)

	def get_energy(self, counter=0, chipid=None):
		"""
		Get the energy value of the current chip threshold
		"""
		if chipid is None:
			chipid = self.chip

		energy_calibration = self.energy_calibration

		if chipid < len(energy_calibration) and counter < len(energy_calibration[chipid]):
			slope = energy_calibration[chipid][counter]['slope']
			offset = energy_calibration[chipid][counter]['offset']
		else:
			self.marslog.print_log("marscamera: no energy calibration data for chip:", chipid, " counter:", counter, "  using default slope=1 offset=0", level="error", method="get_energy")
			slope = 1.0
			offset = 0.0

		threshold = self.get_threshold(counter, chipid)
		return float(threshold - offset) / float(slope)

	def get_threshold(self, counter=0, chipid=None):
		counter = self.parse_csm_counter(counter)
		return marsCameraMP3.get_threshold(self, counter, chipid)

	def set_threshold(self, threshold, counter=0, chipid=None):
		"""
		Sets the threshold of the chip independant of the dacname.

		When in CSM we only use counters 1, 3, 5 and 7, as the rest are in SPM.

		When in equalization mode, we set all the appropriate counters. (Although this should
		also be done in equalisation code, so can maybe be disabled)
		"""
		counter = self.parse_csm_counter(counter)
		if self.equalization and self.colourmode:
			if counter == 0:
				counters = [0, 2, 4, 6]
			elif counter == 1:
				counters = [1, 3, 5, 7]
			else:
				counters = [counter]
			for cnt in counters:
				self.set_dac("Threshold%d" % (cnt), int(threshold), chipid)
		else:
			self.set_dac(("Threshold%d" % (counter)), int(threshold), chipid)

	def get_frame(self, chipid, counter=0, frame=None):
		"""
		Gets the frame for the counter. When charge summing mode
		the counters are 1, 3, 5 and 7 (or rather, just1 in FPM)

		When in equalization it gets the correct counter.
		"""
		if frame is None:
			frame = self._get_frame(chipid)

		counter = self.parse_csm_counter(counter)

		if not self.colourmode:
			if counter == 0 and self.equalization == False or counter == 0 and self.disc_csm_spm == False or counter == 1 and self.equalization and self.disc_csm_spm:
				return frame & 0x000FFF
			elif counter == 1:
				return (frame & 0xFFF000) >> 12
			else:
				return frame
		else:
			if counter in [0, 1, 4, 5]:
				i = 0
			else:
				i = 1
			if counter in [0, 1, 2, 3]:
				j = 1
			else:
				j = 0
			if ((counter in [0, 2, 4, 6]) and (self.equalization == False)) or ((counter in [0, 2, 4, 6]) and (self.disc_csm_spm == False)) or (counter in [1, 3, 5, 7] and self.equalization and self.disc_csm_spm):
				_tempFrames = frame & 0x000FFF
				return _tempFrames[i::2, j::2]
			elif counter in [1, 3, 5, 7]:
				_tempFrames = (frame & 0xFFF000) >> 12
				return _tempFrames[i::2, j::2]
			else:
				return frame

	def set_dac(self, dac_name, dac_value, chipid=None, ignore_lock=False):
		"""
		Sets just one DAC in the marsCamera. Also set the extdac if that is
		the correct dac.
		"""
		if self.has_dac_lock and ignore_lock is False:
			self.marslog.print_log("Set dac called while dac lock active. Aborting.", level="error", method="set_dac")
			import traceback
			traceback.print_stack()
			return
		if chipid is None:
			chipid = self.chip
		if chipid == "all":
			for i in range(self.chipcount):
				self.set_dac(dac_name, dac_value, chipid=i, ignore_lock=ignore_lock)
			return
		dac_name = self.check_dac_name(dac_name)
		if dac_name in self.dacvals[chipid].keys():
			if self.dacidxs[dac_name] == self.extdac:
				self.set_chip_attribute("dac", dac_value, chipid)
				self._set_dac(dac_name, dac_value, chipid=chipid, ignore_lock=ignore_lock)
				self.notify(key="dac")
				return
			else:
				self._set_dac(dac_name, dac_value, chipid=chipid, ignore_lock=ignore_lock)
				dac_value = self._get_dac(dac_name, chipid=chipid)

				try:
					marsprivatetags.set_private_tag(marsdicomref.DCM.ImageData, "RX3_" + str(chipid+1) + "_" + dac_name, dac_value)
				except Exception as msg:
					self.marslog.print_log("Unable to set private tag for dac", dac_name, msg, level="warning", method="set_dac")

		if verbose:
			self.marslog.print_log("Setting DAC", dac_name, level="info", method="set_dac")
		if self.driver == "libmarscamera":
			self.libmars_caller(libmarscamera_py.setDac, self.devid, chipid, dac_name, int(dac_value))
		elif self.driver == "v5camera":
			self.camera_driver[chipid].write_DACs({dac_name:int(dac_value)})
		self.notify(key="dac")

	def get_chip_attribute(self, key, chipid=None):
		"""
		gets the chip attribute, and gets the dac one otherwise.
		"""
		if key.lower() == "dac":
			return self.extdacvalue
		else:
			return marsCamera.get_chip_attribute(self, key, chipid)

	def set_chip_attribute(self, key, val, chipid=None):
		"""
		Set the <val> of the camera-chip <key> attribute.

		Makes sure to check for lowercase values from config files.
		"""
		if chipid is None:
			chipid = self.chip
		if chipid == "all":
			for i in range(self.chipcount):
				self.set_chip_attribute(key, val, i)
			return
		if key.lower() == "csm_spm":
			key = "CSM_SPM"
			self.csm = val
		if key.lower() == "equalization":
			key = "Equalization"
			self.equalizethh = val
		if key.lower() == "colourmode" or key.lower() == "colormode":
			key = "ColourMode"
			self.colourmode = val

		if key.lower() in ["colourmode","colormode", "csm_spm"]:
			if self.colourmode:
				self.pixelsize = 0.11
				if self.csm and self.full_counter is False:
					self.counters = 5
					self.set_counters = 4
					self.csm_counters = 4
					self.set_chip_attribute("counters", 5, chipid=chipid)
					self.frame_order = [0, 1, 3, 5, 7]
					self.frame_type  = ["ARB", "CSM1", "CSM2", "CSM3", "CSM4"]
				elif self.csm:
					self.counters = 8
					self.set_counters = 7
					self.csm_counters = 4
					self.set_chip_attribute("counters", 8, chipid)
					self.frame_order = [0, 1, 3, 5, 7, 2, 4, 6]
					self.frame_type  = ["ARB", "CSM1", "CSM2", "CSM3", "CSM4", "SPM1", "SPM2", "SPM3"]
				else:
					self.counters = 8
					self.set_counters = 8
					self.csm_counters = 0
					self.set_chip_attribute("counters", 8, chipid)
					self.frame_order = [0, 1, 2, 3, 4, 5, 6, 7]
					self.frame_type  = ["SPM1", "SPM2", "SPM3", "SPM4", "SPM5", "SPM6", "SPM7", "SPM8"]
			else:
				self.pixelsize = 0.055
				#if self.csm and self.full_counter is False:
				#	self.counters = 2
				#	self.set_counters = 1
				##	self.csm_counters = 1
				#	self.set_chip_attribute("counters", 2, chipid)
				if self.csm:
					self.counters = 2
					self.set_counters = 1
					self.csm_counters = 1
					self.set_chip_attribute("counters", 2, chipid)
					self.frame_order = [0, 1]
					self.frame_type  = ["ARB", "CSM1"]
				else:
					self.counters = 2
					self.set_counters = 2
					self.csm_counters = 0
					self.set_chip_attribute("counters", 2, chipid)
					self.frame_order = [0, 1]
					self.frame_type  = ["SPM1", "SPM2"]

		if key.lower() == "disc_csm_spm":
			key = "Disc_CSM_SPM"
			self.disc_csm_spm = val
		if key.lower() == "extdac":
			key = "ExtDAC"
			self.extdac = val
		if key.lower() == "dac":
			key = "dac"
			self.extdacvalue = val
		if key.lower() == "gainmode":
			key = "GainMode"
			self.gainmode = val
		if key.lower() == "enable_tp":
			key = "Enable_TP"
			self.enable_tp = val
		if key.lower() == "ctpr":
			key = "ctpr"
			self.ctpr = val
		try:			# only do this for real chip attributes (i.e. don't send pixelsize)
			if key not in ["pixelsize", "counters"]:  # TODO: change this to a test for valid chipAttribute keys..
				if self.driver == "libmarscamera":
					self.libmars_caller(libmarscamera_py.setChipAttribute, self.devid, chipid, key, val)
				elif self.driver == "v5camera":
					if key not in ["dac", "ctpr"]:
						self.camera_driver[chipid].write_OMR({key:val})
					elif key in ["dac"]:
						self.camera_driver[chipid].write_DACs(extdac=val)
		except Exception as msg:
			self.marslog.print_log("marscamera.set_chip_attribute:: Error: %s" % (str(msg)), level="error", method="set_chip_attribute")
#			raise
		else:
			self._set_chip_attribute(key, val, chipid=chipid)
			self.notify(key="attribute")

	##
	# Sets the various OMR values.
	#
	def set_equalization(self, is_eq=True):
		"""
		Set EqualizeTHH (puts Threshold1 results into Counter0 for equalisation purposes)
		"""
		self.equalization = is_eq
		self.set_chip_attribute("Equalization", is_eq, "all")

	def set_disc_csm_spm(self, disc=False):
		self.disc_csm_spm = disc
		self.set_chip_attribute("Disc_CSM_SPM", disc, "all")

	def set_csm(self, is_csm=True):
		"""
		Set Charge Summing Mode (EnablePixelCom).
		"""
		self.csm = is_csm

		self.set_chip_attribute("CSM_SPM", is_csm, "all")
		self.set_colourMode(self.colourmode)

	def set_extdac(self, extdacname="None"):
		if extdacname in self.dacidxs.keys():
			extdac = self.dacidxs[extdacname]
		elif isinstance(extdacname, int):
			extdac = extdacname
		else:
			extdac = 0
		self.set_chip_attribute("ExtDAC", extdac, "all")

	def set_colourMode(self, is_cm=True):
		"""
		Set colourMode

		The scan scripts use self.pixelsize (determines pixel size of pseudo-stitched frames), the GUI uses self.counters
		NOTE:

		* ColourMode can be set by calling set_chip_attribute without calling set_colourMode
		* the system may also have cameras with some chips operating in SPM and some in colourmode
			so pixelsize and counters are chip level attributes [TODO]
		* self.full_counter may be set either before or after calling set_colourMode.

		TODO: we should allow setting colourmode, pixelsize and counters for selected chips.
		"""
		self.colourmode = is_cm

		if self.colourmode:
			self.set_chip_attribute("pixelsize", 0.11, chipid="all")
			if self.csm and self.full_counter is False:
				#self.set_chip_attribute("counters", 4, chipid="all")
				#self.frame_order=[1, 3, 5, 7]  # should be arb + csm i.e [0, 1, 3, 5, 7]
				self.set_chip_attribute("counters", 5, chipid="all")
				self.frame_order=[0, 1, 3, 5, 7]
				self.frame_type  = ["ARB", "CSM", "CSM", "CSM", "CSM"]
			else:
				self.set_chip_attribute("counters", 8, chipid="all")
				if self.csm:
					self.frame_order = [0, 1, 3, 5, 7, 2, 4, 6]
					self.frame_type  = ["ARB", "CSM", "CSM", "CSM", "CSM", "SPM", "SPM", "SPM"]
				else:
					self.frame_order=[0, 1, 2, 3, 4, 5, 6, 7]
					self.frame_type  = ["SPM", "SPM", "SPM", "SPM", "SPM", "SPM", "SPM", "SPM"]
		else:
			self.set_chip_attribute("pixelsize", 0.055, chipid="all")
			if self.csm:
				self.set_chip_attribute("counters", 2, chipid="all")
				self.frame_order=[0, 1]      # [0, 1]
				self.frame_type  = ["ARB", "CSM"]
			else:
				self.set_chip_attribute("counters", 2, chipid="all")
				self.frame_order=[0, 1]
				self.frame_type  = ["SPM", "SPM"]

		self.set_chip_attribute("ColourMode", is_cm, chipid="all")

	def set_full_counter(self, fc):
		self.full_counter = fc
		self.set_colourMode(self.colourmode)

	def load_dacs(self):
		marsCamera.load_dacs(self)

	def load_config(self, chipid="all"):
		if self.csm:
			self.set_csm(False)
			marsCamera.load_config(self, chipid)
			self.set_csm(True)
		else:
			marsCamera.load_config(self, chipid)

	def resend_threshold_dacs(self):
		if self.driver == "libmarscamera":
			for chipid in range(self.chipcount):
				old_th = self._get_dac("Threshold0", chipid=chipid)
				self.libmars_caller(libmarscamera_py.setDac, self.devid, chipid, "Threshold0", int(511))
				self.libmars_caller(libmarscamera_py.setDac, self.devid, chipid, "Threshold0", int(old_th))

				old_th = self._get_dac("Threshold1", chipid=chipid)
				self.libmars_caller(libmarscamera_py.setDac, self.devid, chipid, "Threshold1", int(511))
				self.libmars_caller(libmarscamera_py.setDac, self.devid, chipid, "Threshold1", int(old_th))

		elif self.driver == "v5camera":
			for chipid in range(self.chipcount):
				old_th = self._get_dac("Threshold0", chipid=chipid)
				self.camera_driver[chipid].write_DACs({"Threshold0":int(511)})
				self.camera_driver[chipid].write_DACs({"Threshold0":int(old_th)})
				old_th = self._get_dac("Threshold1", chipid=chipid)
				self.camera_driver[chipid].write_DACs({"Threshold1":int(511)})
				self.camera_driver[chipid].write_DACs({"Threshold1":int(old_th)})

	def expose(self, exptime=None):
		self.resend_threshold_dacs()
		marsCamera.expose(self, exptime)

#	@marsthread.check_num_threads
	def acquire(self, exptime=None, chipid="all"):
		if self.csm and not self.equalization:
			pass
		#self.resend_threshold_dacs()
		marsCamera.acquire(self, exptime, chipid=chipid)

	def update_config(self, config=None):
		"""
		Updates from the configuration file.
		"""
		if config.get_attr(["software_config", "type"]) != RX3_TYPE:
			raise LookupError("Config is for incorrect Medipix type")
		if self.driver == "v5camera":
			for cd in self.camera_driver:
				pass
				#cd.run_command("writeDigitalPots", ["59", "64", "65"])
				#cd.write_OMR({"GainMode":3, "Disc_CSM_SPM":0, "Polarity":0, "Equalization":0, "ColourMode":0, "CSM_SPM":0, "ExtDAC":0, "ExtBGSel":0, "EnableTP":0, "SenseDAC":0})
				#cd.run_command("writeMatrix", ["L", numpy.zeros([98304])])
				#cd.run_command("writeMatrix", ["H", numpy.zeros([98304])])
		if self.driver == "testing":
			self.chipcount = config.get_attr(["mechanical_config", "chip_count"], default=1)
			while len(self.chip_attributes) < self.chipcount:
				self.chip_attributes.append({})
			while len(self._cas_adc) < self.chipcount:
				self._cas_adc.append([])
			old_count = self.masks.shape[0]
			new_masks = numpy.zeros([self.chipcount, 256, 256], dtype='uint32')
			new_masks[0:min(old_count, self.chipcount), :, :] = self.masks[0:min(old_count, self.chipcount), :, :]
			self.masks = new_masks
			new_frames = numpy.zeros([self.chipcount, 256, 256], dtype='uint32')
			new_frames[0:min(old_count, self.chipcount), :, :] = self.frames[0:min(old_count, self.chipcount), :, :]
			self.frames = new_frames
			# Temperature monitor offset and gradient - from Medipix3.0 manual v19
		self._k_temp = [88.75 for chipid in range(self.chipcount)]
		self._m_temp = [-.6073 for chipid in range(self.chipcount)]

		marsCamera.update_config(self, config)
		for i in range(self.chipcount):
			for name in config.get_attr(["software_config", "omr"]).keys():
				self.set_chip_attribute(name, config.get_attr(["software_config", "omr", name]), i)

	def set_equalizeTHH(self, is_eq=True):
		"""
		Override this method so it does nothing. (As it gets called from MP3 legacy code)
		"""
		pass


marsthread.create_RWlock_variable(marsCamera, "_devid", "devid")
marsthread.create_RWlock_variable(marsCamera, "_dacvals", "dacvals")
marsthread.create_RWlock_variable(marsCamera, "_type", "medipix_type")
marsthread.create_RWlock_variable(marsCamera, "_energy_calibration", "energy_calibration")
marsthread.create_RWlock_variable(marsCamera, "_chipcount", "chipcount")
marsthread.create_RWlock_variable(marsCamera, "_counters", "counters")
marsthread.create_RWlock_variable(marsCamera, "_set_counters", "counters")
marsthread.create_RWlock_variable(marsCamera, "_csm_counters", "counters")
marsthread.create_RWlock_variable(marsCamera, "_mode", "mode")
marsthread.create_RWlock_variable(marsCamera, "_pixelsize", "pixelsize")
marsthread.create_RWlock_variable(marsCamera, "_hv", "hv")
marsthread.create_RWlock_variable(marsCamera, "_base_hv", "base_hv")
marsthread.create_RWlock_variable(marsCamera, "_last_good_hv", "last_good_hv")
marsthread.create_RWlock_variable(marsCamera, "_hv_enabled", "hv_enabled")
marsthread.create_RWlock_variable(marsCamera, "_read_bias_current", "read_bias_current")
marsthread.create_RWlock_variable(marsCamera, "_hv_state", "hv_state")
marsthread.create_RWlock_variable(marsCamera, "_hv_idle_control", "hv_idle_control")
marsthread.create_RWlock_variable(marsCamera, "_hv_idle_timeout_start", "hv_idle_timeout_start")
marsthread.create_RWlock_variable(marsCamera, "_hv_warmup_start", "hv_warmup_start")
marsthread.create_RWlock_variable(marsCamera, "_hv_warmup_time", "hv_warmup_time")
marsthread.create_RWlock_variable(marsCamera, "_hv_fixed_output_count", "hv_fixed_output_count")
marsthread.create_RWlock_variable(marsCamera, "_image", "image")
marsthread.create_RWlock_variable(marsCamera, "_frames", "frames")
marsthread.create_RWlock_variable(marsCamera, "_frame_order", "frame_order")
marsthread.create_RWlock_variable(marsCamera, "_frame_type", "frame_type")
marsthread.create_RWlock_variable(marsCamera, "_inhibit_bit", "inhibit_bit")
marsthread.create_RWlock_variable(marsCamera, "_adjust_bits", "adjust_bits")
marsthread.create_RWlock_variable(marsCamera, "_active_high", "active_high")
marsthread.create_RWlock_variable(marsCamera, "_read_hv", "read_hv")
marsthread.create_RWlock_variable(marsCamera, "_read_temp", "read_temp")
marsthread.create_RWlock_variable(marsCamera, "_camera_found", "camera_found")
marsthread.create_RWlock_variable(marsCamera, "_modes", "modes")
marsthread.create_RWlock_variable(marsCamera, "_mode_map", "mode_map")
marsthread.create_RWlock_variable(marsCamera, "_k_temp", "k_temp")
marsthread.create_RWlock_variable(marsCamera, "_m_temp", "m_temp")
marsthread.create_RWlock_variable(marsCamera, "_lm35_temp", "lm35_temp")
marsthread.create_RWlock_variable(marsCamera, "_temperature_history", "temperature_history")
marsthread.create_RWlock_variable(marsCamera, "_hv_history", "hv_history")
marsthread.create_RWlock_variable(marsCamera, "_start_time", "start_time")
marsthread.create_RWlock_variable(marsCamera, "_scanning", "scanning")
marsthread.create_RWlock_variable(marsCamera, "_has_dac_lock", "has_dac_lock")
marsthread.create_RWlock_variable(marsCamera, "_full_counter", "full_counter")

marsthread.create_RWlock_variable(marsCameraMP3, "_cas_adc", "cas_adc")
marsthread.create_RWlock_variable(marsCameraMP3, "_counters", "counters")
marsthread.create_RWlock_variable(marsCameraMP3, "_csm", "csm")
marsthread.create_RWlock_variable(marsCameraMP3, "_eq", "equalizethh")
marsthread.create_RWlock_variable(marsCameraMP3, "_cm", "colourmode")
marsthread.create_RWlock_variable(marsCameraMP3, "_extdac", "extdac")
marsthread.create_RWlock_variable(marsCameraMP3, "_extdacvalue", "extdacvalue")

marsthread.create_RWlock_variable(marsCamera3RX, "_eq", "equalization")
marsthread.create_RWlock_variable(marsCamera3RX, "_disc_csm_spm", "disc_csm_spm")
marsthread.create_RWlock_variable(marsCamera3RX, "_gainmode", "gainmode")
marsthread.create_RWlock_variable(marsCamera3RX, "_enable_tp", "enable_tp")
marsthread.create_RWlock_variable(marsCamera3RX, "_ctpr", "ctpr")
