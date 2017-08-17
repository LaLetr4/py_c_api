# In the case of a problem, probably best to propagate the error
# up to the caller, so it can be relayed to the user.  So no
# special logic in here.

# TODO: figure out good default values for the parameters.

import sys
import marsxray
import marscamera_new
from marscamera_new import WrongConfigException
import marsmotor
import marsconvert
import marsguiref
import marsct.marsconfig
from marsasn import Acquisition_Counter
from marsct import marsdicomref

from marsct.exponentialtimer import ExponentialTimer

# The port binding to prevent multiple users accessing the gui is done in scan_gui.
# scan_gui sets marssystem.do_poweroff if it is sucessful in initiating start up.
# This scheme ensures the process that gets the GUI port is the only process that can eable powering off the gantry
enable_poweroff = False

try:
#	from libmarsct_py import MarsError
	from marsmotor import MarsError
except Exception as msg:
	# the import above may fail if the libmarsct_py import in marsmotor failed. (another user!)
	class MarsError(Exception):
		pass

import marsconfig
import marsthread
import marslog as ml
marslog = ml.logger(source_module="marssystem", group_module="system")
import syslog
import warnings
try:
	# This line silences the warning emitted by gtk if it can't find
	# a display.
	with warnings.catch_warnings(record=True):
		import matplotlib.pyplot as pl
except RuntimeError:
	# Can occur when operating over ssh, for example
	pl = None
import gobject

import os

import time

#from GUI_V5.gui_common import context
objects = {"motor": None, "xray": None, "camera": None, "stability_timer": None}
from marsct import marspowerbox as powerbox


# CONTEXT=context.Context()


def mars_connect_function(source_object=None, target_object=None, source_function=None, target_function=None):
	if (source_object is not None) and (target_object is not None):
		try:
			srcfunc = getattr(source_object, source_function)
		except Exception as msg:
			srcfunc = None
		try:
			tgtfunc = getattr(target_object, target_function)
		except Exception as msg:
			tgtfunc = None

		if tgtfunc is None:
			setattr(target_object, target_function, srcfunc)
	return

#obsolete?
class ScanParameters:

	def __init__(self):

		self.marslog = ml.logger(source_module="marssystem", group_module="Scan", class_name="ScanParameters", object_id=self)
		self.marslog.print_log("Use of obsolete class marssystem.ScanParameters", level="error", method="__init__")
		self.voltage = 50  # in kV
		self.current = 100  # in uA
		self.cammag = 100.0
		self.srcmag = 100.0
		self.THLs = [30, 30, 30, 30]
		self.threshold1 = None
		self.acqtimes = [200.0]  # acquire time now in ms
		self.start = {}
		self.finish = {}
		self.nstops = {}

		self.start["sample"] = -20.0
		self.start["camera_translation"] = 0.0
		self.start["rotation"] = 0.0

		self.finish["sample"] = -20.0
		self.finish["camera_translation"] = 0.0
		self.finish["rotation"] = 360.0

		self.nstops["sample"] = 1
		self.nstops["camera_translation"] = 1
		self.nstops["rotation"] = 722


class Majik:

	def __init__(self):

		self.marslog = ml.logger(source_module="marssystem", group_module="system", class_name="Majik", object_id=self)
		if pl is not None:
			pl.interactive(True)
		else:
			raise RuntimeError("Matplotlib failed to load, cannot run "
                            + "in GUI mode.")

	# Now for some magic:
	# This won't work if you have more than one version uncommented.

	############
	# This version uses the GDK lock to do its drawing - not
	# recommended, but simple.
	# def __getattr__ (self, name):
		# def fun (*args, **kwargs):
			# with marsthread.threadlock:
				# return getattr (pl, name) (*args, **kwargs)

		# return fun
	############

	############
	# This version uses the GTK idle loop.  Simple, works well, but can't
	# do return values.
	# def _do_call (self, name, arg_list):
	#	getattr (pl, name) (*arg_list [0], **arg_list [1])

	# def __getattr__ (self, name):
	#	def fun (*args, **kwargs):
	#		arg_list = (args, kwargs)
	#		gobject.idle_add (self._do_call, name, arg_list)

	#	return fun
	############

	############
	# This version also uses the GTK idle loop, but also deals with
	# return values.  Should work well, but is more complicated than
	# necessary unless one is actually interested in pyplot return
	# values.

	def _do_call(self, name, args, kwargs, event):
		event.ret = getattr(pl, name)(*args, **kwargs)
		event.set()

	def __getattr__(self, name):
		if not callable(getattr(pl, name)):
			raise TypeError("Direct access to MatPlotLib members is not "
                            + "supported at this stage.\nIf you have a good "
                            + "reason for needing direct access to MatPlotLib "
                            + "members,\nplease contact the MARS developers.")

		def fun(*args, **kwargs):
			import threading
			event = threading.Event()
			arg_list = (args, kwargs)
			gobject.idle_add(self._do_call, name, args, kwargs, event)
			event.wait()
			return event.ret  # This is set in _do_call. pylint: disable=E

		fun.__name__ = getattr(pl, name).__name__
		fun.__doc__ = getattr(pl, name).__doc__
		return fun
	############


def override(func, func_name, cls):
	def f(self, *args, **kwargs):
		return func(cls.__inst, *args, **kwargs)

	f.__name__ = func.__name__
	cls.__dict__[func_name] = f

def singleton(cls):
	#pdb.set_trace ()
	# Add single instance to the class
	cls.__real_init__ = cls.__init__

	def init(self, *args, **kwargs):
		cls.__init__ = cls.__real_init__
		cls.__inst = cls(*args, **kwargs)
		# Set up all methods to use the internal instance
		for member_name, member in cls.__dict__.items():
			if callable(member):
				override(member, member_name, cls)
		cls.__init__ = lambda _: None

	cls.__init__ = init

	return cls


@singleton
@marsthread.hasLock
#@marsthread.hasRWLock
class MarsSystem:
#	shutdown = False

	@marsthread.takesLock
#	@marsthread.modifier
	def __init__(self, log_obj=None, with_gui=False):

		self.marslog = ml.logger(source_module="marssystem", group_module="system", class_name="MarsSystem", object_id=self)
#		self._shutdown=False
		self.shutdown = False
		if with_gui:
			self._pl = Majik()
		else:
			self._pl = pl
		self.with_gui = with_gui
		self._log = log_obj
		if self._log is None:
			self._log = self.marslog
		self._camera = None
		self._motor = None
		self._xray = None
		self._stability_timer = None
		self.cam_exception = None
		self.motor_exception = None
		self.xray_exception = None
		self._studyFile = None

		self.machine_config = marsct.marsconfig.marsConfig()
		# Best not to use get_attr while the system is reading from the config.
		self.camera_config = self.machine_config.camera_config # self.machine_config.get_attr(["camera_config"])
		marsguiref.CONFIG["machine"] = self.machine_config
		marsguiref.CONFIG["camera"] = self.camera_config

		self._scan_params = {}   # ScanParameters()
		self._sample_width = 0.0

		mainconf = os.path.join(marsconfig.get_system_confdir(), "main.cfg")
		try:
			self.load_config(mainconf)
		except IOError:
			self.marslog.print_log("Main configuration file '%s' not found." % mainconf, level="error", method="__init__")

		self.getStabilityTimer()
                print "#### marsguiref.CONFIG:"
                print " ", marsguiref.CONFIG
                print "#### marsguiref.CONFIG[\"camera\"]:"
                print marsguiref.CONFIG["camera"]

		#try:
		#	self.load_scan_config(os.path.join(marsconfig.get_user_confdir(), "lastused_scan.cfg"))
		#except IOError:
		#	# Not really a problem if the user doesn't have any previous
		#	# scan settings to open up.
		#	pass

#	@property
#	@marsthread.reader
#	def shutdown(self):
#		return self._shutdown
#
#	@shutdown.setter
#	@marsthread.modifier
#	def shutdown(self, shutdown):
#		self._shutdown = shutdown

	@marsthread.takesLock
	def get_config(self):
		#marsconfig.CONFIGS.main = self.machine_config
		marsguiref.CONFIG["machine"] = self.machine_config
		return self.machine_config

	@marsthread.takesLock
	def get_camera_config(self):
		marsguiref.CONFIG["camera"] = self.camera_config
		return self.camera_config

	def using_gui(self):
		return self.with_gui

	def get_scan_config(self):
		return self.scan_config

	def get_shutdown(self):
		return self.shutdown

	def set_shutdown(self, status):
		self.shutdown = bool(status)

	@marsthread.takesLock
	def motorFound(self):
		return (self._motor is not None) and (not self.shutdown)

	@marsthread.takesLock
	def finalise(self):
		if self.shutdown == True:
			return
		self.shutdown = True
		try:
			if self._camera is not None:
				self._camera.finalise()
		except Exception as msg:
			self.marslog.print_log("Exception in camera finalisation", msg, level="warning", method="finalise")
		try:
			if self._motor is not None:
				self._motor.finalise()
				self._motor = None
		except Exception as msg:
			self.marslog.print_log("exception in motor finalisation", msg, level="warning", method="finalise")
		try:
			if self._xray is not None:
				self._xray.finalise()
				self._xray = None
		except Exception as msg:
			self.marslog.print_log("exception in xray finalisation", msg, level="warning", method="finalise")
		try:
			import scan_module
			scan_module.singleton_dicom_queue.close_down()
			# scan_module.singleton_dicom_queue.__del__()
			self.marslog.print_log("Have closed down the dicom transfer queue", level="debug", method="finalise")
			# if scan_module.singleton_dicom_queue.save_file_thread.isAlive():
			#	scan_module.singleton_dicom_queue.save_file_thread.join()
		except Exception as msg:
			self.marslog.print_log("Unable to close dicom file queue", msg, level="error", method="finalise")
			pass


		if enable_poweroff:
			shutdown_thread = marsthread.new_thread(self.do_power_shutdown)
			shutdown_thread.join()
			powerbox.enable_poweroff_events()

	def do_power_shutdown(self):
		try:
			if powerbox.shutdown["send_gantry_poweroff"]:
#				time.sleep(15.0)
				self.marslog.print_log("MARS system: setting gantry power -> off in 15s", level="info", method="do_power_shutdown")
#				Schedule a process based shutdown after system shutdown.
				os.system("nohup python /usr/marssystem/bin/power_off_gantry.py  &")
		except Exception as msg:
			self.marslog.print_log("Exception in updating powerstate on shutdown. Setting gantry power -> off", msg, level="error", method="do_power_shutdown")
#			powerbox.set_gantry_power("off")
			os.system("nohup python /usr/marssystem/bin/power_off_gantry.py time=20 &")
		try:
			if self._log is not None:
				syslog.closelog()
				self._log = None
		except Exception as msg:
			self.marslog.print_log("Exception in closing log", msg, level="error", method="do_power_shutdown")


	def getLog(self):
		return self._log

	@marsthread.takesLock
	def getStabilityTimer(self):
		if self._stability_timer == None:
			_rise_halftime = marsguiref.CONFIG["machine"].get_attr(("software_config","stability_timer","rise_halftime"), default=120.0)
			_decay_halftime = marsguiref.CONFIG["machine"].get_attr(("software_config","stability_timer","decay_halftime"), default=600.0)
			self._stability_timer = ExponentialTimer("v5.0_stability_timer", rise_halftime=_rise_halftime, decay_halftime=_decay_halftime)
			self._stability_timer.set_off()
			self._stability_timer.update()
			objects["stability_timer"] = self._stability_timer
		return self._stability_timer

	@marsthread.takesLock
	def reinitialiseCamera(self):
		self._camera = None
		self.getCamera()

	@marsthread.takesLock
	def getCamera(self):
		# if self.cam_exception is not None:
			#raise self.cam_exception
		print "#### getCamera"
		if (self._camera is None) and (not self.shutdown):
			try:
				self._camera = marscamera_new.getMarsCamera()
				try:
					self._camera.update_config(self.camera_config)
				except WrongConfigException as msg:
					marsguiref.CONFIG["camera"] = self.camera_config = self._camera.config
					if self._motor is not None:
						self._motor.update_camera_config(self.camera_config)

				except Exception as msg:
					self.marslog.print_log("marssystem.getCamera:(msg 1)", msg, level="error", method="getCamera")
			except Exception as e:
				self.cam_exception = e
				self.marslog.print_log("marssystem.getCamera:(msg 2)", e, level="error", method="getCamera")
				self._camera = None
#				raise   # remove this at least until camera powerup/down is working

			objects["camera"] = self._camera
		return self._camera

	@marsthread.takesLock
	def getXray(self):
		# if self.xray_exception is not None:
			#raise self.xray_exception

		if (self._xray is None) and (not self.shutdown):
			try:
				self._xray = marsxray.marsXray(config=self.machine_config.get_attr(["xray_config"]), system_stability=self._stability_timer)

				if self._xray.config.get_attr(["driver"]) == "libmarsct":
					# only do this for V3 & V4 hardware
					if (self._xray is not None) and (self._motor is not None):
						self._xray.get_lockstate = None
						self._xray.get_lockcabinet = None
						self._motor.set_xrayoff = None  # unlock_cabinet uses this.
						mars_connect_function(self._motor, self._xray, "get_lock", "get_lockstate")
						mars_connect_function(self._motor, self._xray, "lock_cabinet", "lock_cabinet")
						mars_connect_function(self._xray, self._motor, "set_off", "set_xrayoff")
						if self._motor is not None:
							self._xray.get_lockstate = self._motor.get_lock
				elif self._xray.config.get_attr(["driver"]) == "netxray":
					if (self._xray is not None) and (self._motor is not None):
						self._motor.get_lock = None  # override old V4/3 get_lock method
						self._motor.lock_cabinet = None  # override old V4/3 lock_cabinet method
						self._motor.unlock_cabinet = None  # override old V4/3 unlock_cabinet method
						self._motor.set_xrayoff = None
						mars_connect_function(self._xray, self._motor, "get_lockstate", "get_lock")
						mars_connect_function(self._xray, self._motor, "lock_cabinet", "lock_cabinet")
						mars_connect_function(self._xray, self._motor, "unlock_cabinet", "unlock_cabinet")
#						mars_connect_function(self._xray, self._motor, "set_off", "set_xrayoff")

				self._xray.reset_fault()   # needed to turn x-ray power on in V5. Should be ok to call on other systems.
				# On V3/4 systems with the libmarsct xray controller this statement runs correctly
				# when the x-ray power is on (i.e. interlock enabled), but will fail if the x-ray power is off (interlock shutdown).
				# The failure behaviour is unpredictable. sometimes the system can be operated normally, and other times an exception is raised that prevents startup.
				# Adding a try/except clause here to catch this silently (as at 09/12/2014) causes downstream failures.
				# ? how to check xray power status on v3/4.

			except Exception as e:
				self.xray_exception = e
				self.marslog.print_log("marssystem.getXray:", e, level="error", method="getXray")
				raise
			objects["xray"] = self._xray
		return self._xray

	@marsthread.takesLock
	def getMotor(self):
		# if self.motor_exception is not None:
			#raise self.motor_exception

		#import sys
		#import traceback
		# sys.stderr.write('\n\n\n')
		# traceback.print_stack()
		# sys.stderr.write('\n\n\n')

		if (self._motor is None) and (not self.shutdown):
			try:
				self._motor = marsmotor.marsMotor(self.machine_config)

				if self._motor.config.get_attr(["motor_config", "driver"]) in ["libmarsct", "testing"]:
					if (self._xray is not None) and (self._motor is not None):
						self._xray.get_lockstate = None
						self._xray.get_lockcabinet = None
						self._motor.set_xrayoff = None  # unlock_cabinet uses this.
						mars_connect_function(self._motor, self._xray, "get_lock", "get_lockstate")
						mars_connect_function(self._motor, self._xray, "lock_cabinet", "lock_cabinet")
						mars_connect_function(self._xray, self._motor, "set_off", "set_xrayoff")
						self._xray.get_lockstate = self._motor.get_lock
				else:													# V5 netxray,netmotor system by default
					if (self._xray is not None) and (self._motor is not None):
						self._motor.get_lock = None			# override old V4/3 get_lock method
						self._motor.lock_cabinet = None			# override old V4/3 lock_cabinet method
						self._motor.unlock_cabinet = None			# override old V4/3 unlock_cabinet method
# self._motor.set_xrayoff		= None  		# unlock_cabinet uses this.
						mars_connect_function(self._xray, self._motor, "get_lockstate", "get_lock")
						mars_connect_function(self._xray, self._motor, "lock_cabinet", "lock_cabinet")
						mars_connect_function(self._xray, self._motor, "unlock_cabinet", "unlock_cabinet")
#						mars_connect_function(self._xray, self._motor, "set_off", "set_xrayoff")

			except MarsError as e:
				self.motor_exception = e
				self.marslog.print_log("Exception occured on getting motor.", e, level="error", method="getMotor")
#				raise
			else:
				#self._motor.update_config(self.machine_config)
				self._motor.update_camera_config(self.camera_config)
				self._motor.initialise_motors()

				if "main" in marsguiref.WINDOW.keys():
					gobject.idle_add(marsguiref.WINDOW["main"].update_status_bar, "Moving all motors to the load position.")
				self._motor.load_pos()
			marsguiref.CONFIG["motor"] = self.machine_config.motor_config
			objects["motor"] = self._motor
		return self._motor

	@marsthread.takesLock
	def load_config(self, filename):
		"""
		Loads the global config
		"""
		self.machine_config.read_config(filename)
		marsguiref.CONFIG["machine"] = self.machine_config
		marsguiref.CONFIG["camera"] = self.camera_config
		if self._camera is not None:
			try:
				self._camera.update_config(self.camera_config)
			except WrongConfigException as msg:
				self.camera_config = self._camera.config
			except Exception as msg:
				self.marslog.print_log("marssystem.load_config:", msg, level="error", method="load_config")

		if self._motor is not None:
			self._motor.update_config(self.machine_config)
			self._motor.update_camera_config(self.camera_config)
		if self._xray is not None:
			self._xray.update_config(self.machine_config.get_attr(["xray_config"]))


	def getPyPlot(self):
		return self._pl

	def getScanParameters(self):
		return self._scan_params

	@marsthread.takesLock
	def set_sample_diameter(self, width=0.0, unit="mm"):
		from_string = "xray" + marsconvert.UNIT_SUFFIX[unit]
		to_string = "xray" + marsconvert.UNIT_SUFFIX["raw"]
		self._sample_width = int(marsconvert.convert_between_values(self.get_config(), from_string, to_string, width))

	@marsthread.takesLock
	def get_sample_diameter(self):
		return self._sample_width
            
        def test(self):
                print "#### marssystem test"
