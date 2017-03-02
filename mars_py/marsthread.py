"""
This module defines a thread that can be asynchronously interrupted,
useful for running MARS service scripts with a cancel button.
Probably a good idea not to use this class unless absolutely
necessary, since Python goes to efforts not to make threads stoppable,
with reasons.
Stolen from
http://mail.python.org/pipermail/python-list/2005-December/955076.html,
and slightly modified.

Also contains decorators for thread safety.

Author: Alex Opie
"""

import threading
import ctypes
import sys
import sys
import time

def check_num_threads(func):
    """ Counts how many threads are calling this function, and throw a hissy fit if there are more than one. """
    def f2(self, args, kwargs):
        return func(self, *args, **kwargs)

    def f(self, *args, **kwargs):
        thr = threading.current_thread()
        if hasattr(func, "threads") is False:
            func.threads = {}
        func.threads[thr] = time.time()
        count = 0
        for th in func.threads.keys():
            if time.time() - func.threads[th] > 10.0:
                count += 1
        if count >= 2:
            print "More than 2 threads accessing this function, ", count
            import traceback
            traceback.print_stack()
        return f2(self, args, kwargs)

    f.__name__ = func.__name__
    return f

def build_rwlock_dictionary_class(inherit_class):
    class RWlock_dictionary(inherit_class):
        def __init__(self, dictionary, rwlockvariable):
            self._rwlock = rwlockvariable
            self.baseclass = inherit_class
            self.baseclass.__init__(self, dictionary)

        @reader
        def __getitem__(self, key):
            if issubclass(type(self.baseclass.__getitem__(self, key)), dict):
                self.__setitem__(key, self.baseclass.__getitem__(self, key))    # override to a RWlock_dictionary
            return self.baseclass.__getitem__(self, key)

        @modifier
        def __setitem__(self, key, val):
            if issubclass(type(val), dict):
                if hasattr(val, "baseclass"):
                    classtouse = val.baseclass    # Is already a RWlock dictionary so we should use its baseclass.
                else:
                    classtouse = type(val)

                val = build_rwlock_dictionary_class(classtouse)(val, self._rwlock)
            self.baseclass.__setitem__(self, key, val)

    return RWlock_dictionary


def create_RWlock_variable(object, real_variable_name, property_name):
    @reader
    def fget(self):
        return getattr(self, real_variable_name)

    @modifier
    def fset(self, var):
        if issubclass(type(var), dict):
            if hasattr(var, "baseclass"):
                classtouse = var.baseclass    # Is already a RWlock dictionary so we should use its baseclass.
            else:
                classtouse = type(var)

            var = build_rwlock_dictionary_class(classtouse)(var, self._rwlock)
        setattr(self, real_variable_name, var)

    prop = property(fget, fset)
    setattr(object, property_name, prop)

class ThreadLock:

    def __enter__(self):
        gtk.gdk.threads_enter()

    def __exit__(self, type, value, traceback):
        gtk.gdk.threads_leave()

threadlock = ThreadLock()

# This lock is used to prevent problems arising when threads are
# interrupted.    Without this lock, the thread could be killed before
# it got a chance to release a lock it was holding, which would cause
# the application to lock up.
_immortality = threading.Lock()
#_immortal = False

###
# Decorators
###


def get_new_func_name(cls):
    i = 0
    found_one = False
    while (not found_one):
        new_func_name = "__orig_init%i__" % i
        if not hasattr(cls, new_func_name):
            found_one = True
        i += 1

    return new_func_name

##
# Reader/writer lock section

# Apply this to classes requiring a reader/writer lock.


def hasRWLock(cls):
    """ Apply this to classes requiring a reader/writer lock."""
    new_func_name = get_new_func_name(cls)
    setattr(cls, new_func_name, cls.__init__)

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "_rwlock"):
            self._rwlock = RWLock()
        #self.__orig_init__ (*args, **kwargs)
        getattr(self, new_func_name)(*args, **kwargs)

    cls.__init__ = __init__
    return cls

# Apply this to methods that read object state.


def reader(func):
    """ Apply this to methods that read object state."""

    def f(self, *args, **kwargs):
        self._rwlock.acquireR()
        try:
            return func(self, *args, **kwargs)
        finally:
            self._rwlock.releaseR()

    f.__name__ = func.__name__
    return f

# Apply this to methods that modify object state.


def modifier(func):
    """ Apply this to methods that modify object state."""

    def f(self, *args, **kwargs):
        self._rwlock.acquireW()
        try:
            return func(self, *args, **kwargs)
        finally:
            self._rwlock.releaseW()

    f.__name__ = func.__name__
    return f



##
# Simple lock section
# Apply this to classes requiring a simple lock.
def hasLock(cls):
    """ Apply this to classes requiring a simple lock."""
    #cls.__orig_init__ = cls.__init__
    new_func_name = get_new_func_name(cls)
    setattr(cls, new_func_name, cls.__init__)

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "_lock"):
            self._lock = threading.RLock()
        #self.__orig_init__ (*args, **kwargs)
        getattr(self, new_func_name)(*args, **kwargs)

    cls.__init__ = __init__
    return cls

# There's some fairly magical stuff you can do in Python if you
# look hard enough...
dummy_func = ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 0)


def _force_async_check():
    # Reset the counter, forcing a check right now.
    ctypes.pythonapi.Py_AddPendingCall(dummy_func)



def create_lock_decorators(lockname):
    def class_lock(cls):
        """ Apply this to classes requiring a simple lock."""
        #cls.__orig_init__ = cls.__init__
        new_func_name = get_new_func_name(cls)
        setattr(cls, new_func_name, cls.__init__)

        def __init__(self, *args, **kwargs):
            if not hasattr(self, lockname):
                setattr(self, lockname, threading.RLock())
            #self.__orig_init__ (*args, **kwargs)
            getattr(self, new_func_name)(*args, **kwargs)

        cls.__init__ = __init__
        return cls

    def function_lock(func):
        def f(self, *args, **kwargs):
            try:
                #print "Trying to enter lock, lockname, func:", lockname, func
                getattr(self, lockname).acquire()
                ret = func(self, *args, **kwargs)
                _immortality.acquire()
                return ret
            finally:
                #print "Releasing lock, lockname, func:", lockname, func
                try:
                    getattr(self, lockname).release()
                except RuntimeError:
                    pass    # This just means that the thread was interrupted,
                try:
                    _immortality.release()
                except threading.ThreadError:
                    pass
        f.__name__ = func.__name__
        return f

    return class_lock, function_lock

# Apply this decorator to methods that need to acquire the object lock.
#
# There's a whole swag of commented code in here that I've been using
# for debugging, and that I've been trying out as solutions to the
# problem of threads being killed whilst holding a lock.    I'll delete
# it all once I'm satisfied that it is working -- Alex.
#
# I think I've got it figured out now.    Taking the immortality lock
# prevents the exception being set during the finally block.    When the
# exception is set, the checkinterval is set to 0, so delayed raising
# will not cause problems.    The checkinterval is reset to 100 in the
# InterruptableThread implementation.    If the exception is set in the
# middle of _lock acquisition, then releasing _lock one too many times
# won't hurt, as the thread is being killed, so merely needs to drop
# all held locks.
def takesLock(func):
    def f(self, *args, **kwargs):
        #global _immortal, _immortality
        #########
        #_zerocheckinterval_and_force_check ()
        #sys.setcheckinterval (700)
        # if (self._lock.acquire (0)):
            # print "", threading.currentThread (), ": lock avail"
            #self._lock.release ()
        # else:
            # print "", threading.currentThread (), ": lock held by", self._lock._RLock__owner
        #########
        # print " acquiring lock for", str (self.__class__), func.__name__, "in thread", threading.currentThread()

        # sys.setcheckinterval (300)    #Don't need these because if the acquisition gets interrupted,
        # _force_async_check ()             #then it'll all get cleaned up in the finally block anyway.
        try:
            self._lock.acquire()
            #sys.setcheckinterval (100)
            # print " acquired lock for", str (self.__class__), func.__name__, "in thread", threading.currentThread()
            # return func (self, *args, **kwargs)
            ret = func(self, *args, **kwargs)
            # print "    acquiring immortality in", threading.currentThread ()
            #_zerocheckinterval_and_force_check ()
            #sys.setcheckinterval (1000)
            _immortality.acquire()
            # print "    acquired immortality in", threading.currentThread ()
            #_immortal = True
            #sys.setcheckinterval (300)
            #_force_async_check ()
            return ret
        finally:
            # try:
            # print " releasing lock for", str (self.__class__), func.__name__, "in thread", threading.currentThread()
            try:
                self._lock.release()
            except RuntimeError:
                pass    # This just means that the thread was interrupted,
                # and we didn't manage to get the lock before the interrupt.
                # Problems can only arise if the thread was interrupted,
                # in which case the only problem that can occur is that
                # the lock gets released too many times, which isn't a problem
                # because it just needs to be fully released.
            # except SystemExit:
                # try:
                    #self._lock.release ()
                # except RuntimeError:
                    # pass
                # finally:
                    # raise

            try:
                _immortality.release()
            except threading.ThreadError:
                pass
            # except Exception as msg:
                # print " second release attempt for", str (self.__class__), func.__name__, "in thread", threading.currentThread()
                #self._lock.release ()
            # finally:
            # print " released lock for", str (self.__class__), func.__name__, "in thread", threading.currentThread()
            #sys.setcheckinterval (100)
            # if _immortal:
                #_immortal = False
                # print "    releasing immortality in", threading.currentThread ()
                #_immortality.release ()
                # print "    released immortality in", threading.currentThread ()
            # else:
                # print "    (wasn't immortal)"

    f.__name__ = func.__name__
    return f

###
# End of decorators
###


def _runthread(target, *args, **kwargs):
#    print "**starting function", target, "in thread", threading._get_ident ()
    target(*args, **kwargs)
#    print "**finished", target


def new_thread_oldver(target, *args, **kwargs):
    #t = threading.Thread (target = target, args = args, kwargs = kwargs)
    args = tuple([target] + list(args))
    t = threading.Thread(target=_runthread, args=args, kwargs=kwargs)
    t.start()


def new_thread(target, *args, **kwargs):
    thr = StandardThread(target=target, args=args, kwargs=kwargs)
    thr.start()
    return thr

Lock = threading.Lock


class RWLock:

    """
    Reader/Writer lock.    Used to allow multiple threads simultaneous
    read access, so long as no thread is writing, and block write
    access while other threads are reading.

    Methods:
        * acquireR ()
        * releaseR ()
        * acquireW ()
        * releaseW ()

    Things may break if you don't have a matching release () for
    each acquire (); or if you try to release the wrong type
    (i.e., if you call releaseR () following an acquireW () call).

    Both acquire methods are blocking.
    No non-blocking methods exist (yet).

    The lock is re-entrant and upgradable.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.r_holders = []
        self.cond = threading.Condition(self.lock)
        self.upgraded = []

    def acquireR(self):
        """
        Acquire a read lock.
        """
        self.lock.acquire()
        try:
            self.r_holders.append(threading.currentThread())
            self.clean_out_old_threads()
        finally:
            self.lock.release()

    def releaseR(self):
        """
        Release the read lock you currently hold.
        """
        self.lock.acquire()
        try:
            try:
                self.r_holders.remove(threading.currentThread())
            except ValueError:
                raise AssertionError("Attempted to release lock that wasn't held")

            self.clean_out_old_threads()
        finally:
            self.lock.release()

    def clean_out_old_threads(self):
        """
        Must be called with the lock already acquired
        """
        for thr in self.r_holders:
            if thr.is_alive() is False:
                print "Warning: removed defunct read-locked thread"
                self.r_holders.remove(thr)

        for thr in self.upgraded:
            if thr.is_alive() is False:
                print "Warning: removed defunct write-locked thread"
                self.r_holders.remove(thr)

        if (len(self.r_holders) == 0):
            self.cond.notify()

    def acquireW(self):
        """
        Acquire a write lock.
        """
        self.lock.acquire()
        ct = threading.currentThread()
        while (ct in self.r_holders):
            self.r_holders.remove(ct)
            self.upgraded.append(ct)

        self.clean_out_old_threads()

        while (len(self.r_holders)):
            self.cond.wait()

    def releaseW(self):
        """
        Release a write lock you currently hold.
        """
        ct = threading.currentThread()
        while (ct in self.upgraded):
            self.upgraded.remove(ct)
            self.r_holders.append(ct)

        self.clean_out_old_threads()

        self.lock.release()



class StandardThread(threading.Thread):

    """
    A Class that is created to be as close a possible to threading.Thread
    It provides one feature :: keeps a track of creation and deletion events.
    Thus, we know when a simple thread (generated by a call to new_thread())
    is in existance, or not in existance.
    """

    def __init__(self, target=None, args=(), kwargs=None):
        threading.Thread.__init__(self, target=target, args=args, kwargs=kwargs)
        self.local_name = "new_thread() " + str(target)
        this_module.running_threads.append(self.local_name)
        this_module.running_threads.report()

    def __del__(self):
        """Pick up on the thread closing and being deleted"""
        try:
            this_module.running_threads.remove(self.local_name)
        except Exception as msg:
            pass

class TimeoutException(Exception):
    pass

@hasRWLock
class SafeStringList():

    """Mainintain a threadsafe list of strings. It is an ideal singleton object as it could be called
    from anywhere by any thread. Uses own mutex to enable this class to be used anywhere, such as below
    when maintaining a threadsafe list of what threads are active."""

    def __init__(self):
        self.stringlist = []

    @modifier
    def contains(self, msg):
        """return true/false if the supplied string is in the mutex protected list. """
        return str(msg) in self.stringlist

    @modifier
    def append(self, msg):
        """Extend the list by the supplied string in a threadsafe fashion"""
        self.stringlist.append(msg)

    @modifier
    def remove(self, msg):
        """find the supplied string in the list and remove it (threadsafely)"""
        if msg in self.stringlist:
            self.stringlist.remove(msg)

    @modifier
    def report(self):
        """Write a list describing the contents"""
        if False:
            print str(self.stringlist) + " active threads "
            for c, method in enumerate(self.stringlist):
                try:
                    print str(c + 1), str(method)
                except Exception as msg:
                    pass

    @modifier
    def remove_oldest_entries(self, new_length=20):
        """When doing lots of appends, the oldest entries are at the beginning (index 0). Remove them"""
        while len(self.stringlist) > new_length:
            self.stringlist.pop(0)



# The variable "this_module" is a reference to this marsthread module, of which there is only
# instance. This instance is always available, and is deleted at program end.
# Ensures that when the marsthread module is first loaded, we create a list of threads.
# On the subsequent load, no action is taken.
this_module = sys.modules[__name__]
try:
    unused = this_module.running_threads
except Exception as msg:
    this_module.running_threads = SafeStringList()

@hasRWLock
class MessageQueue(object):
    """
    A class that creates a thread with a "send" and a "receive" method to
    allow for a message-like protocol to be established between threads.
    """
    def __init__(self):
        self.queues = {}
        self.locks = {}
        self.msg_id = 0

    @modifier
    def get_new_msg_id(self):
        ret = self.msg_id
        self.msg_id = self.msg_id + 1
        return ret

    @reader
    def check_thread(self, thr_id):
        if thr_id in self.queues:
            return True
        else:
            self.add_thread(thr_id)

    @modifier
    def add_thread(self, thr_id):
        assert thr_id not in self.queues
        assert thr_id not in self.locks
        self.queues[thr_id] = {}
        self.locks[thr_id] = RWLock()

    def simplify_id(self, thr_id):
        if issubclass(type(thr_id), threading.Thread):
            thr_id = thr_id.ident
        assert issubclass(type(thr_id), int)
        return thr_id

    def send(self, thr_id, msg_id, msg):
        thr_id = self.simplify_id(thr_id)
        self.check_thread(thr_id)

        this_id = self.simplify_id(threading.currentThread())
        self.locks[thr_id].acquireW()
        if msg_id not in self.queues[thr_id]:
            self.queues[thr_id][msg_id] = []
        self.queues[thr_id][msg_id].append((this_id, msg))
        self.locks[thr_id].releaseW()

    def receive(self, msg_id="__first__", timeout=0.0):
        """
        Receive a given msg_id.

        Blocks if timeout = 0.0

        Returns a tuple:
         thr_id, msg_id, msg

         -- thr_id is the id of the sender thread (for returning)
         -- msg_id is the id of the message sent.
        """
        this_id = self.simplify_id(threading.currentThread())
        self.check_thread(this_id)
        start_time = time.time()
        while timeout == 0.0 or time.time() - start_time > timeout:
            self.locks[this_id].acquireR()
            if len(self.queues[this_id]) > 0:
                if msg_id == "__first__":
                    msg_id = sorted(self.queues[this_id].keys())[0]
                    try:
                        self.locks[this_id].acquireW()
                        thr_id, msg =    self.queues[this_id][msg_id].pop(0)
                        if len(self.queues[this_id][msg_id]) == 0:
                            self.queues[this_id].pop(msg_id)
                    finally:
                        self.locks[this_id].releaseW()
                        self.locks[this_id].releaseR()
                    return thr_id, msg_id, msg
                elif msg_id in self.queues[this_id]:
                    try:
                        self.locks[this_id].acquireW()
                        thr_id, msg = self.queues[this_id][msg_id].pop(0)
                        if len(self.queues[this_id][msg_id]) == 0:
                            self.queues[this_id].pop(msg_id)
                    finally:
                        self.locks[this_id].releaseW()
                        self.locks[this_id].releaseR()
                    return thr_id, msg_id, msg
            self.locks[this_id].releaseR()
            time.sleep(1e-5)    # 10us sleep between checks.

        # Timeout occured here.
        raise TimeoutException("No message to receive within timeout")



create_RWlock_variable(MessageQueue, "_queues", "queues")
create_RWlock_variable(MessageQueue, "_locks", "locks")
create_RWlock_variable(MessageQueue, "_msg_id", "msg_id")


class InterruptableThread(threading.Thread):

    """
    Thread subclass that has support for asynchronous cancellation.
    Use the terminate () method to raise a SystemExit exception
    within the thread.    Note that this will not necessarily take
    effect immediately, but should work within the number of python
    bytecode instructions set by sys.setcheckinterval (), which
    defaults to 100.    Note that reducing this number will add extra
    overhead to the running of the code (but can be done).
    """

    def start(self, at_end=None, at_end_args=(), **kwargs):
        if False:
            print "In interruptableThread.start"
        self.__original_run = self.run
        self.run = self.__run    # pylint: disable
        self._at_end = at_end
        self._at_end_args = at_end_args
        self._at_end_kwargs = kwargs
        self.local_thread_name = str(self.__original_run)
        this_module.running_threads.append(self.local_thread_name)
        this_module.running_threads.report()
        threading.Thread.start(self)

    def __run(self):
        self._thrd_id = threading._get_ident()
        try:
            self.__original_run()
        except SystemExit:
            sys.setcheckinterval(100)
        except KeyboardInterrupt:
            self.terminate()
        finally:
            if (self._at_end is not None):
                self._at_end(*self._at_end_args, **self._at_end_kwargs)
            self.run = self.__original_run
            this_module.running_threads.remove(self.local_thread_name)

    def raise_exc(self, excpt):
        # print "    raise_exc: acquiring immortality"
        _immortality.acquire()
        # print "    raise_exc: acquired immortality"
        Nr = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self._thrd_id), ctypes.py_object(excpt))
        while Nr > 1:
            print "raise_exc: retrying ", str(excpt)
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self._thrd_id), None)
            time.sleep(0.1)
            Nr = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self._thrd_id), ctypes.py_object(excpt))

        # Set the checkinterval to zero and leave it there until the exception is caught
        sys.setcheckinterval(0)
        _force_async_check()
        # print "    raise_exc: releasing immortality"
        _immortality.release()
        # print "    raise_exc: released immortality"

    def terminate(self):
        """
        Raise a SystemExit exception in the target thread.    This
        may take up to ( sys.getcheckinterval () ) bytecode
        instructions to take effect.
        """
        try:
            if False:
                print "Termination of thread " + str(self.local_thread_name)
            self._at_end = None
            self.raise_exc(SystemExit)
        except threading.ThreadError as msg:    # Not sure that this can actually happen...
            try:
                print "marsthread.terminate: Thread was already done." + str(msg)
            except Exception as msg:
                print "Warning:: marsthread.terminate: Thread was already done."

