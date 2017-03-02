# @package marsct.marsobservable
# This module contains an observable base class for creating observable objects
#
# Sub-class any object to be observed from this module's class and then use the
# marsObservable.add_observer() and marsObservable.notify() methods to obtain
# updates when the state of the observed class changes.
# This may be used with a GUI to refresh the interfaces as required.



class marsObservable(object):

    """
    A class for marssystem objects to inherit from to gain observer properties.

    This means we can add functions that go off on certain events whenever the
    notify command is called. Then when something significant changes in the object,
    notify can be called with regards to the correct event.

    The key "all" has special meaning. If a function is added as an observer to "all",
    then it will observe all notify events. If notify goes off to "all", then every
    function will go off - regardless of its event key.
    """

    def __init__(self):
        # The current list of observer (methods).
        self.observers = {}

    def add_observer(self, observerFunc, key="all"):
        """
        Add an observer to the instance list of observers.
        Returns None.
        """
        if key not in self.observers.keys():
            self.observers[key] = []
        self.observers[key].append(observerFunc)

    def remove_observer(self, observerFunc, key="all"):
        """
        Remove an observer from the list of observers.
        Returns None.
        """
        if key == "all":
            for k in self.observers.keys():
                self.remove_observer(observerFunc, k)
        else:
            if key in self.observers.keys():
                if observerFunc in self.observers[key]:
                    self.observers[key].remove(observerFunc)
                elif observerFunc == "all":
                    self.observers[key] = []
                if len(self.observers[key]) == 0:
                    self.observers.pop(key)

    def notify(self, message=None, key="all"):
        """
        Notify each observer in the list of observers by calling the registered
        function. The call may include a message <key> if required.
        """
        if key == "all":
            for k in self.observers.keys():
                if k != "all":    # notify 'all' in this case is a feedback loop
                    self.notify(message, key=k)
        if key in self.observers.keys():
            for eachFunc in self.observers[key]:
#                print "calling ",eachFunc,message
                if message is None:
                    eachFunc()
                else:
                    eachFunc(message)
        if "all" in self.observers.keys():    # handle 'all' separately here
            for eachFunc in self.observers["all"]:
#                print "calling ",eachFunc,message
                if message is None:
                    eachFunc()
                else:
                    eachFunc(message)

