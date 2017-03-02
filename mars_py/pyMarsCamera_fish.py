
import numpy

#test module for debugging

class marsCameraClient:
    def __init__(self):
        print "__init__ called"
        self.frame = [[],[]]
    def connect(self, address, port = 1234):
        print "connect called: address =", address, "& port =", port
    def disconnect(self):
        print "disconnect called"
    def write_OMR(self, omrvals = {}):
        print "write_OMR called: omrvals =", omrvals
    def write_DACs(self, dacvals = {}, extdac=None, extbg=None):
        print "write_DACs called: extdac =", extdac, "& extbg =", extbg, "& dacvals =", dacvals
    def acquire(self, exptime):
        print "acquire called: exptime =", exptime
    def download_image(self, counter):
        print "download_image called: counter =", counter
        tmp = numpy.zeros([3, 3])
        tmp[0,1] =  1
        tmp[0,2] =  2
        tmp[1,0] = 10
        tmp[1,1] = 11
        tmp[1,2] = 12
        tmp[2,0] = 20
        tmp[2,1] = 21
        tmp[2,2] = 22
        self.frame[counter] = tmp.astype('uint16')
    def get_frame(self, counter):
        print "get_frame called: counter =", counter
        return self.frame[counter].tostring()

def test_print(smth):
    print smth
