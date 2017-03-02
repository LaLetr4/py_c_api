
####### Variables

CXX      := g++
CXXFLAGS := -O2 -g -Wall -fPIC -std=c++0x -Wno-write-strings -Wno-unused-function
CXXFLAGS += -I/usr/include/python2.7
LDFLAGS  := -O2 -g
LDFLAGS  += -L/usr/lib/x86_64-linux-gnu -lpython2.7

####### Files

OBJECTS  := main MarsCamera
TARGET   := code

####### Build rules

$(TARGET): $(addsuffix .o,$(OBJECTS))
	@echo "Linking executable '$@'..."
	@$(CXX) $^ $(LDFLAGS) -o $@

clean:
	@echo "Cleaning..."
	@$(RM) *.o *.d $(TARGET)

####### Compile

%.o: %.cc
%.o: %.cc %.d
	@echo "Compiling $<..."
	@$(CXX) -c $(CXXFLAGS) $< -o $@

####### Dependencies

%.d: %.cc
	@echo "Generating dependencies for $<..."
	@$(CXX) $(CXXFLAGS) -M $< -o $@

include $(wildcard *.d)

.PRECIOUS: %.d %.o

.PHONY: clean
