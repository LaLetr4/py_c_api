
####### Variables

CXX      := g++
CXXFLAGS := -O2 -g -Wall -fPIC -std=c++0x -Wno-write-strings -Wno-unused-function
CXXFLAGS += -I/usr/include/python2.7
# CXXFLAGS += -I/usr/lib64/python2.7/site-packages/numpy/core/include/numpy
CXXFLAGS += -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/
LDFLAGS  := -O2 -g
LDFLAGS  += -L/usr/lib/x86_64-linux-gnu -lpython2.7

####### Files

OBJECTS  := main MarsCamera
TARGET   := code
pref := build

####### Build rules

$(TARGET): $(addprefix $(pref)/,$(addsuffix .o,$(OBJECTS)))
	@echo "Linking executable '$@'..."
	@$(CXX) $^ $(LDFLAGS) -o $@

clean:
	@echo "Cleaning..."
	@$(RM) $(pref)/*.o $(pref)/*.d $(TARGET)

####### Compile

$(pref)/%.o: %.cc
$(pref)/%.o: %.cc $(pref)/%.d
	@echo "Compiling $<..."
	@$(CXX) -c $(CXXFLAGS) $< -o $@

####### Dependencies

$(pref)/%.d: %.cc
	@echo "Generating dependencies for $<..."
	@$(CXX) $(CXXFLAGS) -M $< -o $@

include $(wildcard $(pref)/*.d)

.PRECIOUS: $(pref)/%.d $(pref)/%.o

.PHONY: clean
