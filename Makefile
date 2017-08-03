
####### Variables

CXX      := g++
CXXFLAGS := -O2 -g -Wall -fPIC -std=c++0x -Wno-write-strings -Wno-unused-function
CXXFLAGS += -I/usr/include/python2.7
CXXFLAGS += -I/usr/lib64/python2.7/site-packages/numpy/core/include/numpy/
# CXXFLAGS += -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/
LDFLAGS  := -O2 -g
LDFLAGS  += -L/usr/lib/x86_64-linux-gnu -lpython2.7

####### Files
#MarsCamera 
OBJECTS  := main ConfigParser pyEmbedding
TARGET   := code
pref := build
#on/of detail info about process
# verb := 0

####### Build rules

$(TARGET): $(addprefix $(pref)/,$(addsuffix .o,$(OBJECTS)))
# 	@if [[ $(verb) = 1 ]]; then echo true; else echo false; fi 
ifeq ($(verb), 1)
	@echo "Linking executable '$@'..."
# else
# 	@echo false
endif
# 	
	@$(CXX) $^ $(LDFLAGS) -o $@

clean:
ifeq ($(verb), 1)
	@echo "Cleaning..."
endif
	@$(RM) $(pref)/*.o $(pref)/*.d $(TARGET)

####### Compile

$(pref)/%.o: %.cc
$(pref)/%.o: %.cc $(pref)/%.d
ifeq ($(verb), 1)
	@echo "Compiling $<..."
endif
	@$(CXX) -c $(CXXFLAGS) $< -o $@

####### Dependencies

$(pref)/%.d: %.cc
ifeq ($(verb), 1)
	@echo "Generating dependencies for $<..."
endif
	@$(CXX) $(CXXFLAGS) -M $< -o $@

include $(wildcard $(pref)/*.d)

.PRECIOUS: $(pref)/%.d $(pref)/%.o

.PHONY: clean
