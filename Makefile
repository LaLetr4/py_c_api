
####### Variables

CXX      := g++
CXXFLAGS := -O2 -g -Wall -fPIC -std=c++0x -Wno-write-strings -Wno-unused-function
# CXXFLAGS += -I/usr/include/python2.7
CXXFLAGS += -I/usr/include/python2.7 -I/usr/include/x86_64-linux-gnu/python2.7  -fno-strict-aliasing -Wdate-time -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security -DNDEBUG -g -fwrapv
# CXXFLAGS += -I/usr/lib64/python2.7/site-packages/numpy/core/include/numpy/
CXXFLAGS += -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/
LDFLAGS  := -O2 -g
# LDFLAGS  += -L/usr/lib/x86_64-linux-gnu -lpython2.7
LDFLAGS  += -L/usr/lib/python2.7/config-x86_64-linux-gnu -L/usr/lib -lpython2.7 -lpthread -ldl -lutil -lm -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions


####### Files
#MarsCamera 
OBJECTS  := main ConfigParser pyEmbedding 
pref1 := includes
pref2 := src
TARGET   := code
pref := build
#on/of detail info about process
verb := 1

# include tests/makefile

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

$(pref)/%.o: $(pref2)/%.cc
$(pref)/%.o: $(pref2)/%.cc $(pref)/%.d
ifeq ($(verb), 1)
	@echo "Compiling $<..."
endif
	@$(CXX) -c $(CXXFLAGS) -I$(pref1) $< -o $@

####### Dependencies

$(pref)/%.d: $(pref2)/%.cc
ifeq ($(verb), 1)
	@echo "Generating dependencies for $<..."
endif
	@$(CXX) $(CXXFLAGS) -I$(pref1) -M $< -o $@

include $(wildcard $(pref)/*.d)

.PRECIOUS: $(pref)/%.d $(pref)/%.o

.PHONY: clean
