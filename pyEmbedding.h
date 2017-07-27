
#ifndef pyEmbedding_h
#define pyEmbedding_h

#include <iostream>
#include "ConfigParser.h"
#include <string>
#include <cstring>

using namespace std;

class pyEmbedding{
public:
    const char * modName;
    const char * dictName;
    
    pyEmbedding(std::string);
    
    ~pyEmbedding(){
    }
    
protected:
    
    static unsigned counter;
    
    void setPythonPath(const char * _path);

    //for testing
    void printEmbInfo();

};


#endif
