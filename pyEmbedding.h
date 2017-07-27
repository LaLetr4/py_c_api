
#ifndef pyEmbedding_h
#define pyEmbedding_h

#include <iostream>

using namespace std;

class pyEmbedding{
public:
    const char * modName;
    const char * dictName;
    
    pyEmbedding(const char * _modName, const char * _dictName, const char * _path):modName(_modName),dictName(_dictName){
        setPythonPath("/usr/lib/marsgui:/usr/lib/marsgui/marsct:/home/marsadmin/drv_py_nw/v2");
        cout<<"pyEmbedding object is created\n";
    }
    
    ~pyEmbedding(){
        cout<<"pyEmbedding object is destroyed\n";
    }
    
protected:
    void setPythonPath(const char * _path){
    /*sets the PYTHONPATH = /usr/lib/marsgui:/usr/lib/marsgui/marsct:/home/marsadmin/drv_py_nw/v2*/
    const char * path="PYTHONPATH";
    setenv(path,_path,0); // doesn't overwrite
    cout<<"Set "<<path<<" = "<<getenv(path)<<endl; 
}
    
};


#endif
