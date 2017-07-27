
#include "pyEmbedding.h"

    pyEmbedding::pyEmbedding(std::string settings_file = "conf.ini"){
        ConfigParser my_cfg("conf.ini");
        modName = (my_cfg.GetString("moduleName")).c_str();
        dictName = (my_cfg.GetString("className")).c_str();
        const char * pytPath = (my_cfg.GetString("pythonPath")).c_str();
        setPythonPath(pytPath);
    }
    
    void pyEmbedding::setPythonPath(const char * _path){
        const char * path="PYTHONPATH";
        setenv(path,_path,0); // doesn't overwrite
        cout<<"Set "<<path<<" = "<<getenv(path)<<endl; 
    }
    
    void pyEmbedding::printEmbInfo(){
        cout<<"Module name is "<<modName<<endl;
        cout<<"Class name is "<<dictName<<endl;
    }
    
