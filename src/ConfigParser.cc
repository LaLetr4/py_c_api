#include "ConfigParser.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>

#include <cstdlib>

using namespace std;

ConfigParser::ConfigParser()
{
  Clear();
}

ConfigParser::ConfigParser(string ConfigFileName){
//     cout<<"ConfigParser object is created\n";
  Clear();
  ParseFile(ConfigFileName);
}

ConfigParser::~ConfigParser()
{
  Clear();
//   cout<<"ConfigParser object is destroyed\n";
}

void ConfigParser::Clear(){
  m_configs.clear();
}

void ConfigParser::ParseFile(string ConfigFileName) {
#ifdef TEST
cout<<"Configure an embedding from file "<<ConfigFileName<<endl;
#endif
  m_fname = ConfigFileName;
  ifstream ifile;
  ifile.open(m_fname.c_str());
  if (! ifile) {
    cerr << "Unable open file " << m_fname << " for reading. Exiting" << endl;
    exit(-1);
  }
  string line;
  while(std::getline(ifile,line)) {
    if(line==""||line[0]=='#') continue;
    string flag,value,dummy;
    istringstream istream(line);
    istream >> flag >> dummy >> value;
    if(flag=="" || value=="" || dummy!="=") {
      continue;
    }
    m_configs[flag]=value;
  }
}

string ConfigParser::GetString(string val, string deflt) {
  if (m_configs.count(val)){
    return m_configs[val];
  }
  return deflt;
}

const char * ConfigParser::GetCString(string val) {
  if (m_configs.count(val)){
    return m_configs[val].c_str();
  }
  return 0;
}

void ConfigParser::Print(){
  for (std::map<string,string>::iterator it=m_configs.begin(); it!=m_configs.end(); ++it)
    std::cout << it->first << " => " << it->second <<endl;
}

