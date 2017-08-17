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
};

ConfigParser::ConfigParser(string ConfigFileName){
  Clear();
  ParseFile(ConfigFileName);
}

ConfigParser::~ConfigParser()
{
  Clear();
};

void ConfigParser::Clear(){
  m_configs.clear();
};

void ConfigParser::ParseFile(string ConfigFileName)
{
  m_fname = ConfigFileName;
  ifstream ifile;
  ifile.open(m_fname.c_str());
  if (! ifile)
    {
      cerr << "Unable open file " << m_fname << " for reading. Exiting" << endl;
      exit(-1);
    }
  string line;
  while(std::getline(ifile,line))
    {
      if(line==""||line[0]=='#') continue;
      string flag,value,dummy;
      istringstream istream(line);
      istream >> flag >>dummy>> value;
      if(flag=="" || value=="" || dummy!="=")
	{
          continue;
	}
      m_configs[flag]=value;
    }
}

// string ConfigParser::GetString(string val, string deflt)
// {
//   if (m_configs.count(val)){
//     return m_configs[val];
//   }
//   return deflt;
// };

// long ConfigParser::GetLong(string val, long deflt)
// {
//   if (m_configs.count(val)){
//     return atol(m_configs[val].c_str());
//   }
//   return deflt;
// }
// 
// int ConfigParser::GetInt(string val, int deflt)
// {
//   if (m_configs.count(val)){
//     return atoi(m_configs[val].c_str());
//   }
//   return deflt;
// }
// 
// double ConfigParser::GetDouble(string val, double deflt)
// {
//   if (m_configs.count(val)){
//     return atof(m_configs[val].c_str());
//   }
//   return deflt;
// }

void ConfigParser::Print(){
  for (std::map<string,string>::iterator it=m_configs.begin(); it!=m_configs.end(); ++it)
    std::cout << it->first << " => " << it->second <<endl;
}

