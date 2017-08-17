
#ifndef ConfigParser_h
#define ConfigParser_h

#include <map>

using std::string;

class ConfigParser
{
public:
  /*!A default constructor
   *
   * Creates an instance of the class
   */
  ConfigParser();

   /*!A constructor with in-built parsing
   *
   * Creates an instance of the class and parses config file
   *
   *\param [in] ConfigFileName name of the config file
   */
  ConfigParser(string ConfigFileName);

  /*!
   * A destructor
   *
   * Clears memory
   */
  ~ConfigParser();

  /*!A parsing method
   *
   * Parses config file
   *
   *\param [in] ConfigFileName name of the config file
   */
  void ParseFile(string ConfigFileName);

  /*!Method for finding string type parameter
   *
   * Looks for a parameter with the given name in the config and returns its value
   *
   *\param [in] val name of the parameter
   *\param [in] deflt the value to return if no parameter with such name is found in the config file
   *\return value of the parameter with the given name
   */
  string GetString(string val, string deflt = "");
  const char * GetCString(string val);

  /*!
   * Local cleaning method
   */
  void Clear();

  /*!
   * Method for debugging
   *
   * Prints all parsed parameters
   */
  void Print();

private:
  string m_fname;
  std::map<string,string> m_configs;
};

#endif
