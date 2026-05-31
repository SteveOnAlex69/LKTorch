#include <DebugAssist.hpp>

void throw_error(std::string s){
	std::cerr << std::endl;
	std::cerr << "Error found!!!!!!" << std::endl;
	std::cerr << s << std::endl;
	exit(0);
}

