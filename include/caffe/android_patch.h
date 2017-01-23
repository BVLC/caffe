//
// Created by daniil on 1/11/17.
//

#ifndef CLBLAST_ANDROID_PATCH_H
#define CLBLAST_ANDROID_PATCH_H

#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
namespace std {
    template<typename T>
    std::string to_string(T value) {
        //create an output string stream
        std::ostringstream os;

        //throw the value into the string stream
        os << value;

        //convert the string stream into a string and return
        return os.str();
    }

    inline double stod(string value) {
        return strtod (value.c_str(), NULL);
    }

    inline int stoi(string value) {
        return strtol (value.c_str(),NULL,0);
    }
}
#endif //CLBLAST_ANDROID_PATCH_H


