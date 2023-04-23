//
// Created by maria on 22.04.23.
//

#ifndef ADAS_SPBU_BASE64_H
#define ADAS_SPBU_BASE64_H

#include <vector>
#include <string>
typedef unsigned char BYTE;

std::string base64_encode(BYTE const* buf, unsigned int bufLen);
std::vector<BYTE> base64_decode(std::string const&);


#endif //ADAS_SPBU_BASE64_H
