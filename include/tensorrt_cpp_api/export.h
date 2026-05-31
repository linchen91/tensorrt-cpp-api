#pragma once

#ifdef _WIN32
  #ifdef TRTCPP_BUILD_SHARED
    #define TRTCPP_API __declspec(dllexport)
  #elif defined(TRTCPP_SHARED)
    #define TRTCPP_API __declspec(dllimport)
  #else
    #define TRTCPP_API
  #endif
#else
  #define TRTCPP_API
#endif
