#pragma once
/*
#ifndef __CUDACC__
#define __CUDACC__
#endif
*/
#ifdef __CUDACC__
    #define HOSTDEVICE __host__ __device__
#else
    #define HOSTDEVICE
#endif
