#pragma once

#ifndef DLL_HEADER 
#define DLL_HEADER 
#include <windows.h> 
#ifdef __cplusplus 
#define EXPORT extern "C" __declspec(dllexport) 
#else 
#define EXPORT __declspec(dllexport) 
#endif 
//EXPORT ushort DewarpNSplit(unsigned short rows, unsigned short cols, int widthStep, unsigned char* rgbData, unsigned char** firstPart, unsigned char** secondPart, int* row, int* col);
EXPORT ushort DewarpNSplit(unsigned short rows, unsigned short cols, int widthStep, unsigned char* rgbData, unsigned char** firstPartR, unsigned char** firstPartG, unsigned char** firstPartB, unsigned char** secondPartR, unsigned char** secondPartG, unsigned char** secondPartB, int* rowFirst, int* colFirst, int* rowSecond, int* colSecond);
//EXPORT ushort DewarpNSplit(unsigned short rows, unsigned short cols, int widthStep, unsigned char* rgbData);
EXPORT ushort DewarpNSplitHSV(unsigned short rows, unsigned short cols, int widthStep, unsigned char* rgbData, unsigned char** firstPartR, unsigned char** firstPartG, unsigned char** firstPartB, unsigned char** secondPartR, unsigned char** secondPartG, unsigned char** secondPartB, int* rowFirst1, int* colFirst1, int* rowSecond1, int* colSecond1);
EXPORT ushort SmartRotateImage(unsigned short rows, unsigned short cols, int widthStep, unsigned char* rgbData, unsigned char** firstPartR, unsigned char** firstPartG, unsigned char** firstPartB, int* rowFirst, int* colFirst);
#endif