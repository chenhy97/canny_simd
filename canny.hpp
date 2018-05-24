#ifndef CANNY_HPP
#define CANNY_HPP
#include <iostream>
#include <stdlib.h>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
#include <core/core.hpp>
#include <imgproc/imgproc.hpp>
#include <immintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
//#define USE_SSE2 1
#define PI 3.141592653
using namespace std;
using namespace cv;
Mat createGaussianKernel2D(int ksize,float sigma);
void GaussFilter(const Mat src,Mat &dst,int ksize,float sigma);//高斯滤波
void simd_GaussFilter(const Mat src,Mat &dst,int ksize,float sigma);//高斯滤波simd，加速成功，
void SobelGradDirection(const Mat src,Mat &sobelx,Mat &sobely,float *pointDirection);//Sobel卷积核计算X、Y方向梯度和梯度角
//求梯度角加速成功
void simd_SobelGradDirection(const Mat src,Mat &sobelx,Mat &sobely,float *pointDirection);//Sobel卷积核计算X、Y方向梯度和梯度角
void SobelAmplitude(Mat &sobelx,Mat &sobely,Mat &sobelxy);//求梯度值的幅度值
void simd_SobelAmplitude(Mat &sobelx,Mat &sobely,Mat &sobelxy);//求梯度值的幅度值,成功加速
void inhibit_local_Max(Mat &SobelXY,Mat &Output,float *pointDirection);
void DoubleThreshold(Mat &Input,uchar LowThreshod,uchar highThreshold);
void simd_DoubleThreshold(Mat &Input,uchar LowThreshod,uchar highThreshold);//双阙值抑制成功加速。
void DoubleThresholdLink(Mat &imageInput,uchar lowThreshold,uchar highThreshold);  
void hystersis(Mat &Input,Mat &hystersis_result,double LowThreshod,double highThreshold);
#endif