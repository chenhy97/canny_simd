#ifndef CANNY_HPP
#define CANNY_HPP
#include <iostream>
#include <stdlib.h>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
#include <core/core.hpp>
#include <imgproc/imgproc.hpp>
#define PI 3.141592653
using namespace std;
using namespace cv;
Mat createGaussianKernel2D(int ksize,float sigma);
void GaussFilter(const Mat src,Mat &dst,int ksize,float sigma);//高斯滤波
void SobelGradDirection(const Mat src,Mat &sobelx,Mat &sobely,double *pointDirection);//Sobel卷积核计算X、Y方向梯度和梯度角
void SobelAmplitude(Mat &sobelx,Mat &sobely,Mat &sobelxy);//求梯度值的幅度值
void inhibit_local_Max(Mat &SobelXY,Mat &Output,double *pointDirection);
void DoubleThreshold(Mat &Input,double LowThreshod,double highThreshold);
void DoubleThresholdLink(Mat &imageInput,double lowThreshold,double highThreshold);  
void hystersis(Mat &Input,Mat &hystersis_result,double LowThreshod,double highThreshold);
#endif