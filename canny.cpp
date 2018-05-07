#include "canny.hpp"
#include <iostream>
#include <stdlib.h>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
#include <core/core.hpp>
#include <imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
Mat createGaussianKernel2D(int ksize,float sigma){
	Mat kernel = Mat::zeros(ksize,ksize,CV_32FC1);
	int center = ksize/2;
	float sum = 0;
	for(int i = 0;i < ksize;i ++){
		float *data = kernel.ptr<float>(i);//对kernel的每一行进行操作。
		for(int j = 0;j < ksize;j ++){
			float temp = ((-1)*((i - center)*(i - center)+(j - center)*(j - center)))/(2*sigma*sigma);
			data[j] = (1/(2*PI*sigma*sigma))*exp(temp);
			sum = sum + data[j];
		}
	}
	for(int i = 0;i < ksize;i ++){
		float *data = kernel.ptr<float>(i);
		for(int j = 0;j < ksize;j ++){
			data[j] = data[j]/sum;
		}
	}
	return kernel;
}
void GaussFilter(const Mat src,Mat &dst,int ksize,float sigma){
	Mat kernel = createGaussianKernel2D(ksize,sigma);
	int height = src.rows;
	int width = src.cols;
	int type_src = src.type();
	dst = Mat::zeros(dst.size(),CV_8UC1);
	if(!src.data || src.channels() != 1){
		return;
	}
	double temparray[1000];
	for(int i = 0;i < ksize*ksize;i ++){
		temparray[i] = 0;//初始化处理数组。
	}
	
	int index = 0;
	for(int i = 0;i < ksize;i ++){
		float* data = kernel.ptr<float>(i);
		for(int j = 0;j < ksize;j ++){
			temparray[index] = data[j];//化二维数组为一维数组
			index ++;
		}
	}
	//开始滤波
	for(int i = 0;i < height;i ++){
		uchar *data = dst.ptr<uchar>(i);
		for(int j = 0;j < width;j ++){
			int k = 0;
			for(int l = -ksize/2;l <= ksize/2;l ++){
				for(int g = -ksize/2;g <= ksize/2;g ++){
					//处理边界值
					int row = i + 1;
					int col = j + g;
					row = row < 0?0:row;
					row = row >= height?(height-1):row;
					col = col < 0?0:col;
					col = col >= width?(width-1):col;
					//卷积和
					data[j] = data[j] + temparray[k]*src.at<uchar>(row,col);
					k++;
				}
			}
		}
	}
}
void SobelGradDirection(const Mat src,Mat &sobelx,Mat &sobely,double *pointDirection){
	for(int i =0;i < src.rows - 1;i ++){
		pointDirection[i] = 0;
	}
	sobelx = Mat::zeros(src.size(),CV_32SC1);
	sobely = Mat::zeros(src.size(),CV_32SC1);
	uchar* P = src.data;
	uchar* px = sobelx.data;
	uchar* py = sobely.data;

	int step = src.step;
	int stepXY = sobelx.step;
	long long  k = 0;
	int i,j;
	for(i = 1;i < (src.rows - 1);i ++){
		for(j = 1;j <(src.cols - 1);j ++){
			double gradY = P[(i-1)*step + j + 1] + P[i*step + j + 1]*2 + P[(i + 1)*step + j + 1] -  P[(i-1)*step + j - 1] - P[i*step + j - 1]*2 - P[(i + 1)*step + j - 1];
			py[i * stepXY + j * (stepXY/step)] = abs(gradY);
			double gradX = P[(i+1)*step + j - 1] + P[(i + 1) * step + j] * 2+P[( i + 1) * step+ j + 1]-P[(i-1)*step+j-1]-P[(i-1)*step+j]*2-P[(i-1)*step+j+1];
			px[i * stepXY + j * (stepXY/step)] = abs(gradX);
			if(gradX == 0){
				gradX = 0.0000000001;//防止除法为0
			}
			pointDirection[k] = atan(gradY/gradX)*57.3;
			pointDirection[k] += 90;
			k ++;
		}
	}
	convertScaleAbs(sobelx,sobelx);//对于每个输入数组的元素函数convertScaleAbs 进行三次操作依次是：缩放，得到一个绝对值，转换成无符号8位类型
	convertScaleAbs(sobely,sobely);
}
void SobelAmplitude(Mat &sobelx,Mat &sobely,Mat &SobelXY){
	SobelXY = Mat::zeros(sobelx.size(),CV_32FC1);
	for(int i = 0;i < SobelXY.rows;i ++){
		float *data = SobelXY.ptr<float>(i);
		uchar *datax = sobelx.ptr<uchar>(i);
		uchar *datay = sobely.ptr<uchar>(i);
		for(int j = 0;j < SobelXY.cols;j ++){
			data[j] = sqrt(datax[j] * datax[j] + datay[j]*datay[j]);
		}
	}
	convertScaleAbs(SobelXY,SobelXY);
}
void inhibit_local_Max(Mat &SobelXY,Mat &Output,double *pointDirection){
	Output = SobelXY.clone();
	int k = 0;
	int i,j;
	for(int i = 1;i < SobelXY.rows - 1;i ++){
		for(int j = 1;j < SobelXY.cols - 1;j ++){
			int temp00=SobelXY.at<uchar>((i-1),j-1);  
            int temp01=SobelXY.at<uchar>((i-1),j);  
            int temp02=SobelXY.at<uchar>((i-1),j+1);  
            int temp10=SobelXY.at<uchar>((i),j-1);  
            int temp11=SobelXY.at<uchar>((i),j);  
            int temp12=SobelXY.at<uchar>((i),j+1);  
            int temp20=SobelXY.at<uchar>((i+1),j-1);  
            int temp21=SobelXY.at<uchar>((i+1),j);  
            int temp22=SobelXY.at<uchar>((i+1),j+1);
            if(pointDirection[k]>0&&pointDirection[k]<=45)  
            {  
                if(temp11<=(temp12+(temp12-temp11)*tan(pointDirection[(i-1)*(Output.cols)+j]))||(temp11<=(temp10+(temp20-temp10)*tan(pointDirection[(i-1)*(Output.cols)+j]))))  
                {  
                    Output.at<uchar>(i,j)=0;  
                }  
            }     
            if(pointDirection[k]>45&&pointDirection[k]<=90)  
  
            {  
                if(temp11<=(temp01+(temp02-temp01)/tan(pointDirection[i*(Output.cols - 1)+j]))||temp11<=(temp21+(temp20-temp21)/tan(pointDirection[i*(Output.cols - 1)+j])))  
                {  
                    Output.at<uchar>(i,j)=0;  
  
                }  
            }  
            if(pointDirection[k]>90&&pointDirection[k]<=135)  
            {  
                if(temp11<=(temp01+(temp00-temp01)/tan(180-pointDirection[i*(Output.cols - 1)+j]))||temp11<=(temp21+(temp22-temp21)/tan(180-pointDirection[i*(Output.cols - 1)+j])))  
                {  
                    Output.at<uchar>(i,j)=0;  
                }  
            }  
            if(pointDirection[k]>135&&pointDirection[k]<=180)  
            {  
                if(temp11<=(temp10+(temp00-temp10)*tan(180-pointDirection[i*(Output.cols - 1)+j]))||temp11<=(temp12+(temp22-temp11)*tan(180-pointDirection[i*(Output.cols - 1)+j])))  
                {  
                    Output.at<uchar>(i,j)=0;  
                }  
            }  
            k++;    
		}
	}
}
void DoubleThreshold(Mat &Input,double LowThreshod,double highThreshold){
	for(int i = 0;i < Input.rows;i ++){
		uchar *data = Input.ptr<uchar>(i);
		for(int j = 0;j < Input.cols;j ++){
			if(data[j] > highThreshold){
				data[j] = 255;
			}
			if(data[j] < LowThreshod){
				data[j] = 0;
			}
		}
	}
}
void DoubleThresholdLink(Mat &imageInput,double lowThreshold,double highThreshold)  
{  
    for(int i=1;i<imageInput.rows-1;i++)  
    {  
    	uchar *data0 = imageInput.ptr<uchar>(i - 1);
    	uchar *data1 = imageInput.ptr<uchar>(i);
    	uchar *data2 = imageInput.ptr<uchar>(i + 1);
        for(int j=1;j<imageInput.cols-1;j++)  
        {  
            if(imageInput.at<uchar>(i,j)>lowThreshold&&imageInput.at<uchar>(i,j)<255)  
            {  
                if(data0[j -1] == 255 || data0[j] == 255 || data0[j + 1] == 255 ||  
                   data1[j-1] == 255 || data1[j + 1] == 255||  
                   data2[j -1] == 255 || data2[j] == 255 || data2[j + 1] == 255)  
                {  
                    data1[j] = 255;  
                    DoubleThresholdLink(imageInput,lowThreshold,highThreshold); //递归调用  
                }     
                else  
            {  
                    data1[j] = 0;  
            }                 
            }                 
        }  
    }  
}  