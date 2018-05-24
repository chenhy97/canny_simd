#include <iostream>
#include "io.hpp"
#include "canny.hpp"
#include <sys/time.h>
#include <xmmintrin.h>
using namespace std;
using namespace cv;
int main(int argc,char* argv[]){
	struct timeval start;
	struct timeval end;
	float time_use = 0;
	string name;
	cin >> name;
	Mat img = read_pgm_image(name);
	string save_name;
	cin >> save_name;
	unsigned char lowthreshold,highthreshold;
	cin >> lowthreshold >> highthreshold;
	Mat grayImg;
	if(Conervt2Gray(img,grayImg) == 0){//处理读入灰度图像的情况
		grayImg = img;
	}
	Mat GaussIMG = img;
	float sum_time = 0;
	float simd_time = 0;

	cout << "Scalar Gauss Filtering..." << endl;//高斯滤波
	gettimeofday(&start,NULL);
	GaussFilter(grayImg,GaussIMG,5,1);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
	cout << "Scalar GaussFilter time:" << time_use << "us"<< endl << endl;
	sum_time += time_use;
	gettimeofday(&start,NULL);

	cout << "Simd Gauss Filtering..." << endl;//Simd 高斯滤波
	simd_GaussFilter(grayImg,GaussIMG,5,1);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
	cout << "Simd GaussFilter time:" << time_use << "us"<< endl << endl;
	write_pgm_image(save_name,GaussIMG);
	simd_time += time_use;

	Mat SobelX = GaussIMG;
	Mat SobelY = GaussIMG;
	Mat SobelXY = GaussIMG;
	string X_name = save_name + "X";
	string Y_name = save_name + "Y";
	string XY_name = save_name + "XY";
	string inhibit_name = save_name + "_inhibit";
	string h_name = save_name + "_hyst";
	float *pointDirection;
	pointDirection = new float __attribute__((aligned(16))) [(GaussIMG.rows - 1) *(GaussIMG.cols - 1)];//16字节对齐

	cout <<"+++++++++++++++++++++++++++++++" << endl;
	cout << "Scalar Sobel Grad Getting ...." << endl;//sobel算子求梯度
	gettimeofday(&start,NULL);
	SobelGradDirection(GaussIMG,SobelX,SobelY,pointDirection);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "Scalar Grad Getting time: " << time_use << "us" << endl << endl;
	sum_time += time_use;

	cout << "Simd Sobel Grad Getting ...." << endl;//Simd sobel算子求梯度
	gettimeofday(&start,NULL);
	simd_SobelGradDirection(GaussIMG,SobelX,SobelY,pointDirection);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "Simd Grad1 Getting time: " << time_use << "us" << endl << endl;
	simd_time += time_use;

	cout <<"+++++++++++++++++++++++++++++++" << endl;
	cout << "Scalar Sobel Amplitude Getting ...." << endl;//sobel算子求幅值
	gettimeofday(&start,NULL);
	SobelAmplitude(SobelX,SobelY,SobelXY);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "Scalar Amplitude Getting time: " << time_use << "us" << endl << endl;
	sum_time += time_use;

	cout << "Simd Sobel Amplitude Getting ...." << endl;//SIMD sobel算子求幅值
	gettimeofday(&start,NULL);
	simd_SobelAmplitude(SobelX,SobelY,SobelXY);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "Simd Amplitude Getting time: " << time_use << "us" << endl << endl;
	simd_time += time_use;

	
	
	Mat inhibit_Output = GaussIMG;
	cout <<"+++++++++++++++++++++++++++++++" << endl;
	cout << "Inhibit local Max ...." << endl;//局部非极大值抑制
	gettimeofday(&start,NULL);
	inhibit_local_Max(SobelXY,inhibit_Output,pointDirection);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "Inhibit local Max time: " << time_use << "us" << endl << endl;
	sum_time += time_use;
	simd_time += time_use;

	cout <<"+++++++++++++++++++++++++++++++" << endl;	
	cout << "Scalar DoubleThresholding...." << endl;//双阙值限制
	gettimeofday(&start,NULL);
	DoubleThreshold(inhibit_Output,lowthreshold,highthreshold);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "DoubleThreshold time: " << time_use << "us" << endl << endl;
	sum_time += time_use;

	cout << "Simd DoubleThresholding...." << endl;//SIMD双阙值限制
	gettimeofday(&start,NULL);
	simd_DoubleThreshold(inhibit_Output,lowthreshold,highthreshold);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "DoubleThreshold time: " << time_use << "us" << endl << endl;
	simd_time += time_use;
	write_pgm_image(inhibit_name,inhibit_Output);
	cout << "Scalar Full time is "<< sum_time/1000000 << "s" << endl;
	cout << "Simd Full time is "<< simd_time/1000000 << "s" << endl;
	cout << "Speed up : " << (1 - simd_time/sum_time)*100 << " percents" << endl;
}