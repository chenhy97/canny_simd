#include <iostream>
#include "io.hpp"
#include "canny.hpp"
#include <sys/time.h>
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
	int lowthreshold,highthreshold;
	cin >> lowthreshold >> highthreshold;
	Mat grayImg;
	if(Conervt2Gray(img,grayImg) == 0){//处理读入灰度图像的情况
		grayImg = img;
	}
	Mat GaussIMG = img;

	cout << "Gauss Filtering..." << endl;//高斯滤波
	gettimeofday(&start,NULL);
	GaussFilter(grayImg,GaussIMG,5,1);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
	write_pgm_image(save_name,GaussIMG);
	cout << "Ending Gauss Filtering" << endl;
	cout << "GaussFilter time:" << time_use << "ms"<< endl;

	Mat SobelX = GaussIMG;
	Mat SobelY = GaussIMG;
	Mat SobelXY = GaussIMG;
	string X_name = save_name + "X";
	string Y_name = save_name + "Y";
	string XY_name = save_name + "XY";
	string inhibit_name = save_name + "_inhibit";
	string h_name = save_name + "_hyst";
	double *pointDirection;
	pointDirection = new double[(GaussIMG.rows - 1) *(GaussIMG.cols - 1)];

	cout << "Sobel Grad Getting ...." << endl;//sobel算子求梯度
	gettimeofday(&start,NULL);
	SobelGradDirection(GaussIMG,SobelX,SobelY,pointDirection);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "Ending Grad Getting" << endl;
	cout << "Grad Getting time: " << time_use << "ms" << endl;

	cout << "Sobel Amplitude Getting ...." << endl;//sobel算子求幅值
	gettimeofday(&start,NULL);
	SobelAmplitude(SobelX,SobelY,SobelXY);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "Ending Amplitude Getting" << endl;
	cout << "Amplitude Getting time: " << time_use << "ms" << endl;

	
	Mat inhibit_Output = GaussIMG;

	cout << "Inhibit local Max ...." << endl;//局部非极大值抑制
	gettimeofday(&start,NULL);
	inhibit_local_Max(SobelXY,inhibit_Output,pointDirection);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "Ending Inhibit local Max" << endl;
	cout << "Inhibit local Max time: " << time_use << "ms" << endl;

	cout << "DoubleThresholding...." << endl;//双阙值限制
	gettimeofday(&start,NULL);
	DoubleThreshold(inhibit_Output,lowthreshold,highthreshold);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "Ending DoubleThreshold" << endl;
	cout << "DoubleThreshold time: " << time_use << "ms" << endl;

	
	Mat hysetersis_Img = inhibit_Output;

	cout << "hysetersising...." << endl;//滞后优化
	gettimeofday(&start,NULL);
	DoubleThreshold(inhibit_Output,lowthreshold,highthreshold);
	gettimeofday(&end,NULL);
	time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	cout << "Ending hysetersis" << endl;
	cout << "hysetersis time: " << time_use << "ms" << endl;

	write_pgm_image(inhibit_name,inhibit_Output);
	write_pgm_image(h_name,hysetersis_Img);
}