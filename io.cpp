#include <iostream>
#include <stdlib.h>
#include "io.hpp"
using namespace std;
using namespace cv;
Mat read_pgm_image(string name){
	Mat img = imread(name);
	return img;
}
void write_pgm_image(string filename,Mat photo){
	imwrite(filename+".jpg",photo);//需要扩展名
}
int Conervt2Gray(Mat &img,Mat &imgGray){//一定要取地址，否则是另开副本，无法真正修改！
	if(!img.data || img.channels()!=3){
		return 0;
	}
	imgGray = Mat::zeros(img.size(),CV_8UC1);
	uchar *ptrimg = img.data;//转MAT为可操作数据
	uchar *ptrimgGray = imgGray.data;
	int imgstep = img.step;
	int imgGraystep = imgGray.step;
	for(int i = 0;i < imgGray.rows;i ++){
		for(int j = 0;j < imgGray.cols;j ++){
			ptrimgGray[i * imgGraystep + j] = 0.114 * ptrimg[i * imgstep + 3*j] + 0.587*ptrimg[i * imgstep + 3*j + 1] + 0.299*ptrimg[i * imgstep + 3*j+2];
		}
	}
	return 1;
}
