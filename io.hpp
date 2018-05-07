#ifndef IO_HPP
#define IO_HPP
#include <iostream>
#include <stdlib.h>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
#include <core/core.hpp>
#include <imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
Mat read_pgm_image(string name);
void write_pgm_image(string filename,Mat photo);
int Conervt2Gray(Mat &img,Mat &imgGray);
#endif