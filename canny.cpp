#include "canny.hpp"
#include "sse_mathfun_extension.h"
#include <sys/time.h>
//******导入exp函数********
__m128 FastExpSse (__m128 x)//clang 没有SVLM，也就没有处理向量数学的函数，即没有exp
{
    __m128 a = _mm_set1_ps (12102203.0f); /* (1 << 23) / log(2) */
    __m128i b = _mm_set1_epi32 (127 * (1 << 23) - 298765);
    __m128i t = _mm_add_epi32 (_mm_cvtps_epi32 (_mm_mul_ps (a, x)), b);
    return _mm_castsi128_ps (t);
}
__m128 abs_vec(__m128 x){
  // with clang, this turns into a 16B load,
  // with every calling function getting its own copy of the mask
  	__m128 a = _mm_set_ps1(-0.0);
  	x = _mm_andnot_ps(a,x);
  	return x;
}
//
//***********无法完全进行向量化************瓶颈所在！！！！
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
	//这里可以通过调用函数来实现向量化。
	for(int i = 0;i < ksize;i ++){
		float *data = kernel.ptr<float>(i);
		for(int j = 0;j < ksize;j ++){
			data[j] = data[j]/sum;
		}
	}
	return kernel;
}
Mat simd_to_create(int ksize,float sigma){
	Mat kernel = Mat::zeros(ksize,ksize,CV_32FC1);
	float center = ksize/2;
	float double_sigma = (2*sigma*sigma);
	float PI_sigma = 1/(2*PI*sigma*sigma);
	float sum = 0;//32位
	__attribute__((aligned(16))) float temp[4] = {0,1,2,3};//16位内存对齐
	__m128 X,Y,Z;
	for(int i = 0;i < ksize;i ++){
		/*__attribute__((aligned(16)))*/ float *data = kernel.ptr<float>(i);//16位内存对齐
		int j;
		for(j = 0;j + 4 < ksize;j += 4){//128位专用寄存器，一次性可以处理4组32位变量的运算,4*32 
			X = _mm_load_ps(&temp[0] + j); // 将x加载到X（由于128位可以存放四个32位数据，所以默认一次加载连续的4个参数）
			float temp_data = j - center;
			Z = _mm_set_ps1(temp_data);
			Y = _mm_add_ps(X,Z);
			Y = _mm_mul_ps(Y,Y);
			float temp_i = i - center;
			X = _mm_set_ps1(temp_i);
			X = _mm_mul_ps(X,X);
			Y = _mm_add_ps(Y,X);
			X = _mm_set_ps1(-1);
			Y = _mm_mul_ps(Y,X);
			X = _mm_set_ps1(double_sigma);
			Y = _mm_div_ps(Y,X);//wait to exp(Y)
			Y = FastExpSse(Y);
			X = _mm_set_ps1(PI_sigma);
			Y = _mm_mul_ps(X,Y);
			_mm_storeu_ps(data + j,Y);
			sum = sum + data[j] + data[j + 1] + data[j + 2] + data[j + 3];
		}
		for(;j < ksize;j ++){
			float temp_abc = ((-1)*((i - center)*(i - center)+(j - center)*(j - center)))/double_sigma;
			data[j] = PI_sigma*exp(temp_abc);
			sum = sum + data[j];
		}
	}
	Y = _mm_set_ps1(sum);
	for(int i = 0;i < ksize;i ++){
		/*__attribute__((aligned(16)))*/ float *data =kernel.ptr<float>(i);
		for(int j = 0;j + 4 < ksize;j += 4){
			X = _mm_loadu_ps(data + j);
			X = _mm_div_ps(X,Y);
			_mm_storeu_ps(data + j,X);
		}
		data[4] = data[4]/sum;
	}
	return kernel;
}
void GaussFilter(const Mat src,Mat &dst,int ksize,float sigma){
	//struct timeval start;
	//struct timeval end;	
	//float time_use = 0;
	//gettimeofday(&start,NULL);
	Mat kernel = createGaussianKernel2D(ksize,sigma);
	//gettimeofday(&end,NULL);
	//time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	//cout <<"no test :" << time_use << endl;
	//gettimeofday(&start,NULL);
	//Mat kernel = simd_to_create(ksize,sigma);
	//gettimeofday(&end,NULL);
	//time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	//cout << "test2:" << time_use << endl;
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
					int row = i + l;
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
void simd_GaussFilter(const Mat src,Mat &dst,int ksize,float sigma){
	Mat kernel = simd_to_create(ksize,sigma);
	int height = src.rows;
	int width = src.cols;
	int type_src = src.type();
	dst = Mat::zeros(dst.size(),CV_8UC1);
	if(!src.data || src.channels() != 1){
		return;
	}
	float temparray[100];
	for(int i = 0;i < ksize*ksize;i ++){
		temparray[i] = 0;//初始化处理数组。
	}
	__m128 X,Y,Z;
	int index = 0;
	for(int i = 0;i < ksize;i ++){
		float* data = kernel.ptr<float>(i);
		int j = 0;
		for(j = 0;j + 4 < ksize;j = j + 4){
			X = _mm_loadu_ps(data + j);
			_mm_storeu_ps(temparray + index , X);
		}
		index = index + 4;
		for(;j < ksize;j ++){
			temparray[index] = data[j];
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
					int row = i + l;
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
void SobelGradDirection(const Mat src,Mat &sobelx,Mat &sobely,float *pointDirection){
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
	long long k = 0;
	int i,j;
	for(i = 1;i < (src.rows - 1);i ++){
		for(j = 1;j <(src.cols - 1);j ++){
			float gradY = P[(i-1)*step + j + 1] + P[i*step + j + 1]*2 + P[(i + 1)*step + j + 1] -  P[(i-1)*step + j - 1] - P[i*step + j - 1]*2 - P[(i + 1)*step + j - 1];
			py[i * stepXY + j * (stepXY/step)] = abs(gradY);
			
			float gradX = P[(i+1)*step + j - 1] + P[(i + 1) * step + j] * 2+P[( i + 1) * step+ j + 1]-P[(i-1)*step+j-1]-P[(i-1)*step+j]*2-P[(i-1)*step+j+1];
			px[i * stepXY + j * (stepXY/step)] = abs(gradX);
			if(gradX == 0){
				gradX = 0.0000000001;//防止除法为0
			}
			pointDirection[k] = atan(gradY/gradX)*57.3;
			pointDirection[k] += 90;
			k ++;
		}
	}
	cout << py[1 * stepXY + 2 * (stepXY/step)] << endl;
	convertScaleAbs(sobelx,sobelx);//对于每个输入数组的元素函数convertScaleAbs 进行三次操作依次是：缩放，得到一个绝对值，转换成无符号8位类型
	convertScaleAbs(sobely,sobely);
}
////isn't finshed and not correct,but the thought may be right,but this way may not fast the program
void simd_SobelGradDirection(const Mat src,Mat &sobelx,Mat &sobely,float *pointDirection){
	for(int i =0;i < src.rows - 1;i ++){
		pointDirection[i] = 0;
	}
	sobelx = Mat::zeros(src.size(),CV_32SC1);
	sobely = Mat::zeros(src.size(),CV_32SC1);
	uchar* P = src.data;
	uchar* px = sobelx.data;
	uchar* py = sobely.data;
	float temp[20];
	float temp1[20];
	uchar for_px[512];//can't be the point ,all can not use the point type change
	uchar for_py[512];
	int step = src.step;
	int stepXY = sobelx.step;
	long long k = 0;
	int i,j;
	__m128i X,Y;

	__m128 floats1,floats2,floats3,floats4,sub_min,mask,mask_low1;
	mask_low1 = _mm_set1_ps(1);
	for(i = 1;i < (src.rows - 1);i ++){
		for(j = 1;j + 4 <(src.cols - 1);j = j + 4){
			X = _mm_loadu_si128((__m128i*)(P + ((i - 1) * step + j + 1)));
			Y = _mm_loadu_si128((__m128i*)(P + i * step + j + 1));
			X = _mm_cvtepu8_epi32(X);//8->32 整形。
			Y = _mm_cvtepu8_epi32(Y);
			floats1 = _mm_cvtepi32_ps(X);//整型-> floats
			floats2 = _mm_cvtepi32_ps(Y);
			floats2 = _mm_add_ps(floats2,floats2);
			floats1 = _mm_add_ps(floats1,floats2);
			X = _mm_loadu_si128((__m128i*)(P + ((i + 1) * step + j + 1)));
			X = _mm_cvtepu8_epi32(X);
			floats2 = _mm_cvtepi32_ps(X);
			floats1 = _mm_add_ps(floats1,floats2);
			X = _mm_loadu_si128((__m128i*)(P + ((i - 1) * step + j - 1)));
			X = _mm_cvtepu8_epi32(X);
			floats2 = _mm_cvtepi32_ps(X);
			floats1 = _mm_sub_ps(floats1,floats2);
			X = _mm_loadu_si128((__m128i*)(P + (i * step + j - 1)));
			X = _mm_cvtepu8_epi32(X);
			floats2 = _mm_cvtepi32_ps(X);
			floats2 = _mm_add_ps(floats2,floats2);
			floats1 = _mm_sub_ps(floats1,floats2);
			X = _mm_loadu_si128((__m128i*)(P + ((i + 1) * step + j - 1)));
			X = _mm_cvtepu8_epi32(X);
			floats2 = _mm_cvtepi32_ps(X);
			floats1 = _mm_sub_ps(floats1,floats2);
			floats3 = floats1;
			//_mm_storeu_ps(temp,floats1);
			//
			//
			floats1 = abs_vec(floats1);
			X = _mm_cvtps_epi32(floats1);    // Convert them to 32-bit ints
   			X = _mm_packus_epi32(X, X);        // Pack down to 16 bits
    		X = _mm_packus_epi16(X, X);        // Pack down to 8 bits
    		*(int *)for_py = _mm_cvtsi128_si32(X); // Store the lower 32 bits
    		py[i * stepXY + j * (stepXY/step)] = for_py[0];
    		py[i * stepXY + (j + 1) * (stepXY/step)] = for_py[1];
    		py[i * stepXY + (j + 2) * (stepXY/step)] = for_py[2];
    		py[i * stepXY + (j + 3) * (stepXY/step)] = for_py[3];
			X = _mm_loadu_si128((__m128i*)(P + (i + 1) * step + j - 1));
			Y = _mm_loadu_si128((__m128i*)(P + (i + 1) * step + j ));
			X = _mm_cvtepu8_epi32(X);//8->32 整形。
			Y = _mm_cvtepu8_epi32(Y);
			floats1 = _mm_cvtepi32_ps(X);//整型-> floats
			floats2 = _mm_cvtepi32_ps(Y);
			floats2 = _mm_add_ps(floats2,floats2);
			floats1 = _mm_add_ps(floats1,floats2);
			X = _mm_loadu_si128((__m128i*)(P + ((i + 1) * step + j + 1)));
			X = _mm_cvtepu8_epi32(X);
			floats2 = _mm_cvtepi32_ps(X);
			floats1 = _mm_add_ps(floats1,floats2);
			X = _mm_loadu_si128((__m128i*)(P + ((i - 1) * step + j - 1)));
			X = _mm_cvtepu8_epi32(X);
			floats2 = _mm_cvtepi32_ps(X);
			floats1 = _mm_sub_ps(floats1,floats2);
			X = _mm_loadu_si128((__m128i*)(P + ((i - 1) * step + j)));
			X = _mm_cvtepu8_epi32(X);
			floats2 = _mm_cvtepi32_ps(X);
			floats2 = _mm_add_ps(floats2,floats2);
			floats1 = _mm_sub_ps(floats1,floats2);
			X = _mm_loadu_si128((__m128i*)(P + ((i - 1) * step + j + 1)));
			X = _mm_cvtepu8_epi32(X);
			floats2 = _mm_cvtepi32_ps(X);
			floats1 = _mm_sub_ps(floats1,floats2);
			floats4 = floats1;
			floats1 = abs_vec(floats1);
			//_mm_storeu_ps(temp1,floats4);
			X = _mm_cvtps_epi32(floats1);    // Convert them to 32-bit ints
   			X = _mm_packus_epi32(X, X);        // Pack down to 16 bits
    		X = _mm_packus_epi16(X, X);        // Pack down to 8 bits
    		*(int *)for_px = _mm_cvtsi128_si32(X); // Store the lower 32 bits
			px[i * stepXY + j * (stepXY/step)] = for_px[0];
    		px[i * stepXY + (j + 1) * (stepXY/step)] = for_px[1];
    		px[i * stepXY + (j + 2) * (stepXY/step)] = for_px[2];
    		px[i * stepXY + (j + 3) * (stepXY/step)] = for_px[3];
    		sub_min = _mm_sub_ps(floats4,_mm_setzero_ps());
    		mask = _mm_cmpeq_ps(sub_min,_mm_setzero_ps());
    		mask = _mm_and_ps(mask,mask_low1);//判断gradX是否为0，为0的话，最低位改为1
    		floats4 = _mm_or_ps(mask,floats4);
    		floats3 = _mm_div_ps(floats3,floats4);
    		floats3 = atan_ps(floats3);
    		floats4 = _mm_set1_ps(57.3);
    		floats3 = _mm_mul_ps(floats3,floats4);
    		floats4 = _mm_set1_ps(90);
    		floats3 = _mm_add_ps(floats4,floats3);
    		_mm_storeu_ps(pointDirection + k,floats3);
    		k = k + 4;
		}
		for(;j < (src.cols- 1);j ++){
			float gradY = P[(i-1)*step + j + 1] + P[i*step + j + 1]*2 + P[(i + 1)*step + j + 1] -  P[(i-1)*step + j - 1] - P[i*step + j - 1]*2 - P[(i + 1)*step + j - 1];
			py[i * stepXY + j * (stepXY/step)] = abs(gradY);
			float gradX = P[(i+1)*step + j - 1] + P[(i + 1) * step + j] * 2+P[( i + 1) * step+ j + 1]-P[(i-1)*step+j-1]-P[(i-1)*step+j]*2-P[(i-1)*step+j+1];
			px[i * stepXY + j * (stepXY/step)] = abs(gradX);
			if(gradX == 0){
				gradX = 0.0000000001;//防止除法为0
			}
			pointDirection[k] = atan(gradY/gradX)*57.3;
			pointDirection[k] += 90;
			k ++;
		}
	}
	cout << py[1 * stepXY + 2 * (stepXY/step)] << endl;

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
void simd_SobelAmplitude(Mat &sobelx,Mat &sobely,Mat &SobelXY){
	SobelXY = Mat::zeros(sobelx.size(),CV_32FC1);
	__m128i X,Y;
	__m128 floats1,floats2;
	for(int i = 0;i < SobelXY.rows;i ++){
		float *data = SobelXY.ptr<float>(i);
		uchar *datax = sobelx.ptr<uchar>(i);
		uchar *datay = sobely.ptr<uchar>(i);
		int j;
		for(j = 0;j + 4 < SobelXY.cols;j = j + 4){
			X = _mm_loadu_si128((__m128i*)(datax + j));
			Y = _mm_loadu_si128((__m128i*)(datay + j));
			X = _mm_cvtepu8_epi32(X);//8->32 整形。
			Y = _mm_cvtepu8_epi32(Y);
			floats1 = _mm_cvtepi32_ps(X);//整型-> floats
			floats2 = _mm_cvtepi32_ps(Y);
			floats1 = _mm_mul_ps(floats1,floats1);
			floats2 = _mm_mul_ps(floats2,floats2);
			floats1 = _mm_add_ps(floats1,floats2);
			floats1 = _mm_sqrt_ps(floats1);
			_mm_storeu_ps(data + j,floats1);
		}
		for(;j < SobelXY.cols;j ++){
			data[j] = sqrt(datax[j] * datax[j] + datay[j]*datay[j]);

		}
	}
	convertScaleAbs(SobelXY,SobelXY);
}
void inhibit_local_Max(Mat &SobelXY,Mat &Output,float *pointDirection){
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
void DoubleThreshold(Mat &Input,uchar LowThreshod,uchar highThreshold){
	for(int i = 0;i < Input.rows;i ++){
		uchar *data = Input.ptr<uchar>(i);
		int j;
		for(int j = 0;j < Input.cols;j = j + 1){
			if(data[j] > highThreshold){
				data[j] = 255;
			}
			if(data[j] < LowThreshod){
				data[j] = 0;
			}
		}
	}
}
void simd_DoubleThreshold(Mat &Input,uchar LowThreshod,uchar highThreshold){
	__m128i loaded8,sub_min,sub_max,mask_hi,mask_lo,mask_combinedhi,mask_combinedlo;
 	for(int i = 0;i < Input.rows;i ++){
		uchar *data = Input.ptr<uchar>(i);
		int j;
		for(j = 0;j + 16 < Input.cols;j = j + 16){
			//cout << "a" << endl;
			loaded8 = _mm_loadu_si128((__m128i*)(data + j));
			//subtract 128 from every 8-bit int
 			sub_min = _mm_sub_epi8(loaded8, _mm_set1_epi8(LowThreshod));
 			sub_max = _mm_sub_epi8(loaded8, _mm_set1_epi8(highThreshold));
 			mask_hi = _mm_cmpgt_epi8(sub_max,_mm_setzero_si128());//submax>0 8位全1，否则全0
 			mask_lo = _mm_cmpgt_epi8(sub_min,_mm_setzero_si128());//submin<=0,8位全0，否则全1
 			loaded8 = _mm_and_si128(loaded8, mask_combinedlo);
 			loaded8 = _mm_or_si128(loaded8,mask_hi);
 			_mm_storeu_si128((__m128i *)(data + j), loaded8);
 			//cout << "i: " << i << "j: "<< j << endl;

 			//greater than top limit?
 		}
 		for(;j < Input.cols;j = j + 1){
			if(data[j] > highThreshold){
				data[j] = 255;
			}
			if(data[j] < LowThreshod){
				data[j] = 0;
			}
		}
		//cout << i << endl;
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