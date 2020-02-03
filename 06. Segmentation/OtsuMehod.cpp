#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#include<iostream>
#include<tuple> // for tuple
#define IM_TYPE	CV_8UC3
#define L 256		// # of intensity levels

using namespace cv;
using namespace std;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

tuple<float, Mat> otsu_gray_seg(const Mat input);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;
	float t;

	cvtColor(input, input_gray, CV_RGB2GRAY);

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	tie(t, output) = otsu_gray_seg(input_gray);

	namedWindow("Otsu", WINDOW_AUTOSIZE);
	imshow("Otsu", output);
	std::cout << t << std::endl;

	waitKey(0);

	return 0;
}

tuple<float, Mat> otsu_gray_seg(const Mat input) {

	int row = input.rows;
	int col = input.cols;
	Mat output = Mat::zeros(row, col, input.type());
	int n = row*col;
	float T = 0, var = 0, var_max = 0,mg = 0, m1 = 0, m2 = 0, q1 = 0, q2 = 0, sigma1 = 0, sigma2 = 0;
	int histogram[L] = { 0 };  // initializing histogram values
	float probability[L] = { 0 };

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {   // finding histogram of the image
			histogram[input.at<G>(i, j)]++;
		}
	}

	for (int i = 0; i < L; i++) {     //auxiliary value for computing mean value
		probability[i] = (float)histogram[i] / n;
	}
	for (int i = 0; i < L; i++) {
		mg += i*probability[i];
	}
	for (int t = 0; t < L; t++) {  //update q
		q1 = 0; q2 = 0; m1 = 0; m2 = 0; sigma1 = 0; sigma2 = 0;
		for (int k = 0; k <= t; k++) {
			q1 += probability[k];
		}
		for (int k = 0; k <= t; k++) {
			m1 += k*probability[k]/q1;
			sigma1 += k*k*probability[k] / q1;
		}
		sigma1 -= m1*m1;
		for (int k = t + 1; k < L; k++) {
			q2 += probability[k];
		}
		for (int k = t + 1; k < L; k++) {
			m2 += k*probability[k]/q2;
			sigma2 += k*k*probability[k]/q2;
		}
		sigma2 -= m2*m2;
		var = q1*pow(m1 - mg,2) + q2*pow(m2 - mg,2);
		if (var > var_max) {
			T = t; //threshold
			var_max = var;
		}
	}

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<G>(i, j) > T)
				output.at<G>(i, j) = 255;
			else
				output.at<G>(i, j) = 0;
		}
	}

	return make_tuple(T, output);
}
