#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s);
Mat Inversefilter(const Mat input, int n, double sigma_t, double sigma_s, double d);
Mat FourierTransform(const Mat input, int m, int n, bool inverse);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	//Gaussian smoothing parameters
	int window_radius = 7;
	double sigma_t = 5.0;
	double sigma_s = 5.0;

	//AWGN noise variance
	double noise_var = 0.03;

	//Deconvolution threshold
	double decon_thres = 0.1;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale

	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);	// 8-bit unsigned char -> 64-bit floating point

	Mat h_f = Gaussianfilter(input_gray, window_radius, sigma_t, sigma_s);	// h(x,y) * f(x,y)
	Mat g = Add_Gaussian_noise(h_f, 0, noise_var);		//					+ n(x, y)

	Mat F = Inversefilter(g, window_radius, sigma_t, sigma_s, decon_thres);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("Gaussian Noise", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise", g);

	namedWindow("Deconvolution result", WINDOW_AUTOSIZE);
	imshow("Deconvolution result", F);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

	add(input, NoiseArr, NoiseArr);

	return NoiseArr;
}

Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s) {

	int row = input.rows;
	int col = input.cols;
	float kernelvalue;

	// generate gaussian kernel
	Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);

	Mat output = Mat::zeros(row, col, input.type());

	// convolution with zero padding
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1 = 0.0;
			for (int x = -n; x <= n; x++) { // for each kernel window
				for (int y = -n; y <= n; y++) {
					kernelvalue = kernel.at<float>(x + n, y + n);
					// Gaussian filter with Zero-paddle boundary process:
					if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
						sum1 += kernelvalue*(float)input.at<double>(i + x, j + y);
					}
				}
			}
			output.at<double>(i, j) = (double)sum1;
		}
	}

	return output;
}

Mat Inversefilter(const Mat input, int n, double sigma_t, double sigma_s, double d) {

	int row = input.rows;
	int col = input.cols;

	// generate gaussian kernel
	Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);

	// Perform Fourier Transform on Noise Image(G) and Gaussian Kernel(H)
	Mat G = FourierTransform(input, row, col, false);
	Mat H = FourierTransform(kernel, row, col, false);

	Mat F = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (abs(H.at<double>(i, j)) >= d) {
				// Element-wise divide for compute F (F = G / H)
				F.at<double>(i, j) = G.at<double>(i, j) / H.at<double>(i, j);
			}
			else
				F.at<double>(i, j) = G.at<double>(i, j);
		}
	}

	// Fill the code to perform Inverse Fourier Transform
	F = FourierTransform(F, row, col, true);

	return F;
}

Mat FourierTransform(const Mat input, int m, int n, bool inverse) {

	//expand input image to optimal size
	Mat padded;
	copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat Transformed;

	// Applying DFT
	if (!inverse) {
		dft(padded, Transformed, DFT_COMPLEX_OUTPUT);
	}
	// Reconstructing original image from the DFT coefficients
	else {
		idft(padded, Transformed, DFT_SCALE | DFT_REAL_OUTPUT);
		normalize(Transformed, Transformed, 0, 1, CV_MINMAX);
	}

	return Transformed;
}

Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize) {

	int kernel_size = (2 * n + 1);
	float denom;

	// Initialiazing Gaussian Kernel Matrix
	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	// if "normalize" is true
	// return normalized Guassian Kernel
	// else, return unnormalized one
	if (normalize) {
		kernel.convertTo(kernel, CV_32F, 1.0 / 255);
	}
	denom = 0.0;
	for (int x = -n; x <= n; x++) {  // Denominator in m(s,t)
		for (int y = -n; y <= n; y++) {
			float value1 = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<double>(x + n, y + n) = value1;
			denom += value1;
		}
	}
			for (int x = -n; x <= n; x++) {  // Denominator in m(s,t)
		for (int y = -n; y <= n; y++) {
			kernel.at<double>(x + n, y + n) /= denom;
		}
	}
	return kernel;
}