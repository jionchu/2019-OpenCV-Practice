#include <iostream>
#include <opencv2/opencv.hpp>
#define IM_TYPE	CV_8UC3

using namespace cv;

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output,input_gray;

	cvtColor(input, input_gray, CV_RGB2GRAY);

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	Mat samples(input.rows * input.cols, 3, CV_32F);
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*input.rows, z) = input.at<Vec3b>(y, x)[z];

	// Clustering is performed for each channel (RGB)
	// Note that the intensity value is not normalized here (0~1). You should normalize both intensity and position when using them simultaneously.
	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	Mat new_image(input.size(), input.type());
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x*input.rows, 0);
			for (int z = 0; z < 3; z++)
				new_image.at<Vec3b>(y, x)[z] = centers.at<float>(cluster_idx,z);
		}
	imshow("clustered image", new_image);
	
	Mat samples2(input_gray.rows * input_gray.cols, 1, CV_32F);
	for (int y = 0; y < input_gray.rows; y++)
		for (int x = 0; x < input_gray.cols; x++)
			samples2.at<float>(y + x*input_gray.rows, 0) = input_gray.at<uchar>(y, x);

	Mat labels2;
	Mat centers2;

	kmeans(samples2, clusterCount, labels2, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers2);

	Mat new_image_gray(input_gray.size(), input_gray.type());
	for (int y = 0; y < input_gray.rows; y++)
		for (int x = 0; x < input_gray.cols; x++)
		{
			int cluster_idx = labels2.at<int>(y + x*input_gray.rows, 0);
			new_image_gray.at<uchar>(y, x) = (uchar)centers2.at<float>(cluster_idx, 0);
		}
	imshow("clustered image_gray", new_image_gray);

	waitKey(0);

	return 0;
}