#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	double alpha = 0.5; double beta; double input;
	Mat src1, src2, dst;

	cout << " Simple Linear Blender " << endl;
	cout << "-----------------------" << endl;
	cout << "+ Enter alpha [0.0-1.0]:";
	cin >> input;

	// We use the alpha provided by the user if it is between 0 and 1
	if (input >= 0 && input <= 1) {
		alpha = input;
	}
	// Read each image
	src1 = imread("LinuxLogo.jpg");
	src2 = imread("WindowsLogo.jpg");

	// Check for invalid input
	if (src1.empty()) { cout << "Error loading src1" << endl; return -1; }
	if (src2.empty()) { cout << "Error loading src2" << endl; return -1; }

	// linear blending
	beta = (1.0 - alpha);
	addWeighted(src1, alpha, src2, beta, 0.0, dst);

	namedWindow("Linear Blend");
	imshow("Linear Blend", dst);

	waitKey(0);

	return 0;
}
