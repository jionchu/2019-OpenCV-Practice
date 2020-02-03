#include "hist_func.h"

void hist_ma(Mat &input, Mat &matched, G *trans_func, float *CDF, float *CDF_ref);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat reference = imread("reference.jpg", CV_LOAD_IMAGE_COLOR);
	Mat matched_YUV;

	cvtColor(input, matched_YUV, CV_RGB2YUV);	// RGB -> YUV
	cvtColor(reference, reference, CV_RGB2YUV);

												// split each channel(Y, U, V)
	Mat channels[3];
	split(matched_YUV, channels);
	Mat Y = channels[0];						// U = channels[1], V = channels[2]

	Mat channels_ref[3];
	split(reference, channels_ref);
	Mat Y_ref = channels_ref[0];
												// PDF or transfer function txt files
	FILE *f_matched_PDF_YUV, *f_PDF_RGB;
	FILE *f_trans_func_ma_YUV;

	float **PDF_RGB = cal_PDF_RGB(input);		// PDF of Input image(RGB) : [L][3]
	float *CDF_YUV = cal_CDF(Y);				// CDF of Y channel image
	float *CDF_YUV_ref = cal_CDF(Y_ref);

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
	fopen_s(&f_matched_PDF_YUV, "matched_PDF_YUV.txt", "w+");
	fopen_s(&f_trans_func_ma_YUV, "trans_func_ma_YUV.txt", "w+");

	G trans_func_ma_YUV[L] = { 0 };			// transfer function

											// histogram matching on Y channel
	hist_ma(Y, Y, trans_func_ma_YUV, CDF_YUV, CDF_YUV_ref);

	// merge Y, U, V channels
	merge(channels, 3, matched_YUV);

	// YUV -> RGB (use "CV_YUV2RGB" flag)
	cvtColor(matched_YUV, matched_YUV, CV_YUV2RGB);

	// matched PDF (YUV)
	float *matched_PDF_YUV = cal_PDF(matched_YUV);

	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF_RGB, "%d\t%f\n", i, PDF_RGB[i]);
		fprintf(f_matched_PDF_YUV, "%d\t%f\n", i, matched_PDF_YUV[i]);

		// write transfer functions
		fprintf(f_trans_func_ma_YUV, "%d\t%d\n", i, trans_func_ma_YUV[i]);
	}

	// memory release
	free(PDF_RGB);
	free(CDF_YUV);
	fclose(f_PDF_RGB);
	fclose(f_matched_PDF_YUV);
	fclose(f_trans_func_ma_YUV);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Matched_YUV", WINDOW_AUTOSIZE);
	imshow("Matched_YUV", matched_YUV);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram matching
void hist_ma(Mat &input, Mat &matched, G *trans_func, float *CDF, float *CDF_ref) {
	G trans_func_T[L] = { 0 };
	// compute transfer function s=T(r)
	for (int i = 0; i < L; i++)
		trans_func_T[i] = (G)((L - 1) * CDF[i]);

	G trans_func_G[L] = { 0 };
	// compute trnasfer function s=G(z)
	for (int i = 0; i < L; i++) {
		trans_func_G[i] = (G)((L - 1)*CDF_ref[i]);
	}

	G trans_func_G_rev[L] = { 0 };
	int list[L] = { 0 };
	int j = 0;
	//compute reverse of z=G^(-1)(s)
	for (int i = 0; i < L; i++) {
		if (trans_func_G_rev[trans_func_G[i]] == NULL) {
			trans_func_G_rev[trans_func_G[i]] = (G)i;
			list[j] = trans_func_G[i];
			j++;
		}
	}
	int index = 0;
	while (index <= L) {
		if (trans_func_G_rev[index] == NULL) {
			if (trans_func_G_rev[index - 1] == 255)
				trans_func_G_rev[index] = 255;
			else
				trans_func_G_rev[index] = trans_func_G_rev[index - 1] + 1;
		}
		index++;
	}

	for (int i = 0; i < L; i++)
		trans_func[i] = trans_func_G_rev[trans_func_T[i]];

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			matched.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}