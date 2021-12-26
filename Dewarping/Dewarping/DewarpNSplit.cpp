// Dewarping.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
//
#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include "DewarpNSplit.h"

using namespace cv;
using namespace std;

struct TwoImage
{
	unsigned char* firstPart;
	unsigned char* secondPart;
};

void showImage(const Mat& image, String imageName) {
	/*Mat clone = image.clone();
	int resizeRate = 0.3;
	Size iSize = Size(1280, 720);
	resize(clone, clone, iSize);*/
	String path = "D:\\THINH\\C++\\FSI\\Draft\\";
	imwrite(path + imageName + ".bmp", image);
};


int findContourIndex(vector<Point> contour, vector<vector<Point>>& contours) {
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() == contour.size()) {
			if (equal(contours[i].begin(), contours[i].end(), contour.begin(), contour.end()))
				return i;
		}
	}
	return -1;
};

int findMinAreaIdx(vector<vector<Point>>& approxs) {
	int minIdx = -1;
	int minArea = INT_MAX;
	for (size_t i = 0; i < approxs.size(); i++)
	{
		double currentArea = contourArea(approxs[i]);
		if (currentArea <= minArea) {
			minArea = currentArea;
			minIdx = i;
		}
	}

	return minIdx;
};

int sumOfPoint(const Point& a) {
	return a.x + a.y;
};

int diffOfPoint(const Point& a) {
	return a.y - a.x;
};

float euclideanDist2f(Point2f& a, Point2f& b) {
	return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
}

float getRealAspectRatio(int imageWidth, int imageHeight, vector<Point2f>& approx2f) {
	double u0 = imageWidth / 2;
	double v0 = imageHeight / 2;

	double m1x = (double)approx2f[0].x - u0;
	double m1y = (double)approx2f[0].y - v0;
	double m2x = (double)approx2f[1].x - u0;
	double m2y = (double)approx2f[1].y - v0;
	double m3x = (double)approx2f[3].x - u0;
	double m3y = (double)approx2f[3].y - v0;
	double m4x = (double)approx2f[2].x - u0;
	double m4y = (double)approx2f[2].y - v0;

	double k2 = ((m1y - m4y) * m3x - (m1x - m4x) * m3y + m1x * m4y - m1y * m4x) /
		((m2y - m4y) * m3x - (m2x - m4x) * m3y + m2x * m4y - m2y * m4x);

	double k3 = ((m1y - m4y) * m2x - (m1x - m4x) * m2y + m1x * m4y - m1y * m4x) /
		((m3y - m4y) * m2x - (m3x - m4x) * m2y + m3x * m4y - m3y * m4x);

	double f_squared =
		-((k3 * m3y - m1y) * (k2 * m2y - m1y) + (k3 * m3x - m1x) * (k2 * m2x - m1x)) /
		((k3 - 1) * (k2 - 1));

	double whRatio = sqrt(
		(pow((k2 - 1), 2) + pow((k2 * m2y - m1y), 2) / f_squared + pow((k2 * m2x - m1x), 2) / f_squared) /
		(pow((k3 - 1), 2) + pow((k3 * m3y - m1y), 2) / f_squared + pow((k3 * m3x - m1x), 2) / f_squared)
	);

	if (k2 == 1 && k3 == 1) {
		whRatio = sqrt(
			(pow((m2y - m1y), 2) + pow((m2x - m1x), 2)) /
			(pow((m3y - m1y), 2) + pow((m3x - m1x), 2)));
	}

	return (float)(whRatio);
}


// comparison function object
bool compareContourAreas(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
	double i = fabs(contourArea(cv::Mat(contour1)));

	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i > j);
};

bool compareContourPosition(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
	Rect i = boundingRect(contour1);
	Rect j = boundingRect(contour2);
	return (i.x < j.x);
};

bool compareContourTopLeft(const Rect& i, const Rect& j) {
	int tliX = i.tl().x == 0 ? 1 : i.tl().x;
	int tliY = i.tl().y == 0 ? 1 : i.tl().y;
	int tljX = j.tl().x == 0 ? 1 : j.tl().x;
	int tljY = j.tl().y == 0 ? 1 : j.tl().y;
	return (tliX * tliY) < (tljX * tljY);
};
// Cac contours co the bi trung` nhau/xe lech. nhau 1 ti' nhung van la 1 contour. Ham` kiem tra xem contour
// da ton` tai trong mang? approxs chua
bool checkIfContourAvailable(vector<vector<Point>>& approxs, vector<Point>& currApprox) {
	Rect a = boundingRect(currApprox);
	for (size_t i = 0; i < approxs.size(); i++)
	{
		Rect b = boundingRect(approxs[i]);
		if ((abs(sumOfPoint(b.tl()) - sumOfPoint(a.tl())) < 10) && (abs(sumOfPoint(b.br()) - sumOfPoint(a.br())) < 10))
			return true;
	}
	return false;
};

bool checkIfSumOfPointEqual(const vector<Point>& points, const size_t& indx, const vector<int>& sum) {
	for (size_t i = 0; i < sum.size(); i++)
	{
		if (abs(sumOfPoint(points[indx]) - sumOfPoint(points[sum[i]])) < 10)
			return true;
	}
	return false;
};

bool isContain(int checkNumber, const vector<int>& vector) {
	for (size_t i = 0; i < vector.size(); i++)
	{
		if (vector[i] == checkNumber)
			return true;
	}
	return false;
}

int findNonYellowPage(vector<Mat>& splitted) {
	Mat first = splitted[0].clone();
	//Mat last = splitted[splitted.size() - 1];

	cvtColor(first, first, COLOR_BGR2HSV);
	//cvtColor(last, last, COLOR_BGR2HSV);
	Mat mask;
	//inRange(first, Scalar(20, 95, 20), Scalar(30, 225, 255), mask);
	inRange(first, Scalar(20, 90, 80), Scalar(40, 225, 255), mask);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(mask, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contours.begin(), contours.end(), compareContourAreas);

	// Clear cac anh va bien khong su dung
	hierarchy.clear();
	hierarchy.shrink_to_fit();

	if (contours.size() > 0) {
		Rect rect = boundingRect(contours[0]);

		if (((float)rect.area() / (float)(first.rows * first.cols)) < 0.9) {
			contours.clear();
			contours.shrink_to_fit();
			mask.release();
			first.release();
			return 0;
		}
		else {
			contours.clear();
			contours.shrink_to_fit();
			mask.release();
			first.release();
			return splitted.size() - 1;
		}
	}
	else {
		contours.clear();
		contours.shrink_to_fit();
		mask.release();
		first.release();
		return 0;
	}
}


vector<vector<Point>> findMostCommonAreaIdx(vector<vector<Point>>& approxs) {
	int iterate = 0;
	vector<vector<Point>> listCommon;
	for (size_t i = 0; i < approxs.size(); i++)
	{
		if ((i + 1) >= approxs.size()) {
			if (iterate > 0) {
				listCommon.push_back(approxs[i]);
				return listCommon;
			}
			break;
		}
		double currArea = contourArea(approxs[i]);
		double nextArea = contourArea(approxs[i + 1]);
		if (nextArea / currArea > 0.80) {
			iterate++;
			listCommon.push_back(approxs[i]);
		}
		else
		{
			if (iterate > 0) {
				listCommon.push_back(approxs[i]);
				return listCommon;
			}
		}
	}
	return listCommon;
}

vector<int> findMaxMinPointSumIndx(const vector<Point>& points) {
	int max = 0;
	int maxIdx = 0;
	int min = 999999;
	int minIdx = 0;
	for (size_t i = 0; i < points.size(); i++)
	{
		if (sumOfPoint(points[i]) > max) {
			max = sumOfPoint(points[i]);
			maxIdx = i;
		}

		if (sumOfPoint(points[i]) < min) {
			min = sumOfPoint(points[i]);
			minIdx = i;
		}
	}
	vector<int> result = { maxIdx, minIdx };
	return result;
};


vector<int> findMaxMinPointDiffIndx(const vector<Point>& points, const vector<int>& sum = {}) {
	int max = -99999;
	int maxIdx = 0;
	int min = 999999;
	int minIdx = 0;
	for (size_t i = 0; i < points.size(); i++)
	{
		if (sum.size() > 0) {
			if (std::find(sum.begin(), sum.end(), i) != sum.end())
				continue;
		}
		if (checkIfSumOfPointEqual(points, i, sum)) {
			continue;
		}

		if (diffOfPoint(points[i]) > max) {
			max = diffOfPoint(points[i]);
			maxIdx = i;
		}

		if (diffOfPoint(points[i]) < min) {
			min = diffOfPoint(points[i]);
			minIdx = i;
		}
	}
	vector<int> result = { maxIdx, minIdx };
	return result;
};


// Ham xac' dinh 4 diem cua 4 dinh? cua van ban chung ta dang dewarp. Neu contour chua 12 diem? thi` chuan? hoa' 
// contour ay' ve 4 diem? roi xac dinh cac dinh?
vector<Point2f> rectify(const vector<Point>& approx, int org_width, int org_height) {
	float curWidth = 1500.0;
	float curHeight = 880.0;

	float widthRatio = (float)org_width / curWidth;
	float heightRatio = (float)org_height / curHeight;

	vector<Point2f> result = { Point2f(), Point2f() ,Point2f() ,Point2f() };
	if (approx.size() == 4) {
		vector<int> sum = findMaxMinPointSumIndx(approx);
		vector<int> diff = findMaxMinPointDiffIndx(approx);
		result[0] = Point2f((float)approx[sum[1]].x * widthRatio, (float)approx[sum[1]].y * heightRatio);
		result[2] = Point2f((float)approx[sum[0]].x * widthRatio, (float)approx[sum[0]].y * heightRatio);
		result[1] = Point2f((float)approx[diff[1]].x * widthRatio, (float)approx[diff[1]].y * heightRatio);
		result[3] = Point2f((float)approx[diff[0]].x * widthRatio, (float)approx[diff[0]].y * heightRatio);

		sum.clear();
		sum.shrink_to_fit();
		diff.clear();
		diff.shrink_to_fit();
	}
	else {
		//vector<Point2f> result = { Point2f(), Point2f() ,Point2f() ,Point2f() };
		vector<int> sum = findMaxMinPointSumIndx(approx);
		vector<int> diff = findMaxMinPointDiffIndx(approx, sum);
		result[0] = Point2f((float)approx[sum[1]].x * widthRatio, (float)approx[sum[1]].y * heightRatio);
		result[2] = Point2f((float)approx[sum[0]].x * widthRatio, (float)approx[sum[0]].y * heightRatio);
		result[1] = Point2f((float)approx[diff[1]].x * widthRatio, (float)approx[diff[1]].y * heightRatio);
		result[3] = Point2f((float)approx[diff[0]].x * widthRatio, (float)approx[diff[0]].y * heightRatio);

		sum.clear();
		sum.shrink_to_fit();
		diff.clear();
		diff.shrink_to_fit();
	}
	return result;
};

void RotateImage(Mat& image) {
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	//cvtColor(last, last, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, Scalar(20, 90, 80), Scalar(40, 225, 255), mask);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Rect> rects;

	//showImage(mask, "mask");

	findContours(mask, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contours.begin(), contours.end(), compareContourAreas);

	// Clear cac anh va bien khong su dung
	hierarchy.clear();
	hierarchy.shrink_to_fit();

	// Bien count de gioi han lay 10 contours
	int count = 0;

	// Duyet mang contour
	for (size_t i = 0; i < contours.size(); i++) {
		/*	vector<Point> approx;
			double p = arcLength(contours[i], true);
			approxPolyDP(contours[i], approx, 0.1 * p, true);


	#pragma region Debug
			if (1 == 1) {
				Mat imgCln = mask.clone();
				drawContours(imgCln, contours, i, Scalar(0, 255, 0), -1);
				for (size_t k = 0; k < approx.size(); k++)
				{
					circle(imgCln, approx[k], 2, Scalar(0, 0, 255), -1);
				}
				String path = "D:\\THINH\\C++\\FSI\\Draft\\contours\\";
				imwrite(path + "cntHSV" + to_string(i) + ".bmp", imgCln);
				imgCln.release();
			}
	#pragma endregion
			approx.clear();
			approx.shrink_to_fit();*/

		rects.push_back(boundingRect(contours[i]));

		if (count == 1)
			break;
		count++;
	}

	contours.clear();
	contours.shrink_to_fit();
	sort(rects.begin(), rects.end(), compareContourTopLeft);

	if (rects.size() > 1) {
		if ((rects[1].y - rects[0].br().y) > 0)
			//// So sanh khoang cach tu dau trang toi tl cua rect[0] voi k/c tu br cua rect[1] toi het anh de suy ra anh bi quay phia nao
			//if (rects[0].y < (image.size().height - rects[1].br().y))
			//	rotate(image, image, ROTATE_90_CLOCKWISE);
			//else
			//	rotate(image, image, ROTATE_90_COUNTERCLOCKWISE);

			rotate(image, image, ROTATE_90_COUNTERCLOCKWISE);

		else if (((float)rects[0].br().y - (float)rects[1].y) / rects[0].height < 0.5)
		{
			//// So sanh khoang cach tu dau trang toi tl cua rect[0] voi k/c tu br cua rect[1] toi het anh de suy ra anh bi quay phia nao
			//if (rects[0].y < (image.size().height - rects[1].br().y))
			//	rotate(image, image, ROTATE_90_CLOCKWISE);
			//else
			//	rotate(image, image, ROTATE_90_COUNTERCLOCKWISE);

			rotate(image, image, ROTATE_90_COUNTERCLOCKWISE);

		}
	}
	else
	{
		if ((float)rects[0].width / (float)rects[0].height < 1.5) {
			rotate(image, image, ROTATE_90_COUNTERCLOCKWISE);
		}
	}
	hsv.release();
	mask.release();
	rects.clear();
	rects.shrink_to_fit();
	//int b = 0;
}

EXPORT ushort SmartRotateImage(unsigned short rows, unsigned short cols, int widthStep, unsigned char* rgbData, unsigned char** firstPartR, unsigned char** firstPartG, unsigned char** firstPartB, int* rowFirst, int* colFirst) {
	Mat image = Mat(rows, cols, CV_8UC3, rgbData, widthStep);
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	//cvtColor(last, last, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, Scalar(20, 90, 80), Scalar(40, 225, 255), mask);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Rect> rects;

	//showImage(mask, "mask");

	findContours(mask, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contours.begin(), contours.end(), compareContourAreas);

	// Clear cac anh va bien khong su dung
	hierarchy.clear();
	hierarchy.shrink_to_fit();

	// Bien count de gioi han lay 10 contours
	int count = 0;

	// Duyet mang contour
	for (size_t i = 0; i < contours.size(); i++) {
		rects.push_back(boundingRect(contours[i]));

		if (count == 1)
			break;
		count++;
	}

	contours.clear();
	contours.shrink_to_fit();
	sort(rects.begin(), rects.end(), compareContourTopLeft);

	if (rects.size() > 1) {
		if ((rects[1].y - rects[0].br().y) > 0) {
			// So sanh khoang cach tu dau trang toi tl cua rect[0] voi k/c tu br cua rect[1] toi het anh de suy ra anh bi quay phia nao
			if (rects[0].y < (image.size().height - rects[1].br().y))
				rotate(image, image, ROTATE_90_CLOCKWISE);
			else
				rotate(image, image, ROTATE_90_COUNTERCLOCKWISE);
		}
		else if (((float)rects[0].br().y - (float)rects[1].y) / rects[0].height < 0.5)
		{
			// So sanh khoang cach tu dau trang toi tl cua rect[0] voi k/c tu br cua rect[1] toi het anh de suy ra anh bi quay phia nao
			if (rects[0].y < (image.size().height - rects[1].br().y))
				rotate(image, image, ROTATE_90_CLOCKWISE);
			else
				rotate(image, image, ROTATE_90_COUNTERCLOCKWISE);
		}
		else
		{
			image.release();
			return 0;
		}
	}
	else
	{
		if ((float)rects[0].width / (float)rects[0].height < 1.5) {
			rotate(image, image, ROTATE_90_COUNTERCLOCKWISE);
		}
		else {
			image.release();
			return 0;
		}
	}


	hsv.release();
	mask.release();
	rects.clear();
	rects.shrink_to_fit();

	Mat channel[3];
	split(image, channel);
	int size;

	// Red
	size = channel[2].total() * channel[2].elemSize();
	*firstPartR = new byte[size];  // you will have to delete[] that later
	std::memcpy(*firstPartR, channel[2].data, size * sizeof(byte));

	// Green
	size = channel[1].total() * channel[1].elemSize();
	*firstPartG = new byte[size];  // you will have to delete[] that later
	std::memcpy(*firstPartG, channel[1].data, size * sizeof(byte));

	// Blue
	size = channel[0].total() * channel[0].elemSize();
	*firstPartB = new byte[size];  // you will have to delete[] that later
	std::memcpy(*firstPartB, channel[0].data, size * sizeof(byte));


	channel->release();

	*rowFirst = image.rows;
	*colFirst = image.cols;

	image.release();
	return 1;
}

EXPORT ushort DewarpNSplit(unsigned short rows, unsigned short cols, int widthStep, unsigned char* rgbData, unsigned char** firstPartR, unsigned char** firstPartG, unsigned char** firstPartB, unsigned char** secondPartR, unsigned char** secondPartG, unsigned char** secondPartB, int* rowFirst, int* colFirst, int* rowSecond, int* colSecond) {
	Mat image = Mat(rows, cols, CV_8UC3, rgbData, widthStep);
	//Mat image = imread("D:\\THINH\\DATA\\GCN_XE_MAY_OTO\\Input\\OTO\\z2983895730621_b2af036a2861c84787c3a5c61b83dc9e.jpg");

	RotateImage(image);
	Mat img_org = image.clone();
	resize(image, image, Size(1500, 880));
	Mat gray;
	// Convert grayscale
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// Canny
	Mat canny;
	GaussianBlur(gray, canny, Size(5, 5), 0);
	Canny(canny, canny, 0, 50);
	//showImage(canny, "canny");



	rectangle(canny, Point(0, 0), Point(canny.cols - 1, canny.rows - 1), Scalar(255, 255, 255), 5);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Point> approx;

	findContours(canny, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contours.begin(), contours.end(), compareContourAreas);

	// Clear cac anh va bien khong su dung
	hierarchy.clear();
	hierarchy.shrink_to_fit();
	gray.release();

	//gray.release();
	canny.release();

	// Bien count de gioi han lay 10 contours
	int count = 0;
	int totalContoursArea = 0;
	// Tao mang? chua cac contour phan` can` dewarp
	vector<vector<Point>> approxs;

	// Duyet mang contour
	for (size_t i = 1; i < contours.size(); i++) {
		double p = arcLength(contours[i], true);
		approxPolyDP(contours[i], approx, 0.02 * p, true);

		// Cac contours co the bi trung` nhau/xe lech. nhau 1 ti' nhung van la 1 contour. Ham` kiem tra xem contour
		// da ton` tai trong mang? approxs chua
		if (approxs.size() > 0) {
			if (checkIfContourAvailable(approxs, approx))
				continue;
		}

		// Neu' contours chua' 4 diem? dinh? hoac 12 diem dinh
		if (approx.size() == 4 || approx.size() == 12) {
			approxs.push_back(approx);

			totalContoursArea = totalContoursArea + contourArea(approx);
		}

		if (count == 10 || approxs.size() == 5)
			break;
		count++;
	}

	approx.clear();
	approx.shrink_to_fit();
	contours.clear();
	contours.shrink_to_fit();

	// Sap xep cac contours theo thu tu giam dan dien tich roi loai bo nhung contours co gia tri khong dong` nhat
	sort(approxs.begin(), approxs.end(), compareContourAreas);
	vector<vector<Point>> listSelected = findMostCommonAreaIdx(approxs);
	//size_t skipIdx = -1;
	sort(listSelected.begin(), listSelected.end(), compareContourPosition);


	if (listSelected.size() == 0)
		return 0;
	// Sap xep cac contour theo thu tu ve x
	vector<Mat> splited;
	Point2f prevSize = Point2f(800, 1128);
	bool deadPages = false;
	for (size_t i = 0; i < listSelected.size(); i++)
	{
		vector<Point2f> approx2f;
		approx2f = rectify(listSelected[i], img_org.cols, img_org.rows);

		// widths and heights of the projected image
		float w1 = euclideanDist2f(approx2f[0], approx2f[1]);
		float w2 = euclideanDist2f(approx2f[3], approx2f[2]);

		float h1 = euclideanDist2f(approx2f[0], approx2f[3]);
		float h2 = euclideanDist2f(approx2f[1], approx2f[2]);

		float w = max(w1, w2);
		float h = max(h1, h2);

		// visible aspect ratio
		float ar_vis = w / h;

		// get real aspect ratio
		float ar_real = getRealAspectRatio(img_org.cols, img_org.rows, approx2f);

		float W = 0.0, H = 0.0;
		if ((ar_real > 0 == false) && (ar_real < 0 == false)) {
			deadPages = true;
			break;
		}
		else
		{
			if (ar_real < ar_vis) {
				W = (w);
				H = (W / ar_real);
			}
			else
			{
				H = (h);
				W = (ar_real * H);
			}
		}

		Point2f p2f[4] = { approx2f[0], approx2f[1], approx2f[2], approx2f[3] };
		Point2f p2f2[4] = { Point2f(0.0, 0.0), Point2f(W, 0.0), Point2f(W, H), Point2f(0.0, H) };
		Mat m = getPerspectiveTransform(p2f, p2f2);
		Mat dst;
		warpPerspective(img_org, dst, m, Size((int)W, (int)H));

		splited.push_back(dst);
		//showImage(dst, "dst" + to_string(i));

		// Free memory
		m.release();
		dst.release();
		approx2f.clear();
		approx2f.shrink_to_fit();

		if (splited[splited.size() - 1].rows < (img_org.rows / 2) && splited[splited.size() - 1].cols < (img_org.cols / 2)) {
			deadPages = true;
			break;
		}
	}

	if (deadPages) {
		img_org.release();
		approxs.clear();
		approxs.shrink_to_fit();
		listSelected.clear();
		listSelected.shrink_to_fit();
		splited.clear();
		splited.shrink_to_fit();

		return 0;
	}

	img_org.release();
	approxs.clear();
	approxs.shrink_to_fit();
	listSelected.clear();
	listSelected.shrink_to_fit();

	// Loc. bang mau` nhung trang khong phai mau vang
	int skipIdx = -1;
	if (splited.size() > 2) {
		skipIdx = findNonYellowPage(splited);
	}

	vector<Mat> yellowPages;

	if (skipIdx > 0) {
		for (int i = splited.size() - 1; i >= 0; i--)
		{
			if (i == skipIdx)
				continue;
			cv::rotate(splited[i], splited[i], ROTATE_180);
			yellowPages.push_back(splited[i].clone());
		}
	}
	else
	{
		for (int i = 0; i < splited.size(); i++)
		{
			if (i == skipIdx)
				continue;
			yellowPages.push_back(splited[i].clone());
		}
	}

	splited.clear();
	splited.shrink_to_fit();

	int isize = 0;
	if (yellowPages.size() > 0) {
		// 9. Chuyển ảnh 2 chiều thành mảng byte để trả về
			// Chuyển ảnh 2 chiều thành ảnh 1 chiều

		Mat channel[3];
		split(yellowPages[0], channel);
		int size;

		// Red
		size = channel[2].total() * channel[2].elemSize();
		*firstPartR = new byte[size];  // you will have to delete[] that later
		std::memcpy(*firstPartR, channel[2].data, size * sizeof(byte));

		// Green
		size = channel[1].total() * channel[1].elemSize();
		*firstPartG = new byte[size];  // you will have to delete[] that later
		std::memcpy(*firstPartG, channel[1].data, size * sizeof(byte));

		// Blue
		size = channel[0].total() * channel[0].elemSize();
		*firstPartB = new byte[size];  // you will have to delete[] that later
		std::memcpy(*firstPartB, channel[0].data, size * sizeof(byte));

		isize = size;

		channel->release();
	}
	//test.
	int isize2 = 0;
	if (yellowPages.size() > 1) {
		Mat channel[3];
		split(yellowPages[1], channel);
		int size;

		// Red
		size = channel[2].total() * channel[2].elemSize();
		*secondPartR = new byte[size];  // you will have to delete[] that later
		std::memcpy(*secondPartR, channel[2].data, size * sizeof(byte));

		// Green
		size = channel[1].total() * channel[1].elemSize();
		*secondPartG = new byte[size];  // you will have to delete[] that later
		std::memcpy(*secondPartG, channel[1].data, size * sizeof(byte));

		// Blue
		size = channel[0].total() * channel[0].elemSize();
		*secondPartB = new byte[size];  // you will have to delete[] that later
		std::memcpy(*secondPartB, channel[0].data, size * sizeof(byte));

		isize2 = size;
		channel->release();
	}

	*rowFirst = yellowPages[0].rows;
	*colFirst = yellowPages[0].cols;

	*rowSecond = yellowPages[1].rows;
	*colSecond = yellowPages[1].cols;

	yellowPages.clear();
	yellowPages.shrink_to_fit();

	if (isize == 0 || isize2 == 0)
		return 0;
	if (isize == 0 && isize2 == 0)
		return -1;
	return 1;
}

EXPORT ushort DewarpNSplitHSV(unsigned short rows, unsigned short cols, int widthStep, unsigned char* rgbData, unsigned char** firstPartR, unsigned char** firstPartG, unsigned char** firstPartB, unsigned char** secondPartR, unsigned char** secondPartG, unsigned char** secondPartB, int* rowFirst1, int* colFirst1, int* rowSecond1, int* colSecond1) {
	Mat image = Mat(rows, cols, CV_8UC3, rgbData, widthStep);
	RotateImage(image);
	Mat img_org = image.clone();

	resize(image, image, Size(1500, 880));

	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, Scalar(20, 90, 80), Scalar(40, 225, 255), mask);

	hsv.release();

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Point> approx;

	findContours(mask, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contours.begin(), contours.end(), compareContourAreas);

	// Clear cac anh va bien khong su dung
	hierarchy.clear();
	hierarchy.shrink_to_fit();

	// Bien count de gioi han lay 10 contours
	int count = 0;

	// Tao mang? chua cac contour phan` can` dewarp
	vector<vector<Point>> approxs;

	// Duyet mang contour
	for (size_t i = 0; i < contours.size(); i++) {
		double p = arcLength(contours[i], true);
		approxPolyDP(contours[i], approx, 0.1 * p, true);

		if (approx.size() < 4) {
			approx.clear();
			approx.shrink_to_fit();
			p = arcLength(contours[i], true);
			approxPolyDP(contours[i], approx, 0.02 * p, true);
		}

		approxs.push_back(approx);

		if (count == 1)
			break;
		count++;
	}

	approx.clear();
	approx.shrink_to_fit();
	contours.clear();
	contours.shrink_to_fit();
	image.release();

	if (approxs.size() == 0)
		return 0;
	// Sap xep cac contour theo thu tu ve x
	vector<Mat> splited;

	Point2f prevSize = Point2f(800, 1128);
	for (size_t i = 0; i < approxs.size(); i++)
	{
		vector<Point2f> approx2f;
		approx2f = rectify(approxs[i], img_org.cols, img_org.rows);

		// widths and heights of the projected image
		float w1 = euclideanDist2f(approx2f[0], approx2f[1]);
		float w2 = euclideanDist2f(approx2f[3], approx2f[2]);

		float h1 = euclideanDist2f(approx2f[0], approx2f[3]);
		float h2 = euclideanDist2f(approx2f[1], approx2f[2]);

		float w = max(w1, w2);
		float h = max(h1, h2);

		// visible aspect ratio
		float ar_vis = w / h;

		// get real aspect ratio
		float ar_real = getRealAspectRatio(img_org.cols, img_org.rows, approx2f);

		float W = 0.0, H = 0.0;
		if ((ar_real > 0 == false) && (ar_real < 0 == false)) {
			W = prevSize.x; H = prevSize.y;
		}
		else
		{
			if (ar_real < ar_vis) {
				W = (w);
				H = (W / ar_real);
			}
			else
			{
				H = (h);
				W = (ar_real * H);
			}
		}

		Point2f p2f[4] = { approx2f[0], approx2f[1], approx2f[2], approx2f[3] };
		Point2f p2f2[4] = { Point2f(0.0, 0.0), Point2f(W, 0.0), Point2f(W, H), Point2f(0.0, H) };
		Mat m = getPerspectiveTransform(p2f, p2f2);
		Mat dst;
		warpPerspective(img_org, dst, m, Size((int)W, (int)H));

		prevSize = Point2f(W, H);
		splited.push_back(dst);

		// Free memory
		m.release();
		dst.release();
		approx2f.clear();
		approx2f.shrink_to_fit();
	}
	img_org.release();
	approxs.clear();
	approxs.shrink_to_fit();

	if (splited.size() > 0) {
		// 9. Chuyển ảnh 2 chiều thành mảng byte để trả về
			// Chuyển ảnh 2 chiều thành ảnh 1 chiều

		Mat channel[3];
		split(splited[0], channel);
		int size;

		// Red
		size = channel[2].total() * channel[2].elemSize();
		*firstPartR = new byte[size];  // you will have to delete[] that later
		std::memcpy(*firstPartR, channel[2].data, size * sizeof(byte));

		// Green
		size = channel[1].total() * channel[1].elemSize();
		*firstPartG = new byte[size];  // you will have to delete[] that later
		std::memcpy(*firstPartG, channel[1].data, size * sizeof(byte));

		// Blue
		size = channel[0].total() * channel[0].elemSize();
		*firstPartB = new byte[size];  // you will have to delete[] that later
		std::memcpy(*firstPartB, channel[0].data, size * sizeof(byte));

		channel->release();
	}

	if (splited.size() > 1) {
		Mat channel[3];
		split(splited[1], channel);
		int size;

		// Red
		size = channel[2].total() * channel[2].elemSize();
		*secondPartR = new byte[size];  // you will have to delete[] that later
		std::memcpy(*secondPartR, channel[2].data, size * sizeof(byte));

		// Green
		size = channel[1].total() * channel[1].elemSize();
		*secondPartG = new byte[size];  // you will have to delete[] that later
		std::memcpy(*secondPartG, channel[1].data, size * sizeof(byte));

		// Blue
		size = channel[0].total() * channel[0].elemSize();
		*secondPartB = new byte[size];  // you will have to delete[] that later
		std::memcpy(*secondPartB, channel[0].data, size * sizeof(byte));

		channel->release();
	}
	else {
		splited.clear();
		splited.shrink_to_fit();
		return 0;
	}

	*rowFirst1 = splited[0].rows;
	*colFirst1 = splited[0].cols;

	*rowSecond1 = splited[1].rows;
	*colSecond1 = splited[1].cols;

	splited.clear();
	splited.shrink_to_fit();
	return 1;
}


//
//int main(int argc, char** argv)
//{
//	//Mat image = imread("D:\\THINH\\DATA\\GCN_XE_MAY_OTO\\Input\\OTO\\Test\\z2915266292868_788024e8395fced019b34eccac056847.jpg");
//	//Mat image = imread("D:\\THINH\\DATA\\GCN_XE_MAY_OTO\\Input\\OTO\\Test\\result\\org\\z2983901323886_880ac234de7ad0e9106c7cfe0e63619f.jpg");
//	Mat image = imread("D:\\THINH\\DATA\\GCN_XE_MAY_OTO\\Input\\OTO\\Oto\\z2915266160437_1d2dfc77afc72de1f851e8e7276c7b12.jpg");
//	if (image.empty())
//	{
//		cout << "Could not open or find the image" << endl;
//		cin.get(); //wait for any key press
//		return -1;
//	}
//
//	Mat img_org = image.clone();
//
//	RotateImage(image);
//	showImage(image, "rotateImg");
//	resize(image, image, Size(1500, 880));
//	Mat gray;
//	// Convert grayscale
//	cvtColor(image, gray, COLOR_BGR2GRAY);
//
//	// Canny
//	Mat canny;
//	GaussianBlur(gray, canny, Size(5,5), 0);
//	Canny(canny, canny, 0, 50);
//
//	//dilate(canny, canny, getStructuringElement(MORPH_RECT, Size(3, 1)));
//
//	rectangle(canny, Point(0, 0), Point(canny.cols - 1, canny.rows - 1), Scalar(255, 255, 255), 5);
//	showImage(canny, "canny");
//
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	vector<Point> approx;
//
//	findContours(canny, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
//	sort(contours.begin(), contours.end(), compareContourAreas);
//
//	// Clear cac anh va bien khong su dung
//	hierarchy.clear();
//	hierarchy.shrink_to_fit();
//
//	//gray.release();
//	canny.release();
//
//	// Bien count de gioi han lay 10 contours
//	int count = 0;
//	int totalContoursArea = 0;
//	// Tao mang? chua cac contour phan` can` dewarp
//	vector<vector<Point>> approxs;
//
//	// Duyet mang contour
//	for (size_t i = 1; i < contours.size(); i++) {
//		double p = arcLength(contours[i], true);
//		approxPolyDP(contours[i], approx, 0.02 * p, true);
//
//		// Cac contours co the bi trung` nhau/xe lech. nhau 1 ti' nhung van la 1 contour. Ham` kiem tra xem contour
//		// da ton` tai trong mang? approxs chua
//		if (approxs.size() > 0) {
//			if (checkIfContourAvailable(approxs, approx))
//				continue;
//		}
//
//
//#pragma region Debug
//		if (1 == 1) {
//			Mat imgCln = image.clone();
//			drawContours(imgCln, contours, i, Scalar(0, 255, 0), -1);
//			for (size_t k = 0; k < approx.size(); k++)
//			{
//				circle(imgCln, approx[k], 2, Scalar(0, 0, 255), -1);
//			}
//			String path = "D:\\THINH\\C++\\FSI\\Draft\\contours\\";
//			imwrite(path + "cnt" + to_string(i) + ".bmp", imgCln);
//			imgCln.release();
//		}
//#pragma endregion
//
//		// Neu' contours chua' 4 diem? dinh? hoac 12 diem dinh
//		if (approx.size() == 4 || approx.size() == 12) {
//			approxs.push_back(approx);
//
//			totalContoursArea = totalContoursArea + contourArea(approx);
//
//			if (1 == 1) {
//				Mat imgCln = image.clone();
//				drawContours(imgCln, contours, i, Scalar(0, 255, 0), -1);
//				for (size_t k = 0; k < approx.size(); k++)
//				{
//					circle(imgCln, approx[k], 2, Scalar(0, 0, 255), -1);
//				}
//				String path = "D:\\THINH\\C++\\FSI\\Draft\\contours\\selected\\";
//				imwrite(path + "cnt" + to_string(i) + ".bmp", imgCln);
//				imgCln.release();
//			}
//		}
//
//		if (count == 10 || approxs.size() == 5)
//			break;
//		count++;
//	}
//
//	approx.clear();
//	approx.shrink_to_fit();
//
//	// Sap xep cac contours theo thu tu giam dan dien tich roi loai bo nhung contours co gia tri khong dong` nhat
//	sort(approxs.begin(), approxs.end(), compareContourAreas);
//	vector<vector<Point>> listSelected = findMostCommonAreaIdx(approxs);
//	//size_t skipIdx = -1;
//	sort(listSelected.begin(), listSelected.end(), compareContourPosition);
//	
//
//	if (listSelected.size() == 0)
//		return 0;
//	// Sap xep cac contour theo thu tu ve x
//	vector<Mat> splited;
//
//	image.release();
//
//
//	Point2f prevSize = Point2f(800, 1128);
//
//	bool deadPages = false;
//	for (size_t i = 0; i < listSelected.size(); i++)
//	{
//		vector<Point2f> approx2f;
//		approx2f = rectify(listSelected[i], img_org.cols, img_org.rows);
//
//		// widths and heights of the projected image
//		float w1 = euclideanDist2f(approx2f[0], approx2f[1]);
//		float w2 = euclideanDist2f(approx2f[3], approx2f[2]);
//
//		float h1 = euclideanDist2f(approx2f[0], approx2f[3]);
//		float h2 = euclideanDist2f(approx2f[1], approx2f[2]);
//
//		float w = max(w1, w2);
//		float h = max(h1, h2);
//
//		// visible aspect ratio
//		float ar_vis = w / h;
//
//		// get real aspect ratio
//		float ar_real = getRealAspectRatio(img_org.cols, img_org.rows, approx2f);
//
//		float W = 0.0, H = 0.0;
//		if ((ar_real > 0 == false) && (ar_real < 0 == false)) 
//		{
//			deadPages = true;
//			break;
//		}
//		else
//		{
//			if (ar_real < ar_vis) {
//				W = (w);
//				H = (W / ar_real);
//			}
//			else
//			{
//				H = (h);
//				W = (ar_real * H);
//			}
//		}
//
//		Point2f p2f[4] = { approx2f[0], approx2f[1], approx2f[2], approx2f[3] };
//		Point2f p2f2[4] = { Point2f(0.0, 0.0), Point2f(W, 0.0), Point2f(W, H), Point2f(0.0, H) };
//		Mat m = getPerspectiveTransform(p2f, p2f2);
//		Mat dst;
//		warpPerspective(img_org, dst, m, Size((int)W, (int)H));
//
//		splited.push_back(dst);
//		//showImage(dst, "dst" + to_string(i));
//
//		// Free memory
//		approx2f.clear();
//		approx2f.shrink_to_fit();
//		m.release();
//		dst.release();
//		
//		if (splited[splited.size() - 1].rows < (img_org.rows / 2) && splited[splited.size() - 1].cols < (img_org.cols / 2)) {
//			deadPages = true;
//			break;
//		}
//	}
//
//	if (deadPages) {
//		img_org.release();
//		approxs.clear();
//		approxs.shrink_to_fit();
//		listSelected.clear();
//		listSelected.shrink_to_fit();
//		splited.clear();
//		splited.shrink_to_fit();
//
//		return 0;
//	}
//
//	gray.release();
//	img_org.release();
//	approxs.clear();
//	approxs.shrink_to_fit();
//	listSelected.clear();
//	listSelected.shrink_to_fit();
//
//	// Loc. bang mau` nhung trang khong phai mau vang
//	int skipIdx = -1;
//	if (splited.size() > 2) {
//		skipIdx = findNonYellowPage(splited);
//	}
//
//	vector<Mat> yellowPages;
//
//	if (skipIdx > 0) {
//		for (int i = splited.size() - 1; i >= 0; i--)
//		{
//			if (i == skipIdx)
//				continue;
//			cv::rotate(splited[i], splited[i], ROTATE_180);
//			yellowPages.push_back(splited[i].clone());
//		}
//	}
//	else
//	{
//		for (int i = 0; i < splited.size(); i++)
//		{
//			if (i == skipIdx)
//				continue;
//			yellowPages.push_back(splited[i].clone());
//		}
//	}
//
//	splited.clear();
//	splited.shrink_to_fit();
//	for (size_t i = 0; i < yellowPages.size(); i++)
//	{
//		showImage(yellowPages[i], "dst" + to_string(i));
//	}
//
//	return 0;
//}

//
//int main(int argc, char** argv)
//{
//	//Mat image = imread("D:\\THINH\\DATA\\GCN_XE_MAY_OTO\\Input\\OTO\\Test\\z2915266292868_788024e8395fced019b34eccac056847.jpg");
//	//Mat image = imread("D:\\THINH\\DATA\\GCN_XE_MAY_OTO\\Input\\OTO\\Test\\result\\org\\z2983901323886_880ac234de7ad0e9106c7cfe0e63619f.jpg");
//	Mat image = imread("D:\\THINH\\DATA\\GCN_XE_MAY_OTO\\Input\\OTO\\Oto\\overflow\\z2915258836051_44fd5518243f4aaf70baac63a2ab581a.jpg");
//	if (image.empty())
//	{
//		cout << "Could not open or find the image" << endl;
//		cin.get(); //wait for any key press
//		return -1;
//	}
//
//	RotateImage(image);
//	Mat img_org = image.clone();
//
//	//showImage(image, "rotateImg");
//	resize(image, image, Size(1500, 880));
//
//	Mat hsv;
//	cvtColor(image, hsv, COLOR_BGR2HSV);
//	//cvtColor(last, last, COLOR_BGR2HSV);
//	Mat mask;
//	inRange(hsv, Scalar(20, 90, 80), Scalar(40, 225, 255), mask);
//
//	hsv.release();
//
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	//vector<Rect> rects;
//	vector<Point> approx;
//
//	//showImage(mask, "mask");
//
//	findContours(mask, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
//	sort(contours.begin(), contours.end(), compareContourAreas);
//
//	// Clear cac anh va bien khong su dung
//	hierarchy.clear();
//	hierarchy.shrink_to_fit();
//
//	// Bien count de gioi han lay 10 contours
//	int count = 0;
//
//	// Tao mang? chua cac contour phan` can` dewarp
//	vector<vector<Point>> approxs;
//
//	// Duyet mang contour
//	for (size_t i = 0; i < contours.size(); i++) {
//		double p = arcLength(contours[i], true);
//		approxPolyDP(contours[i], approx, 0.1 * p, true);
//
//#pragma region Debug
//		if (1 == 1) {
//			Mat imgCln = image.clone();
//			drawContours(imgCln, contours, i, Scalar(0, 255, 0), -1);
//			for (size_t k = 0; k < approx.size(); k++)
//			{
//				circle(imgCln, approx[k], 2, Scalar(0, 0, 255), -1);
//			}
//			String path = "D:\\THINH\\C++\\FSI\\Draft\\contours\\";
//			imwrite(path + "cnt" + to_string(i) + ".bmp", imgCln);
//			imgCln.release();
//		}
//#pragma endregion
//
//		if (approx.size() < 4) {
//			approx.clear();
//			approx.shrink_to_fit();
//			p = arcLength(contours[i], true);
//			approxPolyDP(contours[i], approx, 0.02 * p, true);
//		}
//
//		approxs.push_back(approx);
//
//		if (count == 1)
//			break;
//		count++;
//	}
//
//	approx.clear();
//	approx.shrink_to_fit();
//	contours.clear();
//	contours.shrink_to_fit();
//	image.release();
//
//	if (approxs.size() == 0)
//		return 0;
//	// Sap xep cac contour theo thu tu ve x
//	vector<Mat> splited;
//
//	Point2f prevSize = Point2f(800, 1128);
//	for (size_t i = 0; i < approxs.size(); i++)
//	{
//		vector<Point2f> approx2f;
//		approx2f = rectify(approxs[i], img_org.cols, img_org.rows);
//
//		// widths and heights of the projected image
//		float w1 = euclideanDist2f(approx2f[0], approx2f[1]);
//		float w2 = euclideanDist2f(approx2f[3], approx2f[2]);
//
//		float h1 = euclideanDist2f(approx2f[0], approx2f[3]);
//		float h2 = euclideanDist2f(approx2f[1], approx2f[2]);
//
//		float w = max(w1, w2);
//		float h = max(h1, h2);
//
//		// visible aspect ratio
//		float ar_vis = w / h;
//
//		// get real aspect ratio
//		float ar_real = getRealAspectRatio(img_org.cols, img_org.rows, approx2f);
//
//		float W = 0.0, H = 0.0;
//		if ((ar_real > 0 == false) && (ar_real < 0 == false)) {
//			W = prevSize.x; H = prevSize.y;
//		}
//		else
//		{
//			if (ar_real < ar_vis) {
//				W = (w);
//				H = (W / ar_real);
//			}
//			else
//			{
//				H = (h);
//				W = (ar_real * H);
//			}
//		}
//
//		Point2f p2f[4] = { approx2f[0], approx2f[1], approx2f[2], approx2f[3] };
//		Point2f p2f2[4] = { Point2f(0.0, 0.0), Point2f(W, 0.0), Point2f(W, H), Point2f(0.0, H) };
//		Mat m = getPerspectiveTransform(p2f, p2f2);
//		Mat dst;
//		warpPerspective(img_org, dst, m, Size((int)W, (int)H));
//
//		prevSize = Point2f(W, H);
//		splited.push_back(dst);
//
//		// Free memory
//		m.release();
//		dst.release();
//		approx2f.clear();
//		approx2f.shrink_to_fit();
//	}
//	img_org.release();
//	approxs.clear();
//	approxs.shrink_to_fit();
//
//	for (size_t i = 0; i < splited.size(); i++)
//	{
//		showImage(splited[i], "dst" + to_string(i));
//	}
//
//	splited.clear();
//	splited.shrink_to_fit();
//
//	return 0;
//}


#pragma region Debug
/*if (1 == 1) {
	Mat imgCln = image.clone();
	drawContours(imgCln, contours, i, Scalar(0, 255, 0), -1);
	for (size_t k = 0; k < approx.size(); k++)
	{
		circle(imgCln, approx[k], 2, Scalar(0, 0, 255), -1);
	}
	String path = "D:\\THINH\\C++\\FSI\\Draft\\contours\\";
	imwrite(path + "cnt" + to_string(i) + ".bmp", imgCln);
	imgCln.release();
}*/
#pragma endregion


/*if (1 == 1) {
				Mat imgCln = image.clone();
				drawContours(imgCln, contours, i, Scalar(0, 255, 0), -1);
				for (size_t k = 0; k < approx.size(); k++)
				{
					circle(imgCln, approx[k], 2, Scalar(0, 0, 255), -1);
				}
				String path = "D:\\THINH\\C++\\FSI\\Draft\\contours\\selected\\";
				imwrite(path + "cnt" + to_string(i) + ".bmp", imgCln);
				imgCln.release();
			}*/