#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

#include <Windows.h>
#include <vector>
#include <cmath>
#include <unordered_set>

#undef min

// Struct holding all infos about each strip, e.g. length
struct MyStrip {
	int stripeLength;
	int nStop;
	int nStart;
	cv::Point2f stripeVecX;
	cv::Point2f stripeVecY;
};


bool g_bBoxes = false;
bool g_bIsFirstStripe = true;
bool g_bIsFirstMarker = true;
int g_bw_slider_value = 55;
const std::string g_twoDMarkerWindow = "2D Marker";

bool AllValuesDistinct(const std::array<int, 4>& codes)
{
	std::unordered_set<int> s;
	for (int i = 0; i < codes.size(); i++) {
		s.insert(codes[i]);
	}
	return (s.size() == codes.size());
}

// https://stackoverflow.com/questions/7446126/opencv-2d-line-intersection-helper-function
bool FindIntersection(const cv::Point2f& o1, const cv::Point2f& p1, 
	const cv::Point2f& o2, const cv::Point2f& p2, cv::Point2f& r)
{
	cv::Point2f x = o2 - o1;
	cv::Point2f d1 = p1 - o1;
	cv::Point2f d2 = p2 - o2;

	const double cross = d1.x * d2.y - d1.y * d2.x;
	if (std::abs(cross) < /*EPS*/1e-8)
		return false;

	const double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}

cv::Mat calculate_Stripe(double dx, double dy, MyStrip& strip)
{
	const double diffLength = sqrt(dx * dx + dy * dy);
	int stripeLength = static_cast<int>(0.8 * diffLength);
	if (stripeLength < 5)
		stripeLength = 5;
	else if (stripeLength % 2 == 0)
		stripeLength++;

	strip.stripeLength = stripeLength;
	// Direction vectors
	strip.stripeVecX.x = dx / diffLength;
	strip.stripeVecX.y = dy / diffLength;
	strip.stripeVecY.x = strip.stripeVecX.y;
	strip.stripeVecY.y = -strip.stripeVecX.x;

	strip.nStop = strip.stripeLength / 2;
	strip.nStart = -strip.nStop;

	cv::Size stripeSize;
	stripeSize.width = 3;
	stripeSize.height = stripeLength;
	// 8 bit unsigned char with 1 channel, gray
	return cv::Mat(stripeSize, CV_8UC1);
}

int subpixSampleSafe(const cv::Mat& pSrc, const cv::Point2f& p) 
{
	// floorf -> like int casting, but -2.3 will be the smaller number -> -3
	// Point is float, we want to know which color does it have
	const int fx = static_cast<int>(floorf(p.x));
	const int fy = static_cast<int>(floorf(p.y));

	if (fx < 0 || fx >= pSrc.cols - 1 ||
		fy < 0 || fy >= pSrc.rows - 1)
		return 127;

	// Slides 15
	const int px = static_cast<int>(256 * (p.x - floorf(p.x)));
	const int py = static_cast<int>(256 * (p.y - floorf(p.y)));

	// Here we get the pixel of the starting point
	auto i = pSrc.data + fy * pSrc.step + fx;

	// Shift 2^8
	// Internsity
	const int a = i[0] + ((px * (i[1] - i[0])) >> 8);
	i += pSrc.step;
	const int b = i[0] + ((px * (i[1] - i[0])) >> 8);

	// We want to return Intensity for the subpixel
	return a + ((py * (b - a)) >> 8);
}

void GrayScaleImage(const cv::Mat& frame, cv::Mat& result, int thresholdValue, int thresholdType)
{
	cv::Mat grayScaledFrame;
	// Convert into gray image
	cv::cvtColor(frame, grayScaledFrame, CV_BGR2GRAY);
	// Apply threshold from trackbar
	cv::threshold(grayScaledFrame, result, thresholdValue, 255, thresholdType);
}

void AdaptiveGrayScaleImage(const cv::Mat& frame, cv::Mat& result, int thresholdValue, int thresholdType)
{
	cv::Mat grayScaledFrame;
	// Convert into gray image
	cv::cvtColor(frame, grayScaledFrame, CV_BGR2GRAY);
	// Apply threshold from trackbar
	cv::adaptiveThreshold(grayScaledFrame, result, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 33, 5.);
}

/**
 * Sort the 4 points of a shape in the order: topLeft, topRight, bottomLeft, bottomRight
 * https://stackoverflow.com/questions/32105729/how-to-control-the-order-of-points-in-opencv-findcontours
 */
void SortContourPoints(std::vector<cv::Point>& contour)
{
	auto sortX = [](const cv::Point& pt1, const cv::Point& pt2) {return (pt1.y < pt2.y); };
	auto sortY = [](const cv::Point& pt1, const cv::Point& pt2) {return (pt1.x < pt2.x); };
	std::sort(contour.begin(), contour.end(), sortY);
	std::sort(contour.begin(), contour.begin() + 2, sortX);
	std::sort(contour.begin() + 2, contour.end(), sortX);
}

bool IsValidSize(std::vector<cv::Point>& shape, const cv::Rect& boundingBox, const cv::Mat& inputFrame, int sizeLimit)
{
	const double contourSize = cv::contourArea(shape);
	if (boundingBox.height < 20 || boundingBox.width < 20 || boundingBox.width > inputFrame.cols - 10 || boundingBox.height > inputFrame.rows - 10 || contourSize < sizeLimit)
	{
		return false;
	}

	return true;
}

std::vector<std::vector<cv::Point>> FindMarkerShapes(const cv::Mat& input)
{
	// A vector of shapes, each shape represented by another vector of Points
	std::vector<std::vector<cv::Point>> contours, boundingBoxes;
	cv::findContours(input, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	for (auto& shape : contours)
	{
		std::vector<cv::Point> approximation;
		// Connect the points into a shape
		cv::approxPolyDP(shape, approximation, cv::arcLength(shape, true) * 0.02, true);
		// Filter for rectangles
		if (approximation.size() == 4)
		{
			// Create and save a bounding box around our rectangle
			cv::Rect rect = cv::boundingRect(approximation);
			// Filter out small rectangles that are likely just noise
			if (!IsValidSize(shape, rect, input, 1000))
			{
				continue;
			}
			
			if (g_bBoxes)
				boundingBoxes.push_back({ rect.tl(), cv::Point(rect.x + rect.width, rect.y), rect.br(), cv::Point(rect.x, rect.y + rect.height) });
			else
				boundingBoxes.push_back(approximation);
		}
	}

	return boundingBoxes;
}

int IdentifyMarker(const cv::Mat& imageMarker)
{
	int code = 0;
	for (int i = 0; i < 6; ++i) 
	{
		int pixel1 = imageMarker.at<uchar>(0, i); //top
		int pixel2 = imageMarker.at<uchar>(5, i); //bottom
		int pixel3 = imageMarker.at<uchar>(i, 0); //left
		int pixel4 = imageMarker.at<uchar>(i, 5); //right

		if (pixel1 != 0 || pixel2 != 0 || pixel3 != 0 || pixel4 != 0) 
		{
			code = -1;
			break;
		}
	}
	if (code < 0)
	{
		return -1;
	}

	// Copy the BW values into cP -> codePixel on the marker 4x4 (inner part of the marker, no black border).
	// If black then 1 else 0
	int cP[4][4];
	for (int i = 0; i < 4; i++) 
	{
		for (int j = 0; j < 4; j++) 
		{
			// +1 -> no borders!
			cP[i][j] = imageMarker.at<uchar>(i + 1, j + 1) == 0 ? 1 : 0;
		}
	}

	// Save the ID of the marker, for each side
	std::array<int, 4> codes{0, 0, 0, 0};
	
	// Calculate the code from all sides in one loop
	for (int i = 0; i < 16; i++) 
	{
		// /4 to go through the rows
		int row = i >> 2;
		int col = i % 4;

		// Multiplied by 2 to check for black values -> 0*2 = 0
		codes[0] <<= 1;
		codes[0] |= cP[row][col]; // 0°

		// 4x4 structure -> Each column represents one side 
		codes[1] <<= 1;
		codes[1] |= cP[3 - col][row]; // 90°

		codes[2] <<= 1;
		codes[2] |= cP[3 - row][3 - col]; // 180°

		codes[3] <<= 1;
		codes[3] |= cP[col][3 - row]; // 270°
	}
	
	if (!AllValuesDistinct(codes)) {
		return -1;
	}

	code = *std::min_element(codes.begin(), codes.end());
	printf("Found: %04x\n", code);

	return code;
}

void MarkMarkers(cv::Mat& frame, const cv::Mat& greyImage, const std::vector<std::vector<cv::Point>>& boundingBoxes)
{
	for (const auto& box : boundingBoxes)
	{
		// Connect the dots of the box with lines
		cv::polylines(frame, box, true, cv::Scalar(50, 50, 255, 255), 2);
		// Direction vector (x0,y0) and contained point (x1,y1) -> For each line -> 4x4 = 16
		float lineParams[16];
		std::vector<cv::Point2f> linePoints;
		// lineParams is shared, CV_32F -> Same data type like lineParams
		cv::Mat lineParamsMat(cv::Size(4, 4), CV_32F, lineParams);
		// Draw circles on the lines
		for (size_t i = 0; i < box.size(); ++i) 
		{
			// Render the corners, 3 -> Radius, -1 filled circle
			circle(frame, box[i], 3, CV_RGB(0, 255, 0), -1);

			// Euclidic distance, 7 -> parts, both directions dx and dy
			const double dx = (static_cast<double>(box[(i + 1) % 4].x) - static_cast<double>(box[i].x)) / 7.0;
			const double dy = (static_cast<double>(box[(i + 1) % 4].y) - static_cast<double>(box[i].y)) / 7.0;

			MyStrip stripe;
			cv::Mat imagePixelStripe = calculate_Stripe(dx, dy, stripe);

			// Array for edge point centers
			cv::Point2f edgePointCenters[6];

			// First point already rendered, now the other 6 points
			for (int j = 1; j < 7; ++j) 
			{
				// Position calculation
				const double px = static_cast<double>(box[i].x) + static_cast<double>(j) * dx;
				const double py = static_cast<double>(box[i].y) + static_cast<double>(j) * dy;

				cv::Point p;
				p.x = static_cast<int>(px);
				p.y = static_cast<int>(py);
				//circle(frame, p, 2, CV_RGB(0, 0, 255), -1);
				cv::Point2f pFloat;
				pFloat.x = static_cast<float>(p.x);
				pFloat.y = static_cast<float>(p.y);

				for (int m = -1; m <= 1; m++)
				{
					for (int n = stripe.nStart; n <= stripe.nStop; n++)
					{
						cv::Point2f subPixel = pFloat + stripe.stripeVecX * static_cast<float>(m) + stripe.stripeVecY * static_cast<float>(n);
						cv::Point subPixelInt = { static_cast<int>(subPixel.x), static_cast<int>(subPixel.y) };

						// The one (purple color) which is shown in the stripe window
						if (g_bIsFirstStripe)
							circle(frame, subPixelInt, 1, CV_RGB(255, 0, 255), -1);
						else
							circle(frame, subPixelInt, 1, CV_RGB(0, 255, 255), -1);

						const int pixel = subpixSampleSafe(greyImage, subPixel);
						const int w = m + 1;
						const int h = n + (stripe.stripeLength >> 1);
						imagePixelStripe.at<uchar>(h, w) = static_cast<uchar>(pixel);
					}
				}
				cv::Mat grad_y;
				cv::Sobel(imagePixelStripe, grad_y, CV_8UC1, 0, 1);
				double maxIntensity = -1;
				int maxIntensityIndex = 0;
				
				for (int n = 0; n < stripe.stripeLength - 2; ++n)
				{
					if (grad_y.at<uchar>(n, 1) > maxIntensity)
					{
						maxIntensity = grad_y.at<uchar>(n, 1);
						maxIntensityIndex = n;
					}
				}

				const int max1 = maxIntensityIndex - 1, max2 = maxIntensityIndex + 1;
				const double y0 = (maxIntensityIndex <= 0) ? 0 : grad_y.at<uchar>(max1, 1);
				const double y1 = grad_y.at<uchar>(maxIntensityIndex, 1);
				const double y2 = (maxIntensityIndex >= stripe.stripeLength - 3) ? 0 : grad_y.at<uchar>(max2, 1);
				const double pos = (y2 - y0) / (4 * y1 - 2 * y0 - 2 * y2);
				cv::Point2d edgeCenter;
				if (std::isnan(pos))
				{
					continue;
				}

				// Exact point with subpixel accuracy
				double maxIndexShift = maxIntensityIndex - (stripe.stripeLength >> 1);
				edgeCenter = pFloat + (maxIndexShift + pos) * stripe.stripeVecY;
				//edgeCenter.x = static_cast<double>(p.x) + ((maxIndexShift + pos) * stripe.stripeVecY.x);
				//edgeCenter.y = static_cast<double>(p.y) + ((maxIndexShift + pos) * stripe.stripeVecY.y);
				circle(frame, edgeCenter, 2, CV_RGB(0, 0, 255), -1);

				edgePointCenters[j - 1].x = edgeCenter.x;
				edgePointCenters[j - 1].y = edgeCenter.y;

				if (g_bIsFirstStripe)
				{
					cv::Mat iplTmp;
					// The intensity differences on the stripe
					resize(imagePixelStripe, iplTmp, cv::Size(100, 300));

					imshow("stripWindow", iplTmp);
					g_bIsFirstStripe = false;
				}
			}

			// We now have the array of exact edge centers stored in "points", every row has two values -> 2 channels!
			cv::Mat highIntensityPoints(cv::Size(1, 6), CV_32FC2, edgePointCenters);
			// fitLine stores the calculated line in lineParams per column in the following way:
			// vec: direction vectors, point: point on the line location
			// vec.x, vec.y, point.x, point.y
			// Norm 2, 0 and 0.01 -> Optimal parameters
			// i -> Edge points
			cv::fitLine(highIntensityPoints, lineParamsMat.col(i), CV_DIST_L2, 0, 0.01, 0.01);
			cv::Point2f p1;
			// We have to jump through the 4x4 matrix, meaning the next value for the wanted line is in the next row -> +4
			// d = -50 is the scalar -> Length of the line, g: Point + d*Vector
			// p1<----Middle---->p2
			//   <-----100----->
			float dScalar = 200.f;
			p1.x = lineParams[8 + i] - dScalar * lineParams[i];
			p1.y = lineParams[12 + i] - dScalar * lineParams[4 + i];
			cv::Point2f p2;
			p2.x = lineParams[8 + i] + dScalar * lineParams[i];
			p2.y = lineParams[12 + i] + dScalar * lineParams[4 + i];
			cv::line(frame, p1, p2, CV_RGB(0, 255, 255));
			linePoints.push_back(p1);
			linePoints.push_back(p2);
		}

		cv::Point2f corners[4];
		for (size_t j = 0; j < linePoints.size(); j +=2)
		{
			cv::Point2f corner;
			bool bIntersect = FindIntersection(linePoints[j], linePoints[j + 1], linePoints[(j + 2) % linePoints.size()], linePoints[(j + 3) % linePoints.size()], corner);
			cv::circle(frame, corner, 2, CV_RGB(255, 255, 100), -1);
			corners[j / 2] = corner;
		}

		cv::Point2f targetCorners[4];
		targetCorners[0].x = -0.5; targetCorners[0].y = -0.5;
		targetCorners[1].x = 5.5; targetCorners[1].y = -0.5;
		targetCorners[2].x = 5.5; targetCorners[2].y = 5.5;
		targetCorners[3].x = -0.5; targetCorners[3].y = 5.5;
		// Create and calculate the matrix of perspective transform -> non-affine -> parallel stays not parallel
		// Homography is a matrix to describe the transformation from an image region to the 2D projected image
		cv::Mat homographyMatrix(cv::Size(3, 3), CV_32FC1);
		homographyMatrix = cv::getPerspectiveTransform(targetCorners, corners);
		cv::Mat imageMarker(cv::Size{ 6, 6 }, CV_32FC1);
		cv::warpPerspective(greyImage, imageMarker, homographyMatrix.inv(), cv::Size{ 6,6 });
		// Now we have a B/W image of a supposed Marker
		threshold(imageMarker, imageMarker, g_bw_slider_value, 255, CV_THRESH_BINARY);
		int markerId = IdentifyMarker(imageMarker);
		if (markerId < 0)
		{
			continue;
		}

		// Show the first detected marker in the image
		if (g_bIsFirstMarker) {
			imshow(g_twoDMarkerWindow, imageMarker);
			g_bIsFirstMarker = false;
		}
	}
}

int main(int argc, char** argv)
{
	const cv::String keys =
		"{ file f|| Path to video file }"
		"{ debug d|false| Show debug data & controls }"
		"{ help h usage|| Show this help message }"
		"{ delay w|10| Delay between each frame (ms) }"
		"{ bBox bb|false| Use bounding boxes }";

	const auto cmdParser = cv::CommandLineParser(argc, argv, keys);
	if (cmdParser.has("help"))
	{
		cmdParser.printMessage();
		return 0;
	}

	const cv::String debugWindowName = "Grayscaled data";
	cv::String videoFilePath = "";
	cv::String windowName = "Camera Feed";
	auto captureSrc = cv::VideoCapture();
	int thresholdValue = 50, thresholdType = 3;

	bool useVideoFile = false;
	if (cmdParser.has("file"))
	{
		videoFilePath = cmdParser.get<cv::String>("file");
		useVideoFile = true;
	}
	const bool showDebugData = cmdParser.has("debug");
	const int delay = cmdParser.get<int>("delay");
	g_bBoxes = cmdParser.get<bool>("bBox");

	if (!cmdParser.check())
	{
		cmdParser.printErrors();
		return 1;
	}

	if (useVideoFile || !captureSrc.open(0))
	{
		std::cout << "Falling back to video file!" << std::endl;

		if (!captureSrc.open(videoFilePath))
		{
			std::cout << "Couldn't open video file, aborting" << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Successfully opened video file!" << std::endl;
			windowName = "Video Feed";
		}
	}
	else
	{
		std::cout << "Successfully opened camera!" << std::endl;
	}

	cv::namedWindow(windowName);
	if (showDebugData)
	{
		cv::namedWindow(debugWindowName);
		cv::createTrackbar("Threshold value", windowName, &thresholdValue, 255, nullptr);
		cv::createTrackbar("Threshold type", windowName, &thresholdType, 5, nullptr);
		cv::createTrackbar("BW Threshold", windowName, &g_bw_slider_value, 255);
		cv::namedWindow(g_twoDMarkerWindow, CV_WINDOW_NORMAL);
		cv::resizeWindow(g_twoDMarkerWindow, 120, 120);
	}


	while (!GetAsyncKeyState(VK_ESCAPE))
	{
		cv::Mat frame, grayScaledFrame;
		// Get the next frame from the camera / video source
		captureSrc >> frame;

		if (frame.empty())
		{
			break;
		}

		GrayScaleImage(frame, grayScaledFrame, thresholdValue, thresholdType);
		auto boundingBoxes = FindMarkerShapes(grayScaledFrame);
		MarkMarkers(frame, grayScaledFrame, boundingBoxes);
		cv::imshow(windowName, frame);
		if (showDebugData)
		{
			cv::imshow(debugWindowName, grayScaledFrame);
		}
		g_bIsFirstStripe = true;
		g_bIsFirstMarker = true;

		cv::waitKey(delay);
	}

	captureSrc.release();
	cv::destroyAllWindows();
}