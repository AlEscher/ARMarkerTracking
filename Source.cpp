#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

#include <Windows.h>
#include <vector>
#include <cmath>

// Struct holding all infos about each strip, e.g. length
struct MyStrip {
	int stripeLength;
	int nStop;
	int nStart;
	cv::Point2f stripeVecX;
	cv::Point2f stripeVecY;
};


bool g_bBoxes = false;

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
	const int fx = int(floorf(p.x));
	const int fy = int(floorf(p.y));

	if (fx < 0 || fx >= pSrc.cols - 1 ||
		fy < 0 || fy >= pSrc.rows - 1)
		return 127;

	// Slides 15
	const int px = int(256 * (p.x - floorf(p.x)));
	const int py = int(256 * (p.y - floorf(p.y)));

	// Here we get the pixel of the starting point
	auto i = (unsigned char*)((pSrc.data + fy * pSrc.step) + fx);

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
	SortContourPoints(shape);
	if (boundingBox.height < 20 || boundingBox.width < 20 || boundingBox.width > inputFrame.cols - 10 || boundingBox.height > inputFrame.rows - 10 || contourSize < sizeLimit)
	{
		return false;
	}

	return true;
}

std::vector<std::vector<cv::Point>> FindMarkers(const cv::Mat& input)
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

void MarkMarkers(cv::Mat& frame, const cv::Mat& greyImage, const std::vector<std::vector<cv::Point>>& boundingBoxes)
{
	for (const auto& box : boundingBoxes)
	{
		// Connect the dots of the box with lines
		cv::polylines(frame, box, true, cv::Scalar(50, 50, 255, 255), 2);
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

			// First point already rendered, now the other 6 points
			for (int j = 1; j < 7; ++j) 
			{
				// Position calculation
				const double px = static_cast<double>(box[i].x) + static_cast<double>(j) * dx;
				const double py = static_cast<double>(box[i].y) + static_cast<double>(j) * dy;

				cv::Point p;
				p.x = static_cast<int>(px);
				p.y = static_cast<int>(py);
				circle(frame, p, 2, CV_RGB(0, 0, 255), -1);

				for (int m = -1; m <= 1; m++)
				{
					for (int n = stripe.nStart; n <= stripe.nStop; n++)
					{
						cv::Point2f pFloat;
						pFloat.x = static_cast<float>(p.x);
						pFloat.y = static_cast<float>(p.y);
						cv::Point2f subPixel = pFloat + stripe.stripeVecX * static_cast<float>(n) + stripe.stripeVecY * static_cast<float>(m);

						const int pixel = subpixSampleSafe(greyImage, subPixel);
						const int w = m + 1;
						const int h = n + (stripe.stripeLength >> 1);
						imagePixelStripe.at<uchar>(h, w) = static_cast<uchar>(pixel);
					}
				}

				std::vector<double> sobelValues(stripe.stripeLength - 2);
				for (int n = 1; n < (stripe.stripeLength - 1); n++)
				{
					// Take the intensity value from the stripe 
					unsigned char* stripePtr = &(imagePixelStripe.at<uchar>(n - 1, 0));

					// Calculation of the gradient with the sobel for the first row
					const double r1 = -stripePtr[0] - 2. * stripePtr[1] - stripePtr[2];

					// r2 -> Is equal to 0 because of sobel

					// Go two lines for the thrid line of the sobel, step = size of the data type, here uchar
					stripePtr += 2 * imagePixelStripe.step;

					// Calculation of the gradient with the sobel for the third row
					const double r3 = stripePtr[0] + 2. * stripePtr[1] + stripePtr[2];

					// Writing the result into our sobel value vector
					const unsigned int ti = n - 1;
					sobelValues[ti] = r1 + r3;
				}
			}
		}
		
	}
}

int main(int argc, char** argv)
{
	const cv::String keys =
		"{ file f|| Path to video file }"
		"{ debug d|false| Show debug data & controls }"
		"{ help h usage|| Show this help message }"
		"{ delay w|25| Delay between each frame (ms) }"
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
	cv::namedWindow("Adaptive Threshold");
	if (showDebugData)
	{
		cv::namedWindow(debugWindowName);
		cv::createTrackbar("Threshold value", windowName, &thresholdValue, 255, nullptr);
		cv::createTrackbar("Threshold type", windowName, &thresholdType, 5, nullptr);
	}


	while (!GetAsyncKeyState(VK_ESCAPE))
	{
		cv::Mat frame, grayScaledFrame, adaptiveThresholdFrame;
		// Get the next frame from the camera / video source
		captureSrc >> frame;

		if (frame.empty())
		{
			break;
		}

		GrayScaleImage(frame, grayScaledFrame, thresholdValue, thresholdType);
		AdaptiveGrayScaleImage(frame, adaptiveThresholdFrame, 0, 0);
		auto boundingBoxes = FindMarkers(grayScaledFrame);
		MarkMarkers(frame, grayScaledFrame, boundingBoxes);
		cv::imshow(windowName, frame);
		cv::imshow("Adaptive Threshold", adaptiveThresholdFrame);
		if (showDebugData)
		{
			cv::imshow(debugWindowName, grayScaledFrame);
		}

		cv::waitKey(delay);
	}

	captureSrc.release();
	cv::destroyAllWindows();
}