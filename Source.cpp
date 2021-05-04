#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

#include <Windows.h>
#include <vector>


bool g_bBoxes = false;

void GrayScaleImage(const cv::Mat& frame, cv::Mat& result, int thresholdValue, int thresholdType)
{
	cv::Mat grayScaledFrame;
	// Convert into gray image
	cv::cvtColor(frame, grayScaledFrame, CV_BGR2GRAY);
	// Apply threshold from trackbar
	cv::threshold(grayScaledFrame, result, thresholdValue, 255, thresholdType);
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
			if (g_bBoxes)
			{
				const int sizeLimit = 1000;
				// Create and save a bounding box around our rectangle
				cv::Rect rect = cv::boundingRect(approximation);
				// Filter out small rectangles that are likely just noise
				if (rect.height < 20 || rect.width < 20 || rect.width > input.cols - 10 || rect.height > input.rows - 10 || rect.area() < sizeLimit) 
				{
					continue;
				}
				boundingBoxes.push_back({ rect.tl(), cv::Point(rect.x + rect.width, rect.y), rect.br(), cv::Point(rect.x, rect.y + rect.height) });
			}
			else
			{
				boundingBoxes.push_back(approximation);
			}
		}
	}

	return boundingBoxes;
}

void MarkMarkers(cv::Mat& frame, const std::vector<std::vector<cv::Point>>& boundingBoxes)
{
	for (const auto& box : boundingBoxes)
	{
		// Connect the dots of the box with lines
		cv::polylines(frame, box, true, cv::Scalar(50, 50, 255, 255));
		// Draw circles on the lines
		for (size_t i = 0; i < box.size(); ++i) 
		{
			// Render the corners, 3 -> Radius, -1 filled circle
			circle(frame, box[i], 3, CV_RGB(0, 255, 0), -1);

			// Euclidic distance, 7 -> parts, both directions dx and dy
			const double dx = (static_cast<double>(box[(i + 1) % 4].x) - static_cast<double>(box[i].x)) / 7.0;
			const double dy = (static_cast<double>(box[(i + 1) % 4].y) - static_cast<double>(box[i].y)) / 7.0;

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
		"{ bBox bb|true| Use bounding boxes }";

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
	int thresholdValue = 80, thresholdType = 3;

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
		auto boundingBoxes = FindMarkers(grayScaledFrame);
		MarkMarkers(frame, boundingBoxes);
		cv::imshow(windowName, frame);
		if (showDebugData)
		{
			cv::imshow(debugWindowName, grayScaledFrame);
		}

		cv::waitKey(delay);
	}

	captureSrc.release();
	cv::destroyAllWindows();
}