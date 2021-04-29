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
	cv::findContours(input, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

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
				if (rect.area() > sizeLimit)
				{
					boundingBoxes.push_back({ rect.tl(), cv::Point(rect.x + rect.width, rect.y), rect.br(), cv::Point(rect.x, rect.y + rect.height) });
				}
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
		for (size_t i = 0; i < box.size(); i++)
		{
			// Go through all the points of the box
			const int endPoint = (i + 1) % box.size();
			const cv::Vec2i deltaVec = { box[endPoint] - box[i] };
			// Apparently too advanced for openCV, couldn't find any proper method
			const int length = sqrt(deltaVec[0] * deltaVec[0] + deltaVec[1] * deltaVec[1]);

			// Draw 7 equidistant circles
			cv::Scalar circleColor = { 50, 255, 50, 255 };
			size_t circlesDrawn = 0;
			size_t j = 0;
			for (auto line = cv::LineIterator(frame, box[i], box[endPoint]); circlesDrawn < 7; line++, j++)
			{
				const int numCircles = 7;
				// Avoid division by 0 in line.pos()
				if (line.step == 0)
				{
					break;
				}

				if (j % (length / numCircles) == 0)
				{
					cv::circle(frame, line.pos(), length / 50, circleColor);
					circlesDrawn++;
				}
			}
		}
	}
}

int main(int argc, char** argv)
{
	const cv::String keys =
		"{ file f|      | Path to video file }"
		"{ debug d| false| Show debug data & controls }"
		"{ help h usage|      | Show this help message }"
		"{ delay w|25    | Delay between each frame (ms) }"
		"{ bBox bb|false | Use bounding boxes }";

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

	int delay = 25;
	bool showDebugData = true;
	bool useVideoFile = false;
	if (cmdParser.has("file"))
	{
		videoFilePath = cmdParser.get<cv::String>("file");
		useVideoFile = true;
	}
	showDebugData = cmdParser.get<bool>("debug");
	g_bBoxes = cmdParser.get<bool>("bBox");
	delay = cmdParser.get<int>("delay");

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