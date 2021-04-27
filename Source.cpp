#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include <Windows.h>
#include <vector>

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
	const int sizeLimit = 1000;

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
			if (rect.area() > sizeLimit)
			{
				boundingBoxes.push_back({ rect.tl(), cv::Point(rect.x + rect.width, rect.y), rect.br(), cv::Point(rect.x, rect.y + rect.height) });
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
			const cv::Vec2i deltaVecNorm = deltaVec / length;

			// Draw 7 equidistant circles
			cv::Scalar circleColor = { 50, 255, 50, 255 };
			const int numCircles = 7;
			size_t circlesDrawn = 0;
			size_t j = 0;
			for (auto line = cv::LineIterator(frame, box[i], box[endPoint]); circlesDrawn < 7; line++, j++)
			{
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

int main()
{
	cv::String windowName = "Camera Feed";
	cv::String videoName = R"(C:\Dev\Cpp\ARMarkerTracking\ARMarkerTracking\MarkerMovie.MP4)";
	cv::VideoCapture captureSrc = cv::VideoCapture();
	double fps = 30.0;
	// Set to false in order to use webcam
	bool useVideo = true;
	int thresholdValue = 80, thresholdType = 3;

	if (useVideo || !captureSrc.open(0))
	{
		std::cout << "Falling back to video file!" << std::endl;

		if (!captureSrc.open(videoName))
		{
			std::cout << "Couldn't open video file, aborting" << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Successfully opened video file!" << std::endl;
			windowName = "Video Feed";
			// Get video's FPS
			fps = captureSrc.get(cv::CAP_PROP_FPS);
		}
	}
	else
	{
		std::cout << "Successfully opened camera!" << std::endl;
	}

	cv::namedWindow(windowName);
	cv::createTrackbar("Threshold value", windowName, &thresholdValue, 255, nullptr);
	cv::createTrackbar("Threshold type", windowName, &thresholdType, 5, nullptr);

	while (!GetAsyncKeyState(VK_ESCAPE))
	{
		cv::Mat frame, result, grayScaledFrame;
		// Get the next frame from the camera / video source
		captureSrc >> frame;

		if (frame.empty())
		{
			break;
		}

		GrayScaleImage(frame, result, thresholdValue, thresholdType);
		auto boundingBoxes = FindMarkers(result);
		MarkMarkers(frame, boundingBoxes);
		cv::imshow(windowName, frame);
		cv::waitKey(25);
	}

	captureSrc.release();
	cv::destroyAllWindows();
}