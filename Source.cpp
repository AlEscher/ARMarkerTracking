#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>

#include <Windows.h>

void GrayScaleImage(const cv::Mat& frame, cv::Mat& result, int thresholdValue, int thresholdType)
{
	cv::Mat grayScaledFrame;
	cv::cvtColor(frame, grayScaledFrame, CV_BGR2GRAY);
	cv::threshold(grayScaledFrame, result, thresholdValue, 255, thresholdType);
}

int main()
{
	cv::String windowName = "Camera Feed";
	cv::String videoName = R"(C:\Dev\Cpp\ARMarkerTracking\ARMarkerTracking\MarkerMovie.MP4)";
	cv::VideoCapture captureSrc = cv::VideoCapture();
	double fps = 30.0;
	bool useVideo = true;
	int thresholdValue = 0, thresholdType = 3;

	if (useVideo || !captureSrc.open(0))
	{
		std::cout << "Using video file!" << std::endl;

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
	//cv::createTrackbar("Threshold type", windowName, &thresholdType, 5, nullptr);

	while (!GetAsyncKeyState(VK_ESCAPE))
	{
		cv::Mat frame, result, grayScaledFrame;
		captureSrc >> frame;

		if (frame.empty())
		{
			break;
		}

		GrayScaleImage(frame, result, thresholdValue, thresholdType);
		cv::imshow(windowName, result);
		cv::waitKey(25);
	}

	captureSrc.release();
	cv::destroyAllWindows();
}