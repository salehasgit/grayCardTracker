#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	// Create a VideoCapture object and open the input file
	VideoCapture cap(".\\..\\Tracking_GreyCard.mov");//for the first webcam, pass 0 instead of the video file name

	// Check if camera opened successfully
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	// create a VideoWriter object with the same frame size and rate
	int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int fps = cap.get(CV_CAP_PROP_FPS);
	VideoWriter video(".\\..\\tracked_Card.mkv", CV_FOURCC('P', 'I', 'M', '1'), fps, Size(frame_width, frame_height));// save in mkv format to have less trouble opening it on a Mac!

	while (1) { //go through all frames

		Mat frame;
		cap >> frame;// get next frame
		if (frame.empty())
			break;

		video.write(frame);
	}
	video.release();
	cap.release();
}