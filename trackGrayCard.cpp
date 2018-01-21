#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// this func is adapted from https://stackoverflow.com/questions/15277323/opencv-shape-detection
void detect_rectangles(Mat &in)
{
	IplImage* img = cvCloneImage(&(IplImage)in);

	//converting the original image into grayscale
	IplImage* imgGrayScale = cvCreateImage(cvGetSize(img), 8, 1);
	cvCvtColor(img, imgGrayScale, CV_BGR2GRAY);

	//thresholding the grayscale image to get better results
	cvThreshold(imgGrayScale, imgGrayScale, 100, 255, CV_THRESH_BINARY);

	CvSeq* contours;  //hold the pointer to a contour in the memory block
	CvSeq* result;   //hold sequence of points of a contour
	CvMemStorage *storage = cvCreateMemStorage(0); //storage area for all contours

	//finding all contours in the image
	cvFindContours(imgGrayScale, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	//iterating through each contour
	while (contours)
	{
		//obtain a sequence of points of contour, pointed by the variable 'contour'
		result = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);

		//if there are 4 vertices in the contour(It should be a quadrilateral)
		if (result->total == 4)
		{
			/*
			verifications :
			- the quadrilateral is actually a rectangle?
			- it has white or gray color?
			*/

			if (cvContourPerimeter(contours) > 250) // !!!!!!! so much for the verification! maybe later :)
			{
				//iterating through each point
				CvPoint *pt[4];
				for (int i = 0; i < 4; i++) {
					pt[i] = (CvPoint*)cvGetSeqElem(result, i);
				}

				//drawing lines around the quadrilateral
				cvLine(img, *pt[0], *pt[1], cvScalar(0, 255, 0), 4);
				cvLine(img, *pt[1], *pt[2], cvScalar(0, 255, 0), 4);
				cvLine(img, *pt[2], *pt[3], cvScalar(0, 255, 0), 4);
				cvLine(img, *pt[3], *pt[0], cvScalar(0, 255, 0), 4);
			}

		}
		contours = contours->h_next;
	}

	/*
	Further verifications before returning:
	- they have almost the same size?
	- the white rect is on top of the gray rect?
	- they are aligned vertically?
	*/

	// return the center of the gray rect
	cvarrToMat(img).copyTo(in);// !!! sometimes soon! :)
}

int main(int argc, char** argv)
{
	// Create a VideoCapture object and open the input file
	// TODO: more probably relative path won't work on Mac!
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
	// TODO: more probably relative path won't work on Mac!
	VideoWriter video(".\\..\\tracked_Card.mkv", CV_FOURCC('P', 'I', 'M', '1'), fps, Size(frame_width, frame_height));// save in mkv format to have less trouble opening it on a Mac!

	while (1) { //go through all frames

		Mat frame;
		cap >> frame;// get next frame
		if (frame.empty())
			break;

		detect_rectangles(frame);

		/*
		once the grey card is localized in the first frame, finding the centroid and tracking it in the rest of the frames is
		trivial(I suggest a medianflow tracker, which is perfect for slow motion and non - occlusion scenarios which is the case here).
		*/

		video.write(frame);
	}
	video.release();
	cap.release();
}