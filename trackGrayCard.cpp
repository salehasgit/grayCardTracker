#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

using namespace std;
using namespace cv;

void detect_rectangles(Mat &in, Rect2d &cardBBox)
{
	Mat grayImg, grayThresholded ;
	cvtColor(in, grayImg, COLOR_BGR2GRAY);
	threshold(grayImg, grayThresholded, 100, 255, THRESH_BINARY);

	/// Find contours
	vector<vector<Point> > contoursList;
	findContours(grayThresholded, contoursList, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));

	//iterate through contours and find all big rectangles that have white or gray infill
	vector<Point> contour;
	vector<RotatedRect> rbboxList;
	RotatedRect rbbox;
	for (int i = 0; i< contoursList.size(); i++)
	{
		//simplify the contour
		approxPolyDP(contoursList[i], contour, 3, true);

		if (contour.size() == 4)// is a quadrilateral?
		{
			if (contourArea(contour) > 250)// get ride of small rectangles
			{
				Mat mask = Mat::zeros(in.rows, in.cols, CV_8UC1);
				fillConvexPoly(mask, contour, Scalar(255, 255, 255));
				erode(mask, mask, Mat(), Point(-1, -1), 2);// erode a bit to get ride of borders

				//Scalar m = mean(in, mask);// find the average color

				cv::Mat masked;
				int maskArea = countNonZero(mask);
				grayImg.copyTo(masked, mask);
				int countGray = countNonZero((masked > 110) & (masked < 150));
				int countWhite = countNonZero(masked > 220);
				if (countGray/ maskArea > .95 | countWhite / maskArea > .95)// gray or white infills?
				{
					rbbox = minAreaRect(contour);//Finds minimum area rotated rectangle bounding a set of points (a rotated rectangle)
					rbboxList.push_back(rbbox);

					//draw the quadrilateral for visualization
					line(in, contour[0], contour[1], cvScalar(0, 255, 255), 4);
					line(in, contour[1], contour[2], cvScalar(0, 255, 255), 4);
					line(in, contour[2], contour[3], cvScalar(0, 255, 255), 4);
					line(in, contour[3], contour[0], cvScalar(0, 255, 255), 4);
				}

			}

		}
	}

	// for time being, we find first two rectangles which are up-right and very close to each other
	bool found = false;
	for (int i = 0; i < rbboxList.size() & ~found; i++)
		for (int j = i + 1; j < rbboxList.size() & ~found; j++)
			if (abs(rbboxList[i].angle) < 2 & // up-right?
				abs(rbboxList[j].angle) < 2 &
				abs(rbboxList[i].size.width - rbboxList[j].size.width) < 10 & //and same size?
				abs(rbboxList[i].size.height - rbboxList[j].size.height) < 10)
			{
				// if the area of the bbox of two rectangles is the same as the sum of their areas, it means that they are touching and well aligned 
				// find the bbox containing two given rectangles
				cardBBox = Rect2d(rbboxList[i].boundingRect());
				cardBBox |= Rect2d(rbboxList[j].boundingRect());
				if (cardBBox.area() - rbboxList[i].size.area() - rbboxList[i].size.area() < 1500)
					++found;
			}

	/*
	Further verifications for robustness:
	- the quadrilaterals are actually rectangles (parallel sides)?
	- the white rect is on top of the gray rect?
	- they are aligned vertically?
	*/

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

	// create a tracker obj																											  // Create a tracker
	Ptr<Tracker> tracker;
	tracker = TrackerKCF::create();// or TrackerMedianFlow
	Rect2d bbox;// bounding box of the graycard in the first frame

	bool is_first_frame = true;
	Mat frame_temp;
	while (1) { //go through all frames

		Mat frame;
		cap >> frame;// get next frame
		if (frame.empty())
			break;

		frame.copyTo(frame_temp);
		if (is_first_frame) 
		{
			//detect two rectangles
			detect_rectangles(frame, bbox);//TODO: what if bbox is empty
			if(~bbox.empty())
				rectangle(frame, bbox, Scalar(128, 128, 0), 3, 1);	// Display bbox.

			// some extra experiments on the first frame for locating the card
			{
				Mat gray;
				cvtColor(frame, gray, COLOR_BGR2GRAY);
				// filtering based on the gray level
				if (0)
				{
					vector<Mat>bgr(3);
					split(frame_temp, bgr);
					int blackThresh = 40;
					int whiteThresh = 200;
					int WBgain = 10;
					Mat mask0 = (abs(bgr[1] - bgr[0]) < WBgain) & (abs(bgr[1] - bgr[2]) < WBgain) &(abs(bgr[1] - bgr[2]) < WBgain) & bgr[0] < 80 & bgr[1] < blackThresh & bgr[2] < blackThresh;
					Mat mask1 = (abs(bgr[1] - bgr[0]) < WBgain) & (abs(bgr[1] - bgr[2]) < WBgain) &(abs(bgr[1] - bgr[2]) < WBgain) & bgr[0] > whiteThresh & bgr[1] > whiteThresh & bgr[2] > whiteThresh;
					Mat mask01 = (abs(bgr[1] - bgr[0]) < WBgain) & (abs(bgr[1] - bgr[2]) < WBgain) &(abs(bgr[1] - bgr[2]) < WBgain) & bgr[0] < whiteThresh & bgr[1] < whiteThresh & bgr[2] < whiteThresh & bgr[0] > blackThresh & bgr[1] > blackThresh & bgr[2] >blackThresh;
					Mat maskGray = (abs(bgr[1] - bgr[0]) < WBgain) & (abs(bgr[1] - bgr[2]) < WBgain) &(abs(bgr[1] - bgr[2]) < WBgain);
				}

				// simple average of Scharr gradients
				if (0)
				{
					Mat gradX, gradY, gradient, gradientblured;
					// compute the Scharr gradient magnitude in x and y directions
					Sobel(gray, gradX, CV_32F, 1, 0, -1);
					Sobel(gray, gradY, CV_32F, 0, 1, -1);

					//By subtraction of the gradients in x and y directions, we can detect regions of the image that have gray color (are flat)
					subtract(gradY, gradX, gradient);
					convertScaleAbs(gradient, gradient);
					blur(gradient, gradientblured, Size(3, 3));
					threshold(gradientblured, gradientblured, 120, 255, THRESH_BINARY);

					reduce(gradientblured, gradientblured, 1, CV_REDUCE_AVG);
				}

				// 1D cross-correlation
				if (1)
				{
					Mat cols_concat, step_val;

					//build the template
					Mat stairs_sig = Mat::zeros(90, 1, CV_8U);
					repeat(250, 30, 1, step_val);
					step_val.copyTo(stairs_sig.rowRange(0, 30));
					repeat(128, 30, 1, step_val);
					step_val.copyTo(stairs_sig.rowRange(30, 60));
					repeat(5, 30, 1, step_val);
					step_val.copyTo(stairs_sig.rowRange(60, 90));

					//scan each 10th column
					for (int x = 0; x < frame.cols; ++x) {

						frame.copyTo(frame_temp);
						line(frame_temp, Point(x, 0), Point(x, frame.rows), Scalar(0, 255, 0), 1);

						Mat col_curr;
						gray.col(x).copyTo(col_curr);
						blur(col_curr, col_curr, Size(1, 7));

						Mat col_curr_temp;
						col_curr.copyTo(col_curr_temp);

						// do template-matching
						int match_method = 5;// Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
						int result_cols = col_curr_temp.cols - stairs_sig.cols + 1;
						int result_rows = col_curr_temp.rows - stairs_sig.rows + 1;
						Mat result;
						result.create(result_rows, result_cols, CV_32FC1);
						matchTemplate(col_curr_temp, stairs_sig, result, match_method);

						normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
						double minVal; double maxVal; Point minLoc; Point maxLoc;
						Point matchLoc;
						minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
						if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
							matchLoc = minLoc;
						else
							matchLoc = maxLoc;

						int xx = matchLoc.x;
						int yy = matchLoc.y + 45;

						// plot the loc of the best match
						circle(frame_temp, Point(x, yy), 3, Scalar(255, 255, 0), -1);

						//plot the col
						uchar pixel1, pixel0;
						pixel0 = col_curr.at<uchar>(0);
						Point p0(pixel0, 0);
						for (int i = 1; i < col_curr.rows; ++i) {
							pixel1 = col_curr.at<uchar>(i);
							Point p1(pixel1, i);
							line(frame_temp, p0, p1, Scalar(255, 0, 0), 1);
							p0 = p1;
						}

						if (cols_concat.empty())
							col_curr.copyTo(cols_concat);
						else
							hconcat(cols_concat, col_curr, cols_concat);

						Mat hist;
						reduce(cols_concat, hist, 1, CV_REDUCE_AVG);
						pixel0 = hist.at<uchar>(0);
						p0 = Point(pixel0, 0);
						for (int i = 1; i < hist.rows; ++i) {
							pixel1 = hist.at<uchar>(i);
							Point p1(pixel1, i);
							line(frame_temp, p0, p1, Scalar(0, 0, 255), 1);
							p0 = p1;
						}

						video.write(frame_temp);
					}
				}

			}

			//Now that we have bbox, we can track it from now on
			tracker->init(frame, bbox);
			is_first_frame = false;
		}
		else if(~bbox.empty() & tracker->update(frame, bbox) )// track the card in this frame and if succeeded, draw its bbox
			rectangle(frame_temp, bbox, Scalar(128, 128, 0), 3, 1);
		else // if tracking failed
			putText(frame_temp, "Tracking failed...", Point(100, 80), FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 0, 255), 2);

		// one more experiment! 
		// finding the center of gray rectangle based on its color, assuming the gray stripe is the biggest gray region in each frame
		if (1)
		{
			Mat mask;
			inRange(frame, Scalar(110, 110, 110), Scalar(140, 140, 140), mask);// find pixels in the range [110 140]

			//find the biggest blob
			vector<Point> blob;
			float maxArea = 0;
			vector<vector<Point> > contoursList;
			findContours(mask, contoursList, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
			for_each(contoursList.begin(), contoursList.end(), [&blob, &maxArea](vector<Point> el) { if (contourArea(el) > maxArea) { maxArea = contourArea(el); blob = el; } });

			// find center
			Moments M = moments(blob);
			Point center = Point(int(M.m10 / M.m00), int(M.m01 / M.m00));

			// drow a circle at center
			circle(frame_temp, center, 3, Scalar(255, 255, 255), -1);
		}

		video.write(frame_temp);
	}
	video.release();
	cap.release();
}