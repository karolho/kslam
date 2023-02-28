#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include "viewer.hpp"
#include "features.hpp"
#include "frame.hpp"

#define F 525 // focal length

using namespace std;
using namespace cv;

Mat dbg_frame;

int width;
int height;
Mat K;

vector<Frame> frames;

int processFrame(Mat frame, int f_idx)
{
    Mat out;
    Frame newFrame(frame, f_idx);
    struct kpsdes_s kpsdes;

    cout << "Frame " << f_idx << endl;
    
    kpsdes = extractFeatures(frame);
    newFrame.setKpsAndDes(kpsdes);
    drawKeypoints(frame, kpsdes.kps, out, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    if (f_idx < 1) {
        newFrame.setPose(Mat::eye(4, 4, CV_64FC1));
        frames.push_back(newFrame);
        dbg_frame = out;
        return 0;
    }

    Frame lastFrame = frames[frames.size()-1];

    vector<vector<DMatch>> matches1;
    vector<vector<DMatch>> matches2;
    vector<DMatch> good_matches;
    vector<DMatch> temp;

    // Match in both directions
    matches1 = matchFeatures(kpsdes.des, lastFrame.getKpsAndDes().des);
    matches2 = matchFeatures(lastFrame.getKpsAndDes().des, kpsdes.des);

    // Filtering
    for (int i = 0; i < matches1.size(); i++) {
        if (matches1[i][0].distance < matches1[i][1].distance * 0.7 && matches1[i][0].distance < 32) {
            good_matches.push_back(matches1[i][0]);
        }
    }

    for (int i = 0; i < matches2.size(); i++) {
        if (matches2[i][0].distance < matches2[i][1].distance * 0.7 && matches2[i][0].distance < 32) {
            good_matches.push_back(DMatch(matches2[i][0].trainIdx, matches2[i][0].queryIdx, matches2[i][0].distance));
        }
    }
    
    for (int i = 0; i < good_matches.size(); i++) {
        for (int j = 0; j < good_matches.size(); j++) {
            if (j != i) {
                if (good_matches[j].queryIdx == good_matches[i].queryIdx && good_matches[j].trainIdx == good_matches[i].trainIdx) {
                    temp.push_back(good_matches[i]);
                }
            }
        }
    }

    good_matches.clear();
    good_matches = temp;
    temp.clear();

    vector<Point2f> pts1;
    vector<Point2f> pts2;

    if (good_matches.size() < 8) {
        cout << "Frame " << f_idx << ": < 8 matches" << endl;
        return -1;
    }

    for (int i = 0; i < good_matches.size(); i++) {
        pts1.push_back(kpsdes.kps[good_matches[i].queryIdx].pt);
        pts2.push_back(lastFrame.getKpsAndDes().kps[good_matches[i].trainIdx].pt);
    }

    Mat mask;
    Mat H = findHomography(pts1, pts2, mask, RANSAC, 5.0);

    pts1.clear();
    pts2.clear();

    for (int i = 0; i < good_matches.size(); i++) {
        if (!(unsigned int)mask.at<uchar>(i, 0)) {
            temp.push_back(good_matches[i]);
            pts1.push_back(kpsdes.kps[good_matches[i].queryIdx].pt);
            pts2.push_back(lastFrame.getKpsAndDes().kps[good_matches[i].trainIdx].pt);
        }
    }

    good_matches.clear();
    good_matches = temp;
    temp.clear();

    cout << good_matches.size() << " matches" << endl;

    for (int i = 0; i < good_matches.size(); i++) {
        cv::line(out, kpsdes.kps[good_matches[i].queryIdx].pt, lastFrame.getKpsAndDes().kps[good_matches[i].trainIdx].pt, Scalar(255, 0, 0));
    }

    Mat mask_E, R, t, Rt;
    Mat E = findEssentialMat(pts1, pts2, K, Mat(), K, Mat(), LMEDS, 0.999, 5.0, mask_E);
    recoverPose(E, pts1, pts2, K, R, t, mask_E);

    hconcat(R, t, Rt);
    Mat row = (Mat_<double>(1, 4) << 0, 0, 0, 1);
    vconcat(Rt, row, Rt);

    Rt *= lastFrame.getPose();
    newFrame.setPose(Rt);

    cout << newFrame.getPose() << endl;

    dbg_frame = out;
    frames.push_back(newFrame);

    return 0;
}

void runSLAM(VideoCapture cap)
{
    Mat frame;
    int f_idx = 0;
    while (1) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        processFrame(frame, f_idx);

        f_idx++;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        cout << "Please provide a video file" << endl;
        return 1;
    }

    VideoCapture cap(argv[1]);
    Mat frame;

    if (!cap.isOpened()) {
        cout << "Could not open file" << endl;
        return 1;
    }

    width = cap.get(CAP_PROP_FRAME_WIDTH);
    height = cap.get(CAP_PROP_FRAME_HEIGHT);

    // Camera intrinsic matrix
    K = (Mat_<double>(3, 3) << F, 0, width/2, 0, F, height/2, 0, 0, 1);

    thread slamThread(runSLAM, cap);

#ifndef HEADLESS
    runViewer();
#endif

    slamThread.join();

    return 0;
}
