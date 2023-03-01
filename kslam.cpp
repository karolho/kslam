#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include "viewer.hpp"
#include "features.hpp"
#include "frame.hpp"
#include "point.hpp"

#define F 500 // focal length

using namespace std;
using namespace cv;

Mat dbg_frame;

int width;
int height;
Mat K;

vector<Frame> frames;
vector<Point3D> points;

int processFrame(Mat frame, int f_idx)
{
    Mat out;
    Frame newFrame(f_idx);
    struct kpsdes_s kpsdes;

    cout << "Frame " << f_idx << endl;
    
    kpsdes = extractFeatures(frame);
    newFrame.setKpsAndDes(kpsdes);
    drawKeypoints(frame, kpsdes.kps, out, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    if (f_idx < 1) {
        for (int i = 0; i < kpsdes.kps.size(); i++) {
            //Point3D p;
            //p.addObservation(f_idx, i);
            //newFrame.addPoint(p);
            //newFrame.addObservation(p, i);
        }
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
    
    for (int i = 0; i < good_matches.size(); i++) {
        vector<observation> observations = lastFrame.getObservations();
        for (int j = 0; j < observations.size(); j++) {
            if (observations[j].kp_idx == good_matches[i].trainIdx) {
                //observations[j].point.addObservation(f_idx, good_matches[i].queryIdx);
                newFrame.addObservation(observations[j].point, good_matches[i].queryIdx);
                //newFrame.addPoint(observations[j].point);
            }
        }
    }

    cout << good_matches.size() << " matches" << endl;    

    for (int i = 0; i < good_matches.size(); i++) {
        cv::line(out, kpsdes.kps[good_matches[i].queryIdx].pt, lastFrame.getKpsAndDes().kps[good_matches[i].trainIdx].pt, Scalar(255, 0, 0));
    }

    Mat mask_E, R, t, Rt;
    Mat E = findEssentialMat(pts1, pts2, K, Mat(), K, Mat(), RANSAC, 0.999, 5.0, mask_E);
    recoverPose(E, pts1, pts2, K, R, t, mask_E);

    hconcat(R, t, Rt);
    Mat row = (Mat_<double>(1, 4) << 0, 0, 0, 1);
    vconcat(Rt, row, Rt);

    Rt *= lastFrame.getPose();
    newFrame.setPose(Rt);

    // triangulate points
    int count = 0;
    for (int i = 0; i < good_matches.size(); i++) {
        cnt:;

        vector<observation> lastObs = lastFrame.getObservations();
        for (int j = 0; j < lastObs.size(); j++) {
            if (lastObs[j].kp_idx == good_matches[i].trainIdx) {
                newFrame.addObservation(lastObs[j].point, good_matches[i].queryIdx);
                i++;
                goto cnt;
            }
        }

        Point2f p1 = lastFrame.getKpsAndDes().kps[good_matches[i].trainIdx].pt;
        Point2f p2 = newFrame.getKpsAndDes().kps[good_matches[i].queryIdx].pt;
        Point2f tr_points[2] = {p1, p2};

        Mat P1 = K * lastFrame.getPose()(cv::Rect(0, 0, 4, 3));
        Mat P2 = K * newFrame.getPose()(cv::Rect(0, 0, 4, 3));
        Mat PMats[2] = {P1, P2};

        Matx<double, 4, 4> A;
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4; k++) {
                A(j*2, k) = tr_points[j].x * PMats[j].at<double>(2, k) - PMats[j].at<double>(0, k);
                A(j*2+1, k) = tr_points[j].y * PMats[j].at<double>(2, k) - PMats[j].at<double>(1, k);
            }
        }

        Matx<double, 4, 1> w;
        Matx<double, 4, 4> u;
        Matx<double, 4, 4> vt;
        SVD::compute(A, w, u, vt);

        for (int j = 0; j < 3; j++) {
            vt(3, j) /= vt(3, 3);
        }

        Point3D p(Point3f(-vt(3, 0), vt(3, 1), -vt(3, 2))); // why -?
        points.push_back(p);
        lastFrame.addObservation(p, good_matches[i].trainIdx);
        newFrame.addObservation(p, good_matches[i].queryIdx);

        count++;
    }

    cout << "Adding " << count << " new points" << endl;
    cout << points.size() << " points total in map" << endl;

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

        cvtColor(frame, frame, COLOR_BGR2GRAY);

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
