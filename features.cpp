#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "features.hpp"

using namespace std;
using namespace cv;

struct kpsdes_s extractFeatures(Mat frame)
{
    vector<Point2f> pts;
    goodFeaturesToTrack(frame, pts, 3000, 0.01, 7);

    vector<KeyPoint> kps;
    for (int i = 0; i < pts.size(); i++) {
        kps.push_back(KeyPoint(pts[i].x, pts[i].y, 20));
    }

    Mat des;
    Ptr<ORB> orb = ORB::create();
    orb->compute(frame, kps, des);

    struct kpsdes_s kpsdes;
    kpsdes.kps = kps;
    kpsdes.des = des;

    return kpsdes;
}

vector<vector<DMatch>> matchFeatures(Mat des1, Mat des2)
{
    vector<vector<DMatch>> matches;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    matcher->knnMatch(des1, des2, matches, 2);

    return matches;
}