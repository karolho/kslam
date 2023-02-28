#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "features.hpp"

using namespace std;
using namespace cv;

struct kpsdes_s extractFeatures(Mat frame)
{
    Ptr<ORB> orb = ORB::create(4000);
    struct kpsdes_s kpsdes;

    orb->detect(frame, kpsdes.kps);
    orb->compute(frame, kpsdes.kps, kpsdes.des);

    return kpsdes;
}

vector<vector<DMatch>> matchFeatures(Mat des1, Mat des2)
{
    vector<vector<DMatch>> matches;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    matcher->knnMatch(des1, des2, matches, 2);

    return matches;
}