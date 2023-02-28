#ifndef FEATURES_HPP
#define FEATURES_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

struct kpsdes_s {
    std::vector<cv::KeyPoint> kps;
    cv::Mat des;
};

struct kpsdes_s extractFeatures(cv::Mat frame);
std::vector<std::vector<cv::DMatch>> matchFeatures(cv::Mat des1, cv::Mat des2);

#endif