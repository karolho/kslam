#ifndef FRAME_HPP
#define FRAME_HPP

#include <opencv2/opencv.hpp>
#include "features.hpp"

class Frame {

    cv::Mat _image;
    int _f_idx;
    struct kpsdes_s _kpsdes;

    cv::Mat _pose;

  public:
    Frame(cv::Mat image, int f_idx);

    cv::Mat getImage();

    void setKpsAndDes(struct kpsdes_s kpdes);
    struct kpsdes_s getKpsAndDes();

    void setPose(cv::Mat pose);
    cv::Mat getPose();

};

#endif