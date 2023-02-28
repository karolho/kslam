#include "frame.hpp"

Frame::Frame(cv::Mat image, int f_idx)
{
    _image = image;
    _f_idx = f_idx;
}

cv::Mat Frame::getImage()
{
    return _image;
}

void Frame::setKpsAndDes(struct kpsdes_s kpsdes)
{
    _kpsdes = kpsdes;
}

struct kpsdes_s Frame::getKpsAndDes()
{
    return _kpsdes;
}

void Frame::setPose(cv::Mat pose)
{
    _pose = pose;
}

cv::Mat Frame::getPose()
{
    return _pose;
}
