#include "frame.hpp"

Frame::Frame() {}

Frame::Frame(int f_idx)
{
    //_image = image;
    _f_idx = f_idx;
}

/*
cv::Mat Frame::getImage()
{
    return _image;
}
*/

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

void Frame::addObservation(Point3D point, int kp_idx)
{
    struct observation obs;
    obs.point = point;
    obs.f_idx = this->_f_idx;
    obs.kp_idx = kp_idx;

    pointObservations.push_back(obs);
}

std::vector<observation> Frame::getObservations()
{
    return pointObservations;
}

/*
void Frame::addPoint(Point3D point)
{
    points.push_back(point);
}
*/
