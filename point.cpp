#include "point.hpp"

using namespace std;

Point3D::Point3D() {}

Point3D::Point3D(cv::Point3f loc)
{
    _loc = loc;
}

cv::Point3f Point3D::getLoc()
{
    return _loc;
}

void Point3D::addObservation(int f_idx, int kp_idx)
{
    struct observation obs;
    obs.point = *this;
    obs.f_idx = f_idx;
    obs.kp_idx = kp_idx;

    observations.push_back(obs);
}

vector<observation> Point3D::getObservations() {
    return observations;
}
