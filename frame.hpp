#ifndef FRAME_HPP
#define FRAME_HPP

#include <opencv2/opencv.hpp>
#include "features.hpp"
#include "point.hpp"

class Frame {

    //cv::Mat _image;
    int _f_idx;
    struct kpsdes_s _kpsdes;

    cv::Mat _pose;

    std::vector<observation> pointObservations;
    //std::vector<Point> points;

  public:
    Frame();
    Frame(int f_idx);

    cv::Mat getImage();

    void setKpsAndDes(struct kpsdes_s kpdes);
    struct kpsdes_s getKpsAndDes();

    void setPose(cv::Mat pose);
    cv::Mat getPose();

    void addObservation(Point3D point, int kp_idx);
    std::vector<observation> getObservations();

    //void addPoint(Point3D point);

};

#endif