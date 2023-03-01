#ifndef POINT_HPP
#define POINT_HPP

#include <opencv2/core.hpp>

struct observation;

class Point3D {

    cv::Point3f _loc;

    std::vector<observation> observations;

  public:
      Point3D();
      Point3D(cv::Point3f loc);

      cv::Point3f getLoc();

      void setIdx(int p_idx);

      void addObservation(int f_idx, int kp_idx);
      std::vector<observation> getObservations();
};

struct observation {
    Point3D point;
    int f_idx;
    int kp_idx;
};

#endif