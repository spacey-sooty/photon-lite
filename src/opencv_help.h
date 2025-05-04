// Copyright (C) Jade T 2025

#pragma once

#include <vector>

#include <frc/geometry/Rotation3d.h>
#include <frc/geometry/Transform3d.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>

#include "calibration.h"

inline static frc::Rotation3d EDN_TO_NWU = frc::Rotation3d{
    (Eigen::Matrix3d() << 0, -1, 0, 0, 0, -1, 1, 0, 0).finished()};

constexpr static float tagSize = 0.08255;
inline std::vector<cv::Point3d> tagPoints{
    cv::Point3d{-tagSize, tagSize, 0},
    cv::Point3d{tagSize, tagSize, 0},
    cv::Point3d{tagSize, -tagSize, 0},
    cv::Point3d{-tagSize, -tagSize, 0},
};

inline frc::Transform3d solvePnP_tag36h11(Calibration calib,
                                          std::vector<cv::Point2d> points) {
  cv::Mat rvec{3, 1, CV_32F};
  cv::Mat tvec{3, 1, CV_32F};
  cv::Mat calibration{3, 3, CV_64F};
  cv::eigen2cv(calib.ToMatrix(), calibration);
  cv::solvePnP(tagPoints, points, calibration, calib.distortion, rvec, tvec);

  cv::Vec3f tdata;
  cv::Mat twrapped{tvec.rows, tvec.cols, CV_32F};
  tvec.convertTo(twrapped, CV_32F);
  tdata = twrapped.at<cv::Vec3f>(cv::Point{0, 0});
  frc::Translation3d translation_unrotated{units::meter_t{tdata[0]},
                                           units::meter_t{tdata[1]},
                                           units::meter_t{tdata[2]}};
  frc::Translation3d translation = translation_unrotated.RotateBy(EDN_TO_NWU);

  // Get the 'rodriguez' (axis-angle, where the norm is the angle about the
  // normalized direction of the vector)
  cv::Vec3f rdata;
  cv::Mat rwrapped{rvec.rows, rvec.cols, CV_32F};
  rvec.convertTo(rwrapped, CV_32F);
  rdata = rwrapped.at<cv::Vec3f>(cv::Point{0, 0});
  frc::Rotation3d rotation_unrotated{
      Eigen::Vector3d{rdata[0], rdata[1], rdata[2]}};
  frc::Rotation3d rotation = rotation_unrotated.RotateBy(EDN_TO_NWU);

  return frc::Transform3d{translation, rotation};
}
