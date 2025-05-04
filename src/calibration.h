// Copyright (C) Jade T 2025

#pragma once

#include <Eigen/Core>
#include <wpi/array.h>

struct Calibration {
  double fx;
  double fy;
  double cx;
  double cy;
  wpi::array<double, 5> distortion;

  Eigen::Matrix<double, 3, 3> ToMatrix() {
    return Eigen::Matrix<double, 3, 3>{
        {fx, 0, cx},
        {0, fy, cy},
        {0, 0, 1},
    };
  }
};
