#include <apriltag.h>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <ostream>
#include <print>
#include <tag36h11.h>
#include <vector>
#include <Eigen/Core>
#include <sleipnir/optimization/problem.hpp>
#define OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT
#include <opencv2/core/eigen.hpp>

int main(void) {
  // camera calibration TODO
  constexpr double fx = 600;
  constexpr double fy = 600;
  constexpr double cx = 300;
  constexpr double cy = 150;
  Eigen::Matrix<double, 3, 3> A{
    {fx, 0, cx},
    {0, fy, cy},
    {0, 0, 1},
  };

  float tagSize = 0.08255;
  std::vector<cv::Point3d> tagPoints{
    cv::Point3d{-tagSize, tagSize, 0},
    cv::Point3d{tagSize, tagSize, 0},
    cv::Point3d{tagSize, -tagSize, 0},
    cv::Point3d{-tagSize, -tagSize, 0},
  };

  cv::Mat currentFrame;

  cv::VideoCapture capture;
  int deviceID = 0;
  int apiID = cv::CAP_ANY;
  capture.open(deviceID, apiID);

  apriltag_detector_t *td = apriltag_detector_create();
  apriltag_family_t *tf = tag36h11_create();
  apriltag_detector_add_family(td, tf);

  td->quad_decimate = 1;
  td->nthreads = 4;
  td->debug = false;
  td->refine_edges = true;

  if (!capture.isOpened()) {
    std::println("[error] camera {} couldn't be opened.", deviceID);
    return -1;
  }

  cv::Mat gray;
  for (;;) {
    slp::Problem problem = slp::Problem();
    slp::Variable robot_x = problem.decision_variable();
    slp::Variable robot_y = problem.decision_variable();
    slp::Variable robot_theta = problem.decision_variable();
    double robot_z = 0;

    auto cosTheta = slp::cos(robot_theta);
    auto sinTheta = slp::sin(robot_theta);

    slp::VariableMatrix fieldToRobot = {
      {cosTheta, -sinTheta, 0, robot_x},
      {sinTheta, cosTheta, 0, robot_y},
      {0, 0, 1, robot_z},
      {0, 0, 0, 1},
    };

    capture.read(currentFrame);
    cv::cvtColor(currentFrame, gray, cv::COLOR_BGR2GRAY);

    image_u8_t im = {gray.cols, gray.rows, gray.cols, gray.data};
    zarray_t *detections = apriltag_detector_detect(td, &im);

    if (currentFrame.empty()) {
      std::println("[error] grabbed empty currentFrame.");
      break;
    }

    std::vector<cv::Point2d> points{};

    // Draw detection outlines
    for (int i = 0; i < zarray_size(detections); i++) {
      apriltag_detection_t *det;
      zarray_get(detections, i, &det);
      points.emplace_back(cv::Point2d{det->p[0][0], det->p[0][1]});
      points.emplace_back(cv::Point2d{det->p[1][0], det->p[1][1]});
      points.emplace_back(cv::Point2d{det->p[3][0], det->p[3][1]});
      points.emplace_back(cv::Point2d{det->p[2][0], det->p[2][1]});
      line(currentFrame, cv::Point(det->p[0][0], det->p[0][1]),
           cv::Point(det->p[1][0], det->p[1][1]), cv::Scalar(0, 0xff, 0), 2);
      line(currentFrame, cv::Point(det->p[0][0], det->p[0][1]),
           cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0, 0, 0xff), 2);
      line(currentFrame, cv::Point(det->p[1][0], det->p[1][1]),
           cv::Point(det->p[2][0], det->p[2][1]), cv::Scalar(0xff, 0, 0), 2);
      line(currentFrame, cv::Point(det->p[2][0], det->p[2][1]),
           cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0xff, 0, 0), 2);

      std::stringstream ss;
      ss << det->id;
      cv::String text = ss.str();
      int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
      double fontscale = 1.0;
      int baseline;
      cv::Size textsize =
          cv::getTextSize(text, fontface, fontscale, 2, &baseline);
      putText(currentFrame, text,
              cv::Point(det->c[0] - textsize.width / 2.0,
                        det->c[1] + textsize.height / 2.0),
              fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
    }

    if (points.size() >= 4) {
      cv::Mat rvec{3, 3, CV_64F};
      cv::Mat tvec{3, 3, CV_64F};
      std::vector<double> distCoeffs{0.02860487071331241, 0.009126602251891335, 0.0019540088117773633, -0.003596010527440653, -0.13863285564042926, -0.0016559347518291224, -4.017061418060786, 0.005592461908791266};
      cv::Mat calibration{3, 3, CV_64F};
      cv::eigen2cv(A, calibration);
      cv::solvePnP(tagPoints, points, calibration, distCoeffs, rvec, tvec);
      std::cout << "Rotation = " << std::endl << " " << rvec << std::endl;
      std::cout << "Translation = " << std::endl << " " << tvec << std::endl;
    }

    apriltag_detections_destroy(detections);

    imshow("Tag Detections", currentFrame);

    if (cv::waitKey(30) >= 0)
      break;
  }

  apriltag_detector_destroy(td);
  tag36h11_destroy(tf);

  return 0;
}
