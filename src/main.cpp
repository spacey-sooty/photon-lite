// Copyright (C) Jade T 2025

#define OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT
#define IMGUI_DEFINE_MATH_OPERATORS

#include <apriltag.h>
#include <cscore.h>
#include <cscore_cv.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <tag36h11.h>

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <wpi/print.h>
#include <wpi/spinlock.h>
#include <wpigui.h>

namespace gui = wpi::gui;

int main() {
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

  std::atomic<cv::Mat *> latestFrame{nullptr};
  std::vector<cv::Mat *> sharedFreeList;
  wpi::spinlock sharedFreeListMutex;
  std::vector<cv::Mat *> sourceFreeList;
  std::atomic<bool> stopCamera{false};

  cs::UsbCamera camera{"usbcam", 0};
  camera.SetVideoMode(cs::VideoMode::kMJPEG, 640, 480, 60);
  cs::CvSink cvsink{"cvsink"};
  cvsink.SetSource(camera);

  apriltag_detector_t *detector = apriltag_detector_create();
  apriltag_family_t *family = tag36h11_create();
  apriltag_detector_add_family(detector, family);

  detector->quad_decimate = 1;
  detector->nthreads = 2;
  detector->debug = false;
  detector->refine_edges = true;

  std::thread thr([&] {
    cv::Mat frame;
    while (!stopCamera) {
      // get frame from camera
      uint64_t time = cvsink.GrabFrame(frame);
      if (time == 0) {
        wpi::print("error: {}\n", cvsink.GetError());
        continue;
      }

      // get or create a mat, prefer sourceFreeList over sharedFreeList
      cv::Mat *out;
      if (!sourceFreeList.empty()) {
        out = sourceFreeList.back();
        sourceFreeList.pop_back();
      } else {
        {
          std::scoped_lock lock(sharedFreeListMutex);
          for (auto mat : sharedFreeList) {
            sourceFreeList.emplace_back(mat);
          }
          sharedFreeList.clear();
        }
        if (!sourceFreeList.empty()) {
          out = sourceFreeList.back();
          sourceFreeList.pop_back();
        } else {
          out = new cv::Mat;
        }
      }

      // convert to RGBA
      cv::cvtColor(frame, *out, cv::COLOR_BGR2RGBA);

      // make available
      auto prev = latestFrame.exchange(out);

      // put prev on free list
      if (prev) {
        sourceFreeList.emplace_back(prev);
      }
    }
  });

  gui::CreateContext();
  gui::Initialize("Photon Lite", 1024, 768);
  gui::Texture tex;
  gui::AddEarlyExecute([&] {
    auto frame = latestFrame.exchange(nullptr);
    cv::Mat gray;
    if (frame) {
      cv::cvtColor(*frame, gray, cv::COLOR_BGR2GRAY);

      image_u8_t im = {gray.cols, gray.rows, gray.cols, gray.data};
      zarray_t *detections = apriltag_detector_detect(detector, &im);

      std::vector<cv::Point2d> points{};

      for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);
        points.emplace_back(cv::Point2d{det->p[0][0], det->p[0][1]});
        points.emplace_back(cv::Point2d{det->p[1][0], det->p[1][1]});
        points.emplace_back(cv::Point2d{det->p[3][0], det->p[3][1]});
        points.emplace_back(cv::Point2d{det->p[2][0], det->p[2][1]});
        cv::line(*frame, cv::Point(det->p[0][0], det->p[0][1]),
                 cv::Point(det->p[1][0], det->p[1][1]), cv::Scalar(0, 0xff, 0),
                 2);
        cv::line(*frame, cv::Point(det->p[0][0], det->p[0][1]),
                 cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0, 0, 0xff),
                 2);
        cv::line(*frame, cv::Point(det->p[1][0], det->p[1][1]),
                 cv::Point(det->p[2][0], det->p[2][1]), cv::Scalar(0xff, 0, 0),
                 2);
        cv::line(*frame, cv::Point(det->p[2][0], det->p[2][1]),
                 cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0xff, 0, 0),
                 2);

        std::stringstream ss;
        ss << det->id;
        cv::String text = ss.str();
        int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontscale = 1.0;
        int baseline;
        cv::Size textsize =
            cv::getTextSize(text, fontface, fontscale, 2, &baseline);
        cv::putText(*frame, text,
                    cv::Point(det->c[0] - textsize.width / 2.0,
                              det->c[1] + textsize.height / 2.0),
                    fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
      }

      if (points.size() >= 4) {
        cv::Mat rvec{3, 3, CV_64F};
        cv::Mat tvec{3, 3, CV_64F};
        std::vector<double> distCoeffs{
            0.02860487071331241,   0.009126602251891335, 0.0019540088117773633,
            -0.003596010527440653, -0.13863285564042926, -0.0016559347518291224,
            -4.017061418060786,    0.005592461908791266};
        cv::Mat calibration{3, 3, CV_64F};
        cv::eigen2cv(A, calibration);
        cv::solvePnP(tagPoints, points, calibration, distCoeffs, rvec, tvec);
        std::cout << "Rotation = " << std::endl << " " << rvec << std::endl;
        std::cout << "Translation = " << std::endl << " " << tvec << std::endl;

        rvec.release();
        tvec.release();
      }

      apriltag_detections_destroy(detections);

      // create or update texture
      if (!tex || frame->cols != tex.GetWidth() ||
          frame->rows != tex.GetHeight()) {
        tex = gui::Texture(gui::kPixelRGBA, frame->cols, frame->rows,
                           frame->data);
      } else {
        tex.Update(frame->data);
      }
      // put back on shared freelist
      std::scoped_lock lock(sharedFreeListMutex);
      sharedFreeList.emplace_back(frame);
    }

    ImGui::SetNextWindowSize(ImVec2(640, 480), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Video")) {
      // render to window (best fit)
      if (tex && tex.GetWidth() != 0 && tex.GetHeight() != 0) {
        auto drawList = ImGui::GetWindowDrawList();
        ImVec2 windowPos = ImGui::GetWindowPos();
        ImVec2 imageMin = ImGui::GetWindowContentRegionMin();
        ImVec2 imageMax = ImGui::GetWindowContentRegionMax();
        gui::MaxFit(&imageMin, &imageMax, tex.GetWidth(), tex.GetHeight());
        drawList->AddImage(tex, windowPos + imageMin, windowPos + imageMax);
      }
    }
    ImGui::End();
  });

  gui::Main();

  apriltag_detector_destroy(detector);
  tag36h11_destroy(family);

  stopCamera = true;
  thr.join();
}
