// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include <apriltag.h>
#include <atomic>
#include <memory>
#include <tag36h11.h>
#include <thread>
#include <utility>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <wpi/mutex.h>
#include <wpi/print.h>
#include <wpi/spinlock.h>
#include <wpigui.h>

#include "cscore.h"
#include "cscore_cv.h"

namespace gui = wpi::gui;

int main() {
  wpi::spinlock latestFrameMutex;
  std::unique_ptr<cv::Mat> latestFrame;
  wpi::mutex freeListMutex;
  std::vector<std::unique_ptr<cv::Mat>> freeList;
  std::atomic<bool> stopCamera{false};

  cs::UsbCamera camera{"usbcam", 0};
  camera.SetVideoMode(cs::VideoMode::kMJPEG, 640, 480, 30);
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

      // get or create a mat
      std::unique_ptr<cv::Mat> out;
      {
        std::scoped_lock lock{freeListMutex};
        if (!freeList.empty()) {
          out = std::move(freeList.back());
          freeList.pop_back();
        } else {
          out = std::make_unique<cv::Mat>();
        }
      }

      // convert to RGBA
      cv::cvtColor(frame, *out, cv::COLOR_BGR2RGBA);

      {
        // make available
        std::scoped_lock lock{latestFrameMutex};
        latestFrame.swap(out);
      }

      // put the previous frame on free list
      if (out) {
        std::scoped_lock lock{freeListMutex};
        freeList.emplace_back(std::move(out));
      }
    }
  });

  gui::CreateContext();
  gui::Initialize("Hello World", 1024, 768);
  gui::Texture tex;
  gui::AddEarlyExecute([&] {
    cv::Mat gray;
    std::unique_ptr<cv::Mat> frame;
    {
      std::scoped_lock lock{latestFrameMutex};
      latestFrame.swap(frame);
    }
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

      apriltag_detections_destroy(detections);

      // create or update texture
      if (!tex || frame->cols != tex.GetWidth() ||
          frame->rows != tex.GetHeight()) {
        tex = gui::Texture(gui::kPixelRGBA, frame->cols, frame->rows,
                           frame->data);
      } else {
        tex.Update(frame->data);
      }
      {
        std::scoped_lock lock{freeListMutex};
        freeList.emplace_back(std::move(frame));
      }
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
