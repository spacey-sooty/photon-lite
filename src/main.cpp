// Copyright (C) Jade T 2025

#include <atomic>
#include <cassert>
#include <memory>
#include <opencv2/core/types.hpp>
#include <thread>
#include <utility>
#include <vector>

#include <apriltag.h>
#include <cscore.h>
#include <cscore_cv.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <frc/apriltag/AprilTagDetection.h>
#include <frc/apriltag/AprilTagDetector.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <tag36h11.h>
#include <wpi/mutex.h>
#include <wpi/print.h>
#include <wpi/spinlock.h>
#include <wpigui.h>

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

  frc::AprilTagDetector detector;
  detector.AddFamily("tag36h11", 3);

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

      cv::Size size = gray.size();
      frc::AprilTagDetector::Results detections = detector.Detect(size.height, size.width, gray.data);

      std::vector<frc::AprilTagDetection::Point> points{};

      for (size_t i = 0; i < detections.size(); i++) {
        const frc::AprilTagDetection* det = detections[i];
        points.emplace_back(det->GetCorner(0));
        points.emplace_back(det->GetCorner(1));
        points.emplace_back(det->GetCorner(2));
        points.emplace_back(det->GetCorner(3));
        cv::line(*frame, cv::Point(points[0].x, points[0].y),
                 cv::Point(points[1].x, points[1].y), cv::Scalar(0, 0xff, 0),
                 2);
        cv::line(*frame, cv::Point(points[0].x, points[0].y),
                 cv::Point(points[3].x, points[3].y), cv::Scalar(0, 0, 0xff),
                 2);
        cv::line(*frame, cv::Point(points[1].x, points[1].y),
                 cv::Point(points[2].x, points[2].y), cv::Scalar(0xff, 0, 0),
                 2);
        cv::line(*frame, cv::Point(points[2].x, points[2].y),
                 cv::Point(points[3].x, points[3].y), cv::Scalar(0xff, 0, 0),
                 2);

        std::stringstream ss;
        ss << det->GetId();
        cv::String text = ss.str();
        int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontscale = 1.0;
        int baseline;
        cv::Size textsize =
            cv::getTextSize(text, fontface, fontscale, 2, &baseline);
        cv::putText(*frame, text,
                    cv::Point(det->GetCenter().x - textsize.width / 2.0,
                              det->GetCenter().y + textsize.height / 2.0),
                    fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
      }

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

  stopCamera = true;
  thr.join();
}
