#pragma once

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <string>

namespace hydra {

struct CameraCalib {
  double fx, fy, cx, cy;
  int width, height;
  Eigen::Isometry3d body_T_sensor = Eigen::Isometry3d::Identity();
};

// ROS-free writer for full-rate sub-keyframes. Filenames: subkf_<ts>_{rgb.jpg,
// depth.png,meta.json}. Depth stored as 16-bit millimeters. Mirrors the storage
// convention of AgentImageExtractor (agent_image_extractor.cpp:161-233).
class KeyframeWriter {
 public:
  explicit KeyframeWriter(const std::string& output_dir);

  void writeCalib(const CameraCalib& calib);
  void write(uint64_t timestamp_ns,
             const cv::Mat& color_rgb,
             const cv::Mat& depth_m,
             const Eigen::Isometry3d& world_T_body);

 private:
  std::string output_dir_;
};

}  // namespace hydra
