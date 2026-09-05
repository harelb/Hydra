#pragma once

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <cstdint>
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
  // ``world_T_body`` is serialized under the same key / flat row-major 4x4
  // layout as agent_*_meta.json so both dumps load through one code path.
  void write(uint64_t timestamp_ns,
             const cv::Mat& color_rgb,
             const cv::Mat& depth_m,
             const Eigen::Isometry3d& world_T_body = Eigen::Isometry3d::Identity(),
             bool has_pose = false);

 private:
  std::string output_dir_;
};

}  // namespace hydra
