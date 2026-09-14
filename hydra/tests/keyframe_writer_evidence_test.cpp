// Standalone test: link keyframe_writer.cpp, OpenCV and Eigen headers.
#include "hydra/frontend/keyframe_writer.h"
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>
#include <cassert>
#include <filesystem>
#include <fstream>

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path output(argv[1]);
  hydra::KeyframeWriter writer(output.string());
  hydra::CameraCalib calibration{400, 410, 1, 1, 2, 2};
  writer.writeCalib(calibration);
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(2, 3, 4);
  writer.write(123, cv::Mat::zeros(2, 2, CV_8UC3), cv::Mat(2, 2, CV_32FC1, 1.234f), pose);
  std::ifstream input(output / "subkf_123_meta.json");
  const auto meta = nlohmann::json::parse(input);
  assert(meta.at("timestamp_ns") == 123);
  assert(meta.at("world_T_body").at(3) == 2);
  assert(meta.at("world_T_body").at(7) == 3);
  assert(meta.at("world_T_body").at(11) == 4);
  const auto depth = cv::imread((output / "subkf_123_depth.png").string(), cv::IMREAD_UNCHANGED);
  assert(depth.type() == CV_16UC1);
  assert(depth.at<uint16_t>(0, 0) == 1234);
}
