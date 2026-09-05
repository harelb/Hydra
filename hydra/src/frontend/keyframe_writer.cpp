#include "hydra/frontend/keyframe_writer.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <opencv2/opencv.hpp>

namespace hydra {
namespace {

constexpr double kDepthScaleMetersPerUnit = 1.0e-3;
constexpr const char* kDepthEncoding = "16UC1_mm";

std::string isometryToJsonArray(const Eigen::Isometry3d& transform) {
  const Eigen::Matrix4d m = transform.matrix();
  std::stringstream ss;
  ss << std::setprecision(17) << "[";
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      ss << m(r, c);
      if (!(r == 3 && c == 3)) ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

}  // namespace

KeyframeWriter::KeyframeWriter(const std::string& output_dir)
    : output_dir_(output_dir) {
  std::filesystem::create_directories(output_dir_);
}

void KeyframeWriter::writeCalib(const CameraCalib& calib) {
  std::filesystem::path p = std::filesystem::path(output_dir_) / "camera_calib.json";
  std::ofstream f(p);
  f << std::setprecision(17) << "{\n";
  f << "  \"fx\": " << calib.fx << ",\n  \"fy\": " << calib.fy << ",\n";
  f << "  \"cx\": " << calib.cx << ",\n  \"cy\": " << calib.cy << ",\n";
  f << "  \"width\": " << calib.width << ",\n  \"height\": " << calib.height << ",\n";
  f << "  \"depth_scale\": " << kDepthScaleMetersPerUnit << ",\n";
  f << "  \"depth_encoding\": \"" << kDepthEncoding << "\",\n";
  f << "  \"body_T_sensor\": " << isometryToJsonArray(calib.body_T_sensor) << "\n}\n";
}

void KeyframeWriter::write(uint64_t timestamp_ns,
                           const cv::Mat& color_rgb,
                           const cv::Mat& depth_m,
                           const Eigen::Isometry3d& world_T_body,
                           bool has_pose) {
  const std::string base =
      (std::filesystem::path(output_dir_) / ("subkf_" + std::to_string(timestamp_ns)))
          .string();

  if (!color_rgb.empty()) {
    cv::Mat bgr;
    if (color_rgb.channels() == 3) {
      cv::cvtColor(color_rgb, bgr, cv::COLOR_RGB2BGR);
    } else {
      bgr = color_rgb.clone();
    }
    cv::imwrite(base + "_rgb.jpg", bgr);
  }

  if (!depth_m.empty()) {
    cv::Mat depth_to_save;
    if (depth_m.type() == CV_32FC1) {
      depth_m.convertTo(depth_to_save, CV_16UC1, 1.0 / kDepthScaleMetersPerUnit);
    } else {
      depth_to_save = depth_m;
    }
    cv::imwrite(base + "_depth.png", depth_to_save);
  }

  std::ofstream meta(base + "_meta.json");
  meta << "{\n";
  meta << std::setprecision(17);
  meta << "  \"timestamp_ns\": " << timestamp_ns << ",\n";
  if (has_pose) {
    meta << "  \"world_T_body\": " << isometryToJsonArray(world_T_body) << ",\n";
  }
  meta << "  \"rgb_file\": \"subkf_" << timestamp_ns << "_rgb.jpg\",\n";
  meta << "  \"depth_file\": \"subkf_" << timestamp_ns << "_depth.png\",\n";
  meta << "  \"calib\": \"camera_calib.json\"\n";
  meta << "}\n";
}

}  // namespace hydra
