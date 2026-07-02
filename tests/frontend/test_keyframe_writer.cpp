#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "hydra/frontend/keyframe_writer.h"

namespace hydra {

TEST(KeyframeWriter, WritesRgbDepthMeta) {
  auto dir = std::filesystem::temp_directory_path() / "kf_writer_test";
  std::filesystem::remove_all(dir);
  KeyframeWriter writer(dir.string());

  cv::Mat color(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
  cv::Mat depth(4, 4, CV_32FC1, cv::Scalar(1.5f));  // 1.5 m
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(1, 2, 3);

  writer.write(42, color, depth, pose);

  EXPECT_TRUE(std::filesystem::exists(dir / "subkf_42_rgb.jpg"));
  EXPECT_TRUE(std::filesystem::exists(dir / "subkf_42_depth.png"));
  EXPECT_TRUE(std::filesystem::exists(dir / "subkf_42_meta.json"));

  // depth round-trips to 16-bit mm: 1.5 m -> 1500
  cv::Mat loaded = cv::imread((dir / "subkf_42_depth.png").string(),
                              cv::IMREAD_UNCHANGED);
  ASSERT_EQ(loaded.type(), CV_16UC1);
  EXPECT_EQ(loaded.at<uint16_t>(0, 0), 1500);

  std::ifstream meta(dir / "subkf_42_meta.json");
  std::string content((std::istreambuf_iterator<char>(meta)),
                      std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"timestamp_ns\": 42"), std::string::npos);
  EXPECT_NE(content.find("world_T_body"), std::string::npos);

  std::filesystem::remove_all(dir);
}

}  // namespace hydra
