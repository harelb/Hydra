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

  writer.write(42, color, depth);

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
  EXPECT_NE(content.find("subkf_42_rgb.jpg"), std::string::npos);
  EXPECT_NE(content.find("subkf_42_depth.png"), std::string::npos);
  // The stale baked pose was removed; the authoritative pose now lives in the
  // DSG sub-keyframe node, not in the per-image metadata.
  EXPECT_EQ(content.find("world_T_body"), std::string::npos);

  std::filesystem::remove_all(dir);
}

TEST(KeyframeWriter, WritesCalibWithDepthEncoding) {
  auto dir = std::filesystem::temp_directory_path() / "kf_writer_calib_test";
  std::filesystem::remove_all(dir);
  KeyframeWriter writer(dir.string());

  CameraCalib calib;
  calib.fx = 500.0;
  calib.fy = 501.0;
  calib.cx = 320.0;
  calib.cy = 240.0;
  calib.width = 640;
  calib.height = 480;

  writer.writeCalib(calib);

  auto calib_path = dir / "camera_calib.json";
  ASSERT_TRUE(std::filesystem::exists(calib_path));

  std::ifstream f(calib_path);
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"fx\""), std::string::npos);
  EXPECT_NE(content.find("\"depth_scale\""), std::string::npos);
  EXPECT_NE(content.find("\"depth_encoding\": \"16UC1_mm\""), std::string::npos);

  std::filesystem::remove_all(dir);
}

}  // namespace hydra
