/* -----------------------------------------------------------------------------
 * Copyright 2024 Massachusetts Institute of Technology.
 * All Rights Reserved
 * -------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <hydra/active_window/active_window_output.h>
#include <hydra/frontend/agent_image_extractor.h>
#include <hydra/input/camera.h>
#include <hydra/input/input_data.h>

#include <filesystem>

namespace hydra {
namespace {

Camera::Config makeCameraConfig() {
  Camera::Config config;
  config.width = 8;
  config.height = 4;
  config.cx = 4.0;
  config.cy = 2.0;
  config.fx = 4.0;
  config.fy = 4.0;
  config.min_range = 0.1;
  config.max_range = 10.0;
  config.extrinsics = ParamSensorExtrinsics::Config();
  return config;
}

// Number of cv::Mat headers sharing this Mat's pixel buffer (1 == sole owner).
int refCount(const cv::Mat& mat) { return mat.u ? mat.u->refcount : 0; }

}  // namespace

// The extractor must not share pixel buffers with the packet it is handed. The
// active window builds ActiveWindowOutput::sensor_data as a SHALLOW copy of its own
// frame data, so retaining those cv::Mats keeps the active window's frame buffers
// alive past the point it believes it freed them, and leaves this module reading
// pixels that the active window's detached workers can still reach.
TEST(AgentImageExtractor, addFrameSharesNothingWithInput) {
  const auto output_dir = std::filesystem::temp_directory_path() /
                          "hydra_test_agent_image_extractor_share";
  std::filesystem::remove_all(output_dir);

  AgentImageExtractor::Config config;
  config.enabled = true;
  config.image_output_path = output_dir.string();
  AgentImageExtractor extractor(config);

  auto sensor = std::make_shared<Camera>(makeCameraConfig(), "camera");
  auto data = std::make_shared<InputData>(sensor);
  data->timestamp_ns = 1000;
  data->world_T_body = Eigen::Isometry3d::Identity();
  data->color_image = cv::Mat(4, 8, CV_8UC3, cv::Scalar(1, 2, 3));
  data->depth_image = cv::Mat(4, 8, CV_32FC1, cv::Scalar(1.5f));

  ASSERT_EQ(refCount(data->color_image), 1);
  ASSERT_EQ(refCount(data->depth_image), 1);

  ActiveWindowOutput output;
  output.timestamp_ns = data->timestamp_ns;
  output.sensor_data = data;

  extractor.addFrame(output);

  // If the extractor had retained the packet's images, these would now be 2.
  EXPECT_EQ(refCount(data->color_image), 1);
  EXPECT_EQ(refCount(data->depth_image), 1);

  std::filesystem::remove_all(output_dir);
}

// Buffering must survive the producer dropping the packet entirely, which is the
// lifetime the active window actually gives us.
TEST(AgentImageExtractor, addFrameSurvivesInputDestruction) {
  const auto output_dir = std::filesystem::temp_directory_path() /
                          "hydra_test_agent_image_extractor_lifetime";
  std::filesystem::remove_all(output_dir);

  AgentImageExtractor::Config config;
  config.enabled = true;
  config.image_output_path = output_dir.string();
  AgentImageExtractor extractor(config);

  auto sensor = std::make_shared<Camera>(makeCameraConfig(), "camera");
  {
    auto data = std::make_shared<InputData>(sensor);
    data->timestamp_ns = 2000;
    data->world_T_body = Eigen::Isometry3d::Identity();
    data->color_image = cv::Mat(4, 8, CV_8UC3, cv::Scalar(4, 5, 6));
    data->depth_image = cv::Mat(4, 8, CV_32FC1, cv::Scalar(2.5f));

    ActiveWindowOutput output;
    output.timestamp_ns = data->timestamp_ns;
    output.sensor_data = data;
    extractor.addFrame(output);
  }  // packet and all its imagery released here

  // Nothing to assert beyond surviving: with shared buffers this is where a
  // use-after-free would live. Buffered frames are released with the extractor.
  SUCCEED();
}

// Disabled extractors must ignore frames entirely (no allocation, no output dir).
TEST(AgentImageExtractor, addFrameIgnoredWhenDisabled) {
  AgentImageExtractor::Config config;
  config.enabled = false;
  config.image_output_path = "";
  AgentImageExtractor extractor(config);

  auto sensor = std::make_shared<Camera>(makeCameraConfig(), "camera");
  auto data = std::make_shared<InputData>(sensor);
  data->timestamp_ns = 3000;
  data->color_image = cv::Mat(4, 8, CV_8UC3, cv::Scalar(7, 8, 9));
  data->depth_image = cv::Mat(4, 8, CV_32FC1, cv::Scalar(3.5f));

  ActiveWindowOutput output;
  output.timestamp_ns = data->timestamp_ns;
  output.sensor_data = data;
  extractor.addFrame(output);

  EXPECT_EQ(refCount(data->color_image), 1);
  EXPECT_EQ(refCount(data->depth_image), 1);
}

}  // namespace hydra
