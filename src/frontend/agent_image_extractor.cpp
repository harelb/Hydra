/* -----------------------------------------------------------------------------
 * Copyright 2024 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */
#include "hydra/frontend/agent_image_extractor.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "hydra/common/global_info.h"
#include "hydra/input/camera.h"

namespace hydra {

namespace {

// Serialize a 4x4 transform as a flat row-major JSON array. Matches the manual
// JSON convention used elsewhere (see khronos mesh_object_extractor.cpp).
std::string isometryToJsonArray(const Eigen::Isometry3d& transform) {
  const Eigen::Matrix4d m = transform.matrix();
  std::stringstream ss;
  // Full double round-trip precision; default (6 sig figs) loses sub-cm on poses.
  ss << std::setprecision(17);
  ss << "[";
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      ss << m(r, c);
      if (!(r == 3 && c == 3)) {
        ss << ", ";
      }
    }
  }
  ss << "]";
  return ss.str();
}

// Storage convention for persisted depth: 16-bit PNG in millimeters. The Python
// reprojection multiplies stored values by depth_scale to recover meters.
constexpr double kDepthScaleMetersPerUnit = 1.0e-3;
constexpr const char* kDepthEncoding = "16UC1_mm";

}  // namespace

void declare_config(AgentImageExtractor::Config& config) {
  using namespace config;
  name("AgentImageExtractor::Config");
  field(config.min_translation_m, "min_translation_m");
  field(config.min_rotation_deg, "min_rotation_deg");
  field(config.image_output_path, "image_output_path");
  field(config.enabled, "enabled");

  check(config.min_translation_m, GE, 0.0f, "min_translation_m");
  check(config.min_rotation_deg, GE, 0.0f, "min_rotation_deg");
}

AgentImageExtractor::AgentImageExtractor(const Config& config) : config(config) {
  if (config.enabled && !config.image_output_path.empty()) {
    std::filesystem::path output_dir(config.image_output_path);
    if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directories(output_dir);
    }
  }
}

void AgentImageExtractor::updateGraph(DynamicSceneGraph& graph, const ActiveWindowOutput& input) {
  if (!config.enabled || config.image_output_path.empty()) {
    return;
  }

  const auto layer_id = graph.getLayerKey(DsgLayers::AGENTS);
  if (!layer_id) {
    return;
  }

  const auto& prefix = GlobalInfo::instance().getRobotPrefix();
  const auto layer = graph.findLayer(layer_id->layer, prefix.key);
  if (!layer) {
    return;
  }

  // Iterate over all agent nodes we haven't processed yet or find the latest
  // Since updateGraph is called repeatedly, we check the newest nodes.
  // We can just iterate and skip ones with image_folder already populated,
  // or just track the distance from the last_keyframe_.
  
  // Get active nodes, sort by timestamp
  std::vector<const SceneGraphNode*> agent_nodes;
  for (const auto& [node_id, node] : layer->nodes()) {
    agent_nodes.push_back(node.get());
  }
  std::sort(agent_nodes.begin(), agent_nodes.end(), [](const auto& a, const auto& b) {
    return a->template attributes<AgentNodeAttributes>().timestamp.count() <
           b->template attributes<AgentNodeAttributes>().timestamp.count();
  });

  for (const auto* node : agent_nodes) {
    auto& attrs = node->template attributes<AgentNodeAttributes>();
    
    // Skip if already processed
    if (!attrs.image_folder.empty()) {
      continue;
    }

    const Eigen::Vector3d current_pos = attrs.position;
    const Eigen::Quaterniond current_rot = attrs.world_R_body;

    bool should_trigger = false;

    if (!last_keyframe_.initialized) {
      should_trigger = true;
    } else {
      const double translation_diff = (current_pos - last_keyframe_.position).norm();
      const double angular_diff =
          last_keyframe_.orientation.angularDistance(current_rot) * 180.0 / M_PI;

      if (translation_diff >= config.min_translation_m ||
          angular_diff >= config.min_rotation_deg) {
        should_trigger = true;
      }
    }

    if (should_trigger) {
      if (!input.sensor_data) {
        // No underlying camera data available this frame
        continue;
      }

      // Update state
      last_keyframe_.position = current_pos;
      last_keyframe_.orientation = current_rot;
      last_keyframe_.initialized = true;

      const auto& sensor_data = *input.sensor_data;

      const std::string name = "agent_" + std::to_string(attrs.timestamp.count());
      std::filesystem::path base_path =
          std::filesystem::path(config.image_output_path) / name;

      // Write the run-level calibration once. Reprojecting a stored mask to 3D
      // requires intrinsics + extrinsics, which are constant for a fixed camera,
      // so we keep them out of the per-keyframe metadata.
      if (!calib_written_) {
        const auto* camera = dynamic_cast<const Camera*>(&sensor_data.getSensor());
        if (camera) {
          const auto& cc = camera->getConfig();
          std::filesystem::path calib_path =
              std::filesystem::path(config.image_output_path) / "camera_calib.json";
          std::ofstream calib(calib_path);
          calib << std::setprecision(17);
          calib << "{\n";
          calib << "  \"fx\": " << cc.fx << ",\n";
          calib << "  \"fy\": " << cc.fy << ",\n";
          calib << "  \"cx\": " << cc.cx << ",\n";
          calib << "  \"cy\": " << cc.cy << ",\n";
          calib << "  \"width\": " << cc.width << ",\n";
          calib << "  \"height\": " << cc.height << ",\n";
          calib << "  \"depth_scale\": " << kDepthScaleMetersPerUnit << ",\n";
          calib << "  \"depth_encoding\": \"" << kDepthEncoding << "\",\n";
          calib << "  \"body_T_sensor\": "
                << isometryToJsonArray(camera->body_T_sensor()) << "\n";
          calib << "}\n";
          calib_written_ = true;
        } else {
          VLOG(1) << "[AgentImageExtractor] Sensor is not a Camera; skipping "
                     "calibration export (reprojection will be unavailable).";
        }
      }

      // Save RGB (OpenCV uses BGR)
      if (!sensor_data.color_image.empty()) {
        cv::Mat rgb_image;
        if (sensor_data.color_image.channels() == 3) {
          cv::cvtColor(sensor_data.color_image, rgb_image, cv::COLOR_RGB2BGR);
        } else {
          rgb_image = sensor_data.color_image.clone();
        }
        cv::imwrite(base_path.string() + "_rgb.jpg", rgb_image);
      }

      // Save Depth losslessly. After input conversion depth_image is CV_32FC1 in
      // meters (see input_conversion.cpp); PNG cannot store float, so we convert
      // to 16-bit millimeters to round-trip cleanly.
      if (!sensor_data.depth_image.empty()) {
        const cv::Mat& depth = sensor_data.depth_image;
        cv::Mat depth_to_save;
        if (depth.type() == CV_32FC1) {
          depth.convertTo(depth_to_save, CV_16UC1, 1.0 / kDepthScaleMetersPerUnit);
        } else {
          // Already integer depth (assumed millimeters); store as-is.
          depth_to_save = depth;
        }
        cv::imwrite(base_path.string() + "_depth.png", depth_to_save);
      }

      // Per-keyframe metadata: dynamic data only (pose + file references). Static
      // calibration lives in camera_calib.json.
      {
        std::ofstream meta(base_path.string() + "_meta.json");
        meta << "{\n";
        meta << "  \"timestamp_ns\": " << attrs.timestamp.count() << ",\n";
        meta << "  \"world_T_body\": "
             << isometryToJsonArray(sensor_data.world_T_body) << ",\n";
        meta << "  \"rgb_file\": \"" << name << "_rgb.jpg\",\n";
        meta << "  \"depth_file\": \"" << name << "_depth.png\",\n";
        meta << "  \"calib\": \"camera_calib.json\"\n";
        meta << "}\n";
      }

      attrs.image_folder = base_path.string();

      VLOG(3) << "[AgentImageExtractor] Triggered keyframe extraction for agent " 
              << spark_dsg::NodeSymbol(node->id).str() 
              << " @ " << attrs.timestamp.count() << " ns";
    }
  }
}

}  // namespace hydra
