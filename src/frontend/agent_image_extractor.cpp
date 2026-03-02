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

namespace hydra {

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

      std::stringstream ss;
      ss << "agent_" << attrs.timestamp.count();
      std::filesystem::path base_path =
          std::filesystem::path(config.image_output_path) / ss.str();
          
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
      
      // Save Depth
      if (!sensor_data.depth_image.empty()) {
        cv::imwrite(base_path.string() + "_depth.png", sensor_data.depth_image);
      }

      attrs.image_folder = base_path.string();
      
      VLOG(3) << "[AgentImageExtractor] Triggered keyframe extraction for agent " 
              << spark_dsg::NodeSymbol(node->id).str() 
              << " @ " << attrs.timestamp.count() << " ns";
    }
  }
}

}  // namespace hydra
