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
#pragma once

#include <config_utilities/config_utilities.h>
#include <hydra/active_window/active_window_output.h>
#include <hydra/common/dsg_types.h>
#include <hydra/common/output_sink.h>
#include <spark_dsg/dynamic_scene_graph.h>

#include <memory>
#include <string>

namespace hydra {

/**
 * @brief An extractor that monitors the dynamic scene graph's Agent nodes
 *        and saves keyframe images (RGB, Depth, etc.) to disk based on
 *        translation and rotation thresholds.
 */
class AgentImageExtractor {
 public:
  struct Config {
    // Minimum translation (in meters) required to trigger a new keyframe
    float min_translation_m = 1.0f;

    // Minimum rotation (in degrees) required to trigger a new keyframe
    float min_rotation_deg = 30.0f;

    // Directory path to save keyframe images to
    std::string image_output_path{""};

    // Toggle whether feature is active
    bool enabled{false};
  } const config;

  explicit AgentImageExtractor(const Config& config);
  virtual ~AgentImageExtractor() = default;

  /**
   * @brief Analyze the latest graph updates, checks thresholds, and potentially
   *        extracts keyframes from the FrameDataBuffer to link to Agent nodes.
   * @param graph The entire dynamic scene graph
   * @param input The current active window output packet holding sensor imagery
   */
  void updateGraph(DynamicSceneGraph& graph, const ActiveWindowOutput& input);

 private:
  struct LastKeyframe {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    bool initialized = false;
  } last_keyframe_;
};

void declare_config(AgentImageExtractor::Config& config);

}  // namespace hydra
