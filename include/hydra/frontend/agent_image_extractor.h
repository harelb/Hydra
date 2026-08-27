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

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

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

    // Maximum |t_node - t_frame| (in seconds) accepted when pairing an agent node
    // with the sensor frame it was created from. Agent nodes are stamped with the
    // active-window output timestamp they were created from (see
    // PoseGraphFromOdom::update), so a correct pairing normally has dt == 0; the
    // tolerance only covers trackers that restamp their pose graph nodes.
    double max_pairing_time_diff_s = 0.05;

    // Number of recent sensor frames retained for pairing. The frontend collates
    // several active-window outputs into a single spin when it falls behind, so the
    // buffer has to span the worst-case backlog. Frames are held by shared_ptr, so
    // this bounds how much imagery the extractor keeps alive.
    size_t max_buffered_frames = 30;
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

  /**
   * @brief Record the sensor data of an active-window output so it stays available
   *        for pairing with the agent node that is created from the same packet.
   *        Must be called for EVERY active-window output, including the ones that get
   *        collated away before the frontend spin, otherwise their imagery is lost.
   * @param input An active window output packet, before any collation
   */
  void addFrame(const ActiveWindowOutput& input);

 private:
  //! One buffered sensor frame, keyed by the active-window output timestamp (which is
  //! also the timestamp the pose graph tracker stamps its agent nodes with).
  struct BufferedFrame {
    uint64_t timestamp_ns = 0;
    std::shared_ptr<InputData> data;
  };

  //! Index of the buffered frame nearest to timestamp_ns at or after search_start, or
  //! std::nullopt if the nearest one is further away than max_pairing_time_diff_s.
  std::optional<size_t> findFrame(const std::vector<BufferedFrame>& frames,
                                  size_t search_start,
                                  uint64_t timestamp_ns) const;

  struct LastKeyframe {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    bool initialized = false;
  } last_keyframe_;

  //! Whether the run-level camera_calib.json has been written yet.
  bool calib_written_ = false;

  //! Timestamp of the newest agent node we have already made a decision about (written
  //! a keyframe for, gated out, or given up on). Nodes are visited in time order, so
  //! this keeps every node from being reconsidered on every call.
  uint64_t last_decided_ns_ = 0;

  //! Recent sensor frames, oldest first. Written from the active-window thread by
  //! addFrame and read from the frontend spin thread by updateGraph.
  std::deque<BufferedFrame> frames_;
  mutable std::mutex frame_mutex_;
};

void declare_config(AgentImageExtractor::Config& config);

}  // namespace hydra
