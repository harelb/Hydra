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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "hydra/common/global_info.h"
#include "hydra/input/camera.h"

// hydra/common/dsg_types.h (deleted upstream in #174) used to supply this
using namespace spark_dsg;

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
  field(config.max_pairing_time_diff_s, "max_pairing_time_diff_s", "s");
  field(config.max_buffered_frames, "max_buffered_frames");
  field(config.max_deferred_updates, "max_deferred_updates");

  check(config.min_translation_m, GE, 0.0f, "min_translation_m");
  check(config.min_rotation_deg, GE, 0.0f, "min_rotation_deg");
  check(config.max_pairing_time_diff_s, GE, 0.0, "max_pairing_time_diff_s");
  check(config.max_buffered_frames, GT, size_t(0), "max_buffered_frames");
  check(config.max_deferred_updates, GT, size_t(0), "max_deferred_updates");
}

AgentImageExtractor::AgentImageExtractor(const Config& config) : config(config) {
  if (config.enabled && !config.image_output_path.empty()) {
    std::filesystem::path output_dir(config.image_output_path);
    if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directories(output_dir);
    }
  }
}

void AgentImageExtractor::addFrame(const ActiveWindowOutput& input) {
  if (!config.enabled || config.image_output_path.empty() || !input.sensor_data) {
    return;
  }

  const auto& data = *input.sensor_data;

  // Take a PRIVATE copy of everything we will need later. The packet's InputData
  // shares its cv::Mat pixel buffers with the active window's own frame data (see
  // BufferedFrame), so retaining it would couple this module's lifetime and thread
  // safety to the active window's frame buffer. Converting into the on-disk format
  // here allocates fresh buffers, so the copy is free: we were paying for these
  // conversions at write time anyway, and doing them on this thread keeps the
  // frontend spin cheap.
  BufferedFrame frame;
  frame.timestamp_ns = input.timestamp_ns;
  frame.world_T_body = data.world_T_body;

  if (!data.color_image.empty()) {
    if (data.color_image.channels() == 3) {
      cv::cvtColor(data.color_image, frame.color_bgr, cv::COLOR_RGB2BGR);
    } else {
      frame.color_bgr = data.color_image.clone();
    }
  }

  if (!data.depth_image.empty()) {
    if (data.depth_image.type() == CV_32FC1) {
      // After input conversion depth_image is CV_32FC1 in meters (see
      // input_conversion.cpp); PNG cannot store float, so we round-trip through
      // 16-bit millimeters.
      data.depth_image.convertTo(
          frame.depth_mm, CV_16UC1, 1.0 / kDepthScaleMetersPerUnit);
    } else {
      // Already integer depth (assumed millimeters); store as-is.
      frame.depth_mm = data.depth_image.clone();
    }
  }

  std::lock_guard<std::mutex> lock(frame_mutex_);

  // Capture the calibration by value the first time we see a camera. Reprojecting a
  // stored mask to 3D needs intrinsics + extrinsics, which are constant for a fixed
  // camera, so they stay out of the per-keyframe metadata.
  if (!calib_) {
    const auto* camera = dynamic_cast<const Camera*>(&data.getSensor());
    if (camera) {
      const auto& cc = camera->getConfig();
      CameraCalib calib;
      calib.fx = cc.fx;
      calib.fy = cc.fy;
      calib.cx = cc.cx;
      calib.cy = cc.cy;
      calib.width = cc.width;
      calib.height = cc.height;
      calib.body_T_sensor = camera->body_T_sensor();
      calib_ = calib;
    } else {
      LOG_FIRST_N(WARNING, 1) << "[AgentImageExtractor] Sensor is not a Camera; "
                                 "skipping calibration export (reprojection will be "
                                 "unavailable).";
    }
  }

  if (!frames_.empty() && frames_.back().timestamp_ns >= input.timestamp_ns) {
    // Duplicate or out-of-order packet: the buffer must stay sorted for findFrame.
    return;
  }

  frames_.push_back(std::move(frame));
  while (frames_.size() > config.max_buffered_frames) {
    frames_.pop_front();
  }
}

std::optional<size_t> AgentImageExtractor::findFrame(
    const std::vector<BufferedFrame>& frames,
    size_t search_start,
    uint64_t timestamp_ns) const {
  const auto tolerance_ns =
      static_cast<uint64_t>(config.max_pairing_time_diff_s * 1.0e9);

  std::optional<size_t> best;
  uint64_t best_diff = 0;
  for (size_t i = search_start; i < frames.size(); ++i) {
    const auto frame_ns = frames[i].timestamp_ns;
    const uint64_t diff =
        frame_ns > timestamp_ns ? frame_ns - timestamp_ns : timestamp_ns - frame_ns;
    if (diff > tolerance_ns) {
      // Frames are sorted, so once we are past the node we can only get worse.
      if (frame_ns > timestamp_ns) {
        break;
      }
      continue;
    }

    if (!best || diff < best_diff) {
      best = i;
      best_diff = diff;
    }
  }

  return best;
}

void AgentImageExtractor::updateGraph(spark_dsg::SceneGraph& graph,
                                      const ActiveWindowOutput&) {
  // NOTE(harel): deliberately does NOT call addFrame. GraphBuilder::processNextInput
  // already buffered this packet (and the ones collated away with it) on the active
  // window's thread; calling it here as well would only add a second concurrent
  // producer for no gain.
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

  // Snapshot the frames buffered by addFrame, moving nothing: the copy is of owned
  // cv::Mat headers, so it is a refcount bump each, and it keeps the disk writes
  // below out of the critical section shared with the active window thread.
  std::vector<BufferedFrame> frames;
  std::optional<CameraCalib> calib;
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    frames.assign(frames_.begin(), frames_.end());
    calib = calib_;
  }

  if (frames.empty()) {
    return;  // nothing to pair against yet
  }

  const auto newest_frame_ns = frames.back().timestamp_ns;

  // Get active nodes, sort by timestamp
  std::vector<const SceneGraphNode*> agent_nodes;
  for (const auto& [node_id, node] : layer->nodes()) {
    agent_nodes.push_back(node.get());
  }
  std::sort(agent_nodes.begin(), agent_nodes.end(), [](const auto& a, const auto& b) {
    return a->template attributes<AgentNodeAttributes>().timestamp.count() <
           b->template attributes<AgentNodeAttributes>().timestamp.count();
  });

  // Each buffered frame belongs to at most one agent node; nodes are visited in time
  // order, so consumed frames are always a prefix of the snapshot.
  size_t search_start = 0;

  for (const auto* node : agent_nodes) {
    auto& attrs = node->template attributes<AgentNodeAttributes>();

    // Skip if already processed
    if (!attrs.image_folder.empty()) {
      continue;
    }

    const auto node_ns = static_cast<uint64_t>(attrs.timestamp.count());
    if (node_ns <= last_decided_ns_) {
      continue;  // already decided on a previous call
    }

    // Pair the node with the sensor frame it was actually created from. Attaching
    // whatever frame happens to be current instead silently gives every node of a
    // collated backlog the same image at the wrong pose, which cannot be detected
    // downstream.
    const auto frame_idx = findFrame(frames, search_start, node_ns);
    if (!frame_idx) {
      if (node_ns > newest_frame_ns) {
        // The frame stream has not reached this node yet, so its frame may still be
        // coming. Wait for it -- but only for a bounded number of updates: if it
        // never arrives, everything behind this node would stall forever.
        if (node_ns != deferred_node_ns_) {
          deferred_node_ns_ = node_ns;
          deferred_count_ = 0;
        }

        if (++deferred_count_ <= config.max_deferred_updates) {
          break;  // later nodes are newer still, so nothing behind it can pair either
        }

        LOG_EVERY_N(WARNING, 10)
            << "[AgentImageExtractor] Agent " << spark_dsg::NodeSymbol(node->id).str()
            << " @ " << node_ns << " ns waited " << deferred_count_
            << " updates for a sensor frame that never arrived; abandoning it so the "
               "queue can drain.";
      }

      // Either the frame stream has already moved past this node or we gave up
      // waiting. A missing keyframe is honest; a mispaired one is not.
      last_decided_ns_ = node_ns;
      deferred_node_ns_ = 0;
      deferred_count_ = 0;
      VLOG(2) << "[AgentImageExtractor] No sensor frame within "
              << config.max_pairing_time_diff_s << " s of agent "
              << spark_dsg::NodeSymbol(node->id).str() << " @ " << node_ns
              << " ns; skipping keyframe.";
      continue;
    }

    const auto& frame = frames[*frame_idx];
    search_start = *frame_idx + 1;
    last_decided_ns_ = node_ns;
    deferred_node_ns_ = 0;
    deferred_count_ = 0;

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

    if (!should_trigger) {
      continue;
    }

    // Update state
    last_keyframe_.position = current_pos;
    last_keyframe_.orientation = current_rot;
    last_keyframe_.initialized = true;

    const std::string name = "agent_" + std::to_string(attrs.timestamp.count());
    std::filesystem::path base_path =
        std::filesystem::path(config.image_output_path) / name;

    // Write the run-level calibration once. It is constant for a fixed camera, so we
    // keep it out of the per-keyframe metadata.
    if (!calib_written_ && calib) {
      std::filesystem::path calib_path =
          std::filesystem::path(config.image_output_path) / "camera_calib.json";
      std::ofstream out(calib_path);
      out << std::setprecision(17);
      out << "{\n";
      out << "  \"fx\": " << calib->fx << ",\n";
      out << "  \"fy\": " << calib->fy << ",\n";
      out << "  \"cx\": " << calib->cx << ",\n";
      out << "  \"cy\": " << calib->cy << ",\n";
      out << "  \"width\": " << calib->width << ",\n";
      out << "  \"height\": " << calib->height << ",\n";
      out << "  \"depth_scale\": " << kDepthScaleMetersPerUnit << ",\n";
      out << "  \"depth_encoding\": \"" << kDepthEncoding << "\",\n";
      out << "  \"body_T_sensor\": " << isometryToJsonArray(calib->body_T_sensor)
          << "\n";
      out << "}\n";
      calib_written_ = true;
    }

    // Images are already in their on-disk form (see addFrame).
    if (!frame.color_bgr.empty()) {
      cv::imwrite(base_path.string() + "_rgb.jpg", frame.color_bgr);
    }

    if (!frame.depth_mm.empty()) {
      cv::imwrite(base_path.string() + "_depth.png", frame.depth_mm);
    }

    // Per-keyframe metadata: dynamic data only (pose + file references). Static
    // calibration lives in camera_calib.json. frame_timestamp_ns records which
    // sensor frame the imagery came from, so a mispairing is auditable offline.
    {
      std::ofstream meta(base_path.string() + "_meta.json");
      meta << "{\n";
      meta << "  \"timestamp_ns\": " << attrs.timestamp.count() << ",\n";
      meta << "  \"frame_timestamp_ns\": " << frame.timestamp_ns << ",\n";
      meta << "  \"world_T_body\": " << isometryToJsonArray(frame.world_T_body)
           << ",\n";
      meta << "  \"rgb_file\": \"" << name << "_rgb.jpg\",\n";
      meta << "  \"depth_file\": \"" << name << "_depth.png\",\n";
      meta << "  \"calib\": \"camera_calib.json\"\n";
      meta << "}\n";
    }

    attrs.image_folder = base_path.string();

    VLOG(3) << "[AgentImageExtractor] Triggered keyframe extraction for agent "
            << spark_dsg::NodeSymbol(node->id).str() << " @ "
            << attrs.timestamp.count() << " ns";
  }

  // Frames older than the last node we decided on can never be paired with anything.
  if (last_decided_ns_) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    while (!frames_.empty() && frames_.front().timestamp_ns <= last_decided_ns_) {
      frames_.pop_front();
    }
  }
}

}  // namespace hydra
