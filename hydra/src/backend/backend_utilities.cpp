/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
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
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hydra/backend/backend_utilities.h"

#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph_types.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace hydra::utils {

using spark_dsg::AgentNodeAttributes;
using spark_dsg::DsgLayers;
using spark_dsg::KhronosObjectAttributes;
using spark_dsg::NodeSymbol;
using spark_dsg::SceneGraph;

namespace {

// Directory under $ADT4_OUTPUT_DIR that the frontend extractors write into, used as a
// fallback when no populated node is available to derive the directory from.
std::filesystem::path outputSubdir(const char* subdir) {
  const char* out = std::getenv("ADT4_OUTPUT_DIR");
  if (!out) {
    return {};
  }
  return std::filesystem::path(out) / subdir;
}

}  // namespace

std::optional<uint64_t> getTimeNs(const SceneGraph& graph, gtsam::Symbol key) {
  NodeSymbol node(key.chr(), key.index());
  if (!graph.hasNode(node)) {
    LOG(ERROR) << "Missing node << " << node.str() << "when logging loop closure";
    return std::nullopt;
  }

  return graph.getNode(node).attributes<AgentNodeAttributes>().timestamp.count();
}

size_t reconcileAgentImageFolders(SceneGraph& graph) {
  const auto agents_key = graph.getLayerKey(DsgLayers::AGENTS);
  if (!agents_key) {
    return 0;
  }
  const auto agents_layer = agents_key->layer;

  // Derive the agents directory from an already-populated agent node (robust to the
  // configured output path), falling back to $ADT4_OUTPUT_DIR/agents.
  std::filesystem::path agents_dir;
  for (const auto& [partition, layer] : graph.layer_partition(agents_layer)) {
    for (const auto& [node_id, node] : layer->nodes()) {
      const auto* attrs = node->tryAttributes<AgentNodeAttributes>();
      if (attrs && !attrs->image_folder.empty()) {
        agents_dir = std::filesystem::path(attrs->image_folder).parent_path();
        break;
      }
    }
    if (!agents_dir.empty()) {
      break;
    }
  }
  if (agents_dir.empty()) {
    agents_dir = outputSubdir("agents");
  }
  if (agents_dir.empty()) {
    return 0;
  }

  size_t filled = 0;
  for (const auto& [partition, layer] : graph.layer_partition(agents_layer)) {
    for (const auto& [node_id, node] : layer->nodes()) {
      auto* attrs = node->tryAttributes<AgentNodeAttributes>();
      if (!attrs || !attrs->image_folder.empty()) {
        continue;
      }
      // The extractor names files agent_<timestamp_ns>; the node timestamp is the same
      // value, so the prefix is reconstructed exactly (no pose/index guessing).
      const auto prefix =
          agents_dir / ("agent_" + std::to_string(attrs->timestamp.count()));
      if (std::filesystem::exists(prefix.string() + "_meta.json")) {
        attrs->image_folder = prefix.string();
        ++filled;
      }
    }
  }
  if (filled) {
    LOG(INFO) << "[reconcileAgentImageFolders] restored image_folder on " << filled
              << " agent node(s) from " << agents_dir;
  }
  return filled;
}

size_t reconcileObjectImageFolders(SceneGraph& graph) {
  if (!graph.hasLayer(DsgLayers::OBJECTS)) {
    return 0;
  }
  const auto& layer = graph.getLayer(DsgLayers::OBJECTS);

  std::filesystem::path images_dir;
  for (const auto& [node_id, node] : layer.nodes()) {
    const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (attrs && !attrs->image_folder.empty()) {
      images_dir = std::filesystem::path(attrs->image_folder).parent_path();
      break;
    }
  }
  if (images_dir.empty()) {
    images_dir = outputSubdir("images");
  }
  if (images_dir.empty()) {
    return 0;
  }

  size_t filled = 0;
  for (const auto& [node_id, node] : layer.nodes()) {
    auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs || !attrs->image_folder.empty()) {
      continue;
    }
    const NodeSymbol sym(node_id);
    const auto folder =
        images_dir /
        (std::string(1, sym.category()) + "_" + std::to_string(sym.categoryId()));
    if (std::filesystem::exists(folder)) {
      attrs->image_folder = folder.string();
      ++filled;
    }
  }
  if (filled) {
    LOG(INFO) << "[reconcileObjectImageFolders] restored image_folder on " << filled
              << " object node(s) from " << images_dir;
  }
  return filled;
}

}  // namespace hydra::utils
