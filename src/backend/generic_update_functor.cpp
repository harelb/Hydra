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
#include "hydra/backend/generic_update_functor.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/scene_graph_types.h>

#include <filesystem>

#include "hydra/utils/timing_utilities.h"

namespace hydra {
namespace {

static const auto registration =
    config::RegistrationWithConfig<UpdateFunctor,
                                   GenericUpdateFunctor,
                                   GenericUpdateFunctor::Config>(
        "GenericUpdateFunctor");

using spark_dsg::KhronosObjectAttributes;
using spark_dsg::NodeSymbol;

void moveImageFiles(const std::filesystem::path& src,
                    const std::filesystem::path& dest) {
  if (src == dest || !std::filesystem::exists(src)) {
    return;
  }
  if (!std::filesystem::exists(dest)) {
    std::filesystem::create_directories(dest);
  }
  size_t moved = 0;
  for (const auto& entry : std::filesystem::directory_iterator(src)) {
    try {
      std::filesystem::rename(entry.path(), dest / entry.path().filename());
      ++moved;
    } catch (const std::exception& e) {
      LOG(WARNING) << "[GenericUpdateFunctor] failed to move " << entry.path() << ": "
                   << e.what();
    }
  }
  LOG(INFO) << "[image-union] moved " << moved << " files " << src << " -> " << dest;
  try {
    std::filesystem::remove(src);
  } catch (...) {
  }
}

// Final crop directory for a node, matching the temp->final rename in call().
std::filesystem::path finalImagePath(const std::filesystem::path& images_root,
                                     NodeId node) {
  const NodeSymbol sym(node);
  return images_root /
         (std::string(1, sym.category()) + "_" + std::to_string(sym.categoryId()));
}

// Consolidate image folders when nodes merge. nodes[0] is the surviving node.
// The attributes come from the odometric unmerged graph, whose image_folder paths
// still point at the frontend's temp dirs (only the merged graph sees the temp->final
// rename in call()); the crops actually live at each node's final path, so the union
// works on those and the stale pointer is rewritten to the surviving node's final path.
NodeAttributes::Ptr mergeKhronosImageFolders(const DynamicSceneGraph& graph,
                                             const std::vector<NodeId>& nodes) {
  if (nodes.empty()) {
    return nullptr;
  }
  const auto parent = graph.findNode(nodes[0]);
  if (!parent) {
    // the frontend deleted the surviving node from the odometric graph; nothing to
    // rebuild the merged attributes from
    return nullptr;
  }
  auto attrs_ptr = parent->attributes().clone();
  auto* surviving = dynamic_cast<KhronosObjectAttributes*>(attrs_ptr.get());
  if (!surviving) {
    LOG(INFO) << "[image-union] " << NodeSymbol(nodes[0]).str()
              << ": attrs are not KhronosObjectAttributes, skipping";
    return attrs_ptr;
  }

  const char* output_dir_env = std::getenv("ADT4_OUTPUT_DIR");
  if (!output_dir_env) {
    LOG(INFO) << "[image-union] ADT4_OUTPUT_DIR unset, skipping";
    return attrs_ptr;
  }
  const std::filesystem::path images_root =
      std::filesystem::path(output_dir_env) / "images";
  const auto dest = finalImagePath(images_root, nodes[0]);
  LOG(INFO) << "[image-union] parent=" << NodeSymbol(nodes[0]).str()
            << " children=" << (nodes.size() - 1) << " dest=" << dest
            << " dest_exists=" << std::filesystem::exists(dest);

  // covers a merge that lands before call() renamed the surviving node's temp dir
  if (!surviving->image_folder.empty()) {
    moveImageFiles(std::filesystem::path(surviving->image_folder), dest);
  }
  for (size_t i = 1; i < nodes.size(); ++i) {
    const auto child = graph.findNode(nodes[i]);
    const auto* from = child ? child->tryAttributes<KhronosObjectAttributes>() : nullptr;
    if (from && !from->image_folder.empty()) {
      moveImageFiles(std::filesystem::path(from->image_folder), dest);
    }
    const auto child_final = finalImagePath(images_root, nodes[i]);
    LOG(INFO) << "[image-union]   child=" << NodeSymbol(nodes[i]).str()
              << " attr=" << (from ? from->image_folder : "<not-khronos>")
              << " final=" << child_final
              << " final_exists=" << std::filesystem::exists(child_final);
    moveImageFiles(child_final, dest);
  }

  if (std::filesystem::exists(dest)) {
    surviving->image_folder = dest.string();
  }
  return attrs_ptr;
}

}  // namespace

using timing::ScopedTimer;

void declare_config(GenericUpdateFunctor::Config& config) {
  using namespace config;
  name("GenericUpdateFunctor::Config");
  base<VerbosityConfig>(config);
  field(config.layer, "layer");
  field(config.deformation_interpolator, "deformation_interpolator");
  field(config.enable_merging, "enable_merging");
  config.matcher.setOptional();
  field(config.matcher, "node_matcher");
  field(config.merge_proposer, "merge_proposer");
  checkCondition(!config.layer.empty(), "layer must be non-empty!");
}

GenericUpdateFunctor::GenericUpdateFunctor(const Config& config)
    : config(config::checkValid(config)),
      node_matcher(config.matcher.create()),
      merge_proposer(config.merge_proposer),
      deformation_interpolator(config.deformation_interpolator) {}

UpdateFunctor::Hooks GenericUpdateFunctor::hooks() const {
  auto my_hooks = UpdateFunctor::hooks();
  if (config.enable_merging && node_matcher) {
    my_hooks.find_merges = [this](const auto& graph, const auto& info) {
      return findMerges(graph, info);
    };

    if (config.layer == spark_dsg::DsgLayers::OBJECTS) {
      my_hooks.merge = [this](const auto& graph, const auto& nodes) {
        auto attrs = mergeKhronosImageFolders(graph, nodes);
        if (attrs) {
          // the merged attributes are cloned from the odometric unmerged graph;
          // re-apply the surviving node's last deformation so the merge result
          // stays in the optimized frame
          deformation_interpolator.applyLastTransform(nodes.front(), *attrs);
        }
        return attrs;
      };
    }
  }

  return my_hooks;
}

void GenericUpdateFunctor::call(const DynamicSceneGraph& unmerged,
                                SharedDsgInfo& dsg,
                                const UpdateInfo::ConstPtr& info) const {
  ScopedTimer spin_timer("backend/update_" + config.layer, info->timestamp_ns);
  if (!unmerged.hasLayer(config.layer)) {
    return;
  }

  const auto new_loopclosure = info->loop_closure_detected;
  const auto& layer = unmerged.getLayer(config.layer);
  active_tracker.clear();  // reset from previous pass
  const auto view = new_loopclosure ? LayerView(layer) : active_tracker.view(layer);
  deformation_interpolator.interpolateNodePositions(unmerged, *dsg.graph, info, view);
  MLOG(1) << "[Hydra Backend] " << config.layer << " update: " << layer.numNodes()
          << " nodes";

  if (config.layer != spark_dsg::DsgLayers::OBJECTS) {
    return;
  }
  if (!dsg.graph->hasLayer(config.layer)) {
    return;
  }
  const char* output_dir_env = std::getenv("ADT4_OUTPUT_DIR");
  if (!output_dir_env) {
    return;
  }
  const std::filesystem::path images_root =
      std::filesystem::path(output_dir_env) / "images";

  // the frontend's temp pointers live on the UNMERGED graph (the merged copy of an
  // archived node is never refreshed by mergeGraph, so it can't be trusted for
  // bookkeeping); rename on disk and mirror the final path onto the merged copy,
  // where it sticks precisely because archived attributes are never overwritten
  for (const auto& [node_id, node] : unmerged.getLayer(config.layer).nodes()) {
    const auto* khronos = node->tryAttributes<KhronosObjectAttributes>();
    if (!khronos || khronos->image_folder.empty()) {
      continue;
    }
    const auto final_path = finalImagePath(images_root, node_id);
    const std::filesystem::path current(khronos->image_folder);
    if (current.string().find("/temp/") != std::string::npos) {
      moveImageFiles(current, final_path);
    }

    const auto target_node = dsg.graph->findNode(node_id);
    if (!target_node) {
      continue;
    }
    const auto* target_attrs = target_node->tryAttributes<KhronosObjectAttributes>();
    if (!target_attrs || target_attrs->image_folder == final_path.string()) {
      continue;
    }
    auto attrs_ptr = target_node->attributes().clone();
    dynamic_cast<KhronosObjectAttributes*>(attrs_ptr.get())->image_folder =
        final_path.string();
    dsg.graph->setNodeAttributes(node_id, std::move(attrs_ptr));
  }
}

MergeList GenericUpdateFunctor::findMerges(const DynamicSceneGraph& graph,
                                           const UpdateInfo::ConstPtr& info) const {
  if (!node_matcher) {
    LOG(WARNING) << "No node matcher!";
    return {};
  }

  const auto new_lcd = info->loop_closure_detected;
  const auto& layer = graph.getLayer(config.layer);
  // freeze layer view to avoid messing with tracker
  const auto view = new_lcd ? LayerView(layer) : active_tracker.view(layer, true);

  MergeList proposals;
  merge_proposer.findMerges(
      layer,
      view,
      [this](const SceneGraphNode& lhs, const SceneGraphNode& rhs) {
        return node_matcher->match(lhs.attributes(), rhs.attributes());
      },
      proposals);
  return proposals;
}

}  // namespace hydra
