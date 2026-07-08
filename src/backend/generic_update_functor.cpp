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
  if (!std::filesystem::exists(src)) {
    return;
  }
  if (!std::filesystem::exists(dest)) {
    std::filesystem::create_directories(dest);
  }
  for (const auto& entry : std::filesystem::directory_iterator(src)) {
    try {
      std::filesystem::rename(entry.path(), dest / entry.path().filename());
    } catch (const std::exception& e) {
      LOG(WARNING) << "[GenericUpdateFunctor] failed to move " << entry.path() << ": "
                   << e.what();
    }
  }
  try {
    std::filesystem::remove(src);
  } catch (...) {
  }
}

// Consolidate image folders when nodes merge. nodes[0] is the surviving node.
NodeAttributes::Ptr mergeKhronosImageFolders(const DynamicSceneGraph& graph,
                                             const std::vector<NodeId>& nodes) {
  if (nodes.empty()) {
    return nullptr;
  }
  auto attrs_ptr = graph.getNode(nodes[0]).attributes().clone();
  auto* surviving = dynamic_cast<KhronosObjectAttributes*>(attrs_ptr.get());

  for (size_t i = 1; i < nodes.size(); ++i) {
    const auto* from = graph.getNode(nodes[i]).tryAttributes<KhronosObjectAttributes>();
    if (!from || from->image_folder.empty()) {
      continue;
    }
    if (!surviving) {
      break;
    }
    if (surviving->image_folder.empty()) {
      surviving->image_folder = from->image_folder;
    } else {
      moveImageFiles(std::filesystem::path(from->image_folder),
                     std::filesystem::path(surviving->image_folder));
    }
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

  const auto& backend_layer = dsg.graph->getLayer(config.layer);
  for (const auto& [node_id, node] : backend_layer.nodes()) {
    auto attrs_ptr = node->attributes().clone();
    auto* khronos = dynamic_cast<KhronosObjectAttributes*>(attrs_ptr.get());
    if (!khronos || khronos->image_folder.empty()) {
      continue;
    }
    const std::filesystem::path current(khronos->image_folder);
    if (current.string().find("/temp/") == std::string::npos) {
      continue;  // already renamed
    }
    NodeSymbol sym(node_id);
    const auto final_path = images_root / (std::string(1, sym.category()) + "_" +
                                           std::to_string(sym.categoryId()));
    moveImageFiles(current, final_path);
    khronos->image_folder = final_path.string();
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
