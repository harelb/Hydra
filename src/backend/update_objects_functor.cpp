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
#include "hydra/backend/update_objects_functor.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/printing.h>

#include <filesystem>

#include "hydra/backend/backend_utilities.h"
#include "hydra/utils/mesh_utilities.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {
namespace {

static const auto reg = config::RegistrationWithConfig<UpdateFunctor,
                                                       UpdateObjectsFunctor,
                                                       UpdateObjectsFunctor::Config>(
    "UpdateObjectsFunctor");

}

using kimera_pgmo::MeshOffsetInfo;
using timing::ScopedTimer;

using SemanticLabel = SemanticNodeAttributes::Label;
using MergeId = std::optional<NodeId>;

NodeAttributes::Ptr mergeObjectAttributes(const VerbosityConfig& config,
                                          const DynamicSceneGraph& graph,
                                          const std::vector<NodeId>& nodes) {
  if (nodes.empty()) {
    return nullptr;
  }

  auto iter = nodes.begin();
  auto attrs_ptr = graph.getNode(*iter).attributes().clone();
  auto& new_attrs =
      *CHECK_NOTNULL(dynamic_cast<spark_dsg::ObjectNodeAttributes*>(attrs_ptr.get()));

  // Cast to KhronosObjectAttributes to access image_folder
  auto* new_khronos_attrs =
      dynamic_cast<spark_dsg::KhronosObjectAttributes*>(&new_attrs);

  ++iter;
  while (iter != nodes.end()) {
    const auto& from_attrs = graph.getNode(*iter).attributes<ObjectNodeAttributes>();
    utils::mergeIndices(from_attrs.mesh_connections, new_attrs.mesh_connections);

    // Merge images logic
    if (new_khronos_attrs) {
      const auto* from_khronos_attrs =
          dynamic_cast<const spark_dsg::KhronosObjectAttributes*>(&from_attrs);

      if (from_khronos_attrs && !from_khronos_attrs->image_folder.empty()) {
        if (new_khronos_attrs->image_folder.empty()) {
          // If target has no folder, adopt the source's folder.
          new_khronos_attrs->image_folder = from_khronos_attrs->image_folder;
        } else {
          // Move files from source (which is being merged/deleted) to target
          // (new_attrs). Both paths might be absolute or relative (relative to
          // ADT4_OUTPUT_DIR). We assume they are valid paths.
          std::filesystem::path src_path(from_khronos_attrs->image_folder);
          std::filesystem::path dest_path(new_khronos_attrs->image_folder);

          // If paths are same, nothing to do.
          if (src_path != dest_path && std::filesystem::exists(src_path)) {
            try {
              if (!std::filesystem::exists(dest_path)) {
                std::filesystem::create_directories(dest_path);
              }

              // Iterate and move
              for (const auto& entry : std::filesystem::directory_iterator(src_path)) {
                std::filesystem::path src_file = entry.path();
                std::filesystem::path dest_file = dest_path / src_file.filename();
                // Merge: overwrite or keep?
                // Khronos frames have unique timestamps usually, so overwrite/move
                // should be fine. If conflict, we might want to keep the one already
                // there, or overwrite. Let's overwrite/rename.
                std::filesystem::rename(src_file, dest_file);
              }
              // Source should modify its image folder? No, source is being merged away.
              // We should delete the old folder.
              std::filesystem::remove_all(src_path);
            } catch (const std::exception& e) {
              LOG(WARNING) << "Failed to merge image folders: " << e.what();
            }
          }
        }
      }
    }

    ++iter;
  }

  if (new_attrs.mesh_connections.empty()) {
    MLOG(1) << "merge is empty: " << displayNodeSymbolContainer(nodes);
    return attrs_ptr;
  }

  auto mesh = graph.mesh();
  if (!updateObjectGeometry(*mesh, new_attrs)) {
    MLOG(1) << "merge geometry invalid: " << displayNodeSymbolContainer(nodes);
  }

  return attrs_ptr;
}

void declare_config(UpdateObjectsFunctor::Config& config) {
  using namespace config;
  name("UpdateObjectsFunctor::Config");
  base<VerbosityConfig>(config);
  field(config.allow_connection_merging, "allow_connection_merging");
  field(config.merge_proposer, "merge_proposer");
}

UpdateObjectsFunctor::UpdateObjectsFunctor(const Config& config)
    : config(config::checkValid(config)), merge_proposer(config.merge_proposer) {}

UpdateFunctor::Hooks UpdateObjectsFunctor::hooks() const {
  auto my_hooks = UpdateFunctor::hooks();
  my_hooks.find_merges = [this](const auto& graph, const auto& info) {
    return findMerges(graph, info);
  };

  if (config.allow_connection_merging) {
    my_hooks.merge = [this](const auto& graph, const auto& nodes) {
      merged_nodes_.insert(nodes.begin(), nodes.end());
      return mergeObjectAttributes(config, graph, nodes);
    };

    // merging indices means that we have archived objects with active vertices
    my_hooks.mesh_update = [this](const auto& graph, const auto& offsets) {
      updateMeshIndices(graph, offsets);
    };
  }

  return my_hooks;
}

void UpdateObjectsFunctor::call(const DynamicSceneGraph& unmerged,
                                SharedDsgInfo& dsg,
                                const UpdateInfo::ConstPtr& info) const {
  ScopedTimer spin_timer("backend/update_objects", info->timestamp_ns);
  if (!unmerged.hasLayer(DsgLayers::OBJECTS)) {
    MLOG(2) << "skipping object update due to missing layer";
    return;
  }

  // we want to use the unmerged graph for most things
  const auto& objects = unmerged.getLayer(DsgLayers::OBJECTS);

  VLOG(5) << "UpdateObjectsFunctor running on " << objects.nodes().size() << " nodes.";

  // we want to iterate over the unmerged graph
  const auto new_loopclosure = info->loop_closure_detected;
  active_tracker.clear();  // reset from previous pass
  LayerView view = new_loopclosure ? LayerView(objects) : active_tracker.view(objects);

  // apply updates to every attribute that may have changed since the last call
  size_t num_changed = 0;
  // we want to use the optimized mesh (unmerged doesn't have a mesh)
  const auto mesh = dsg.graph->mesh();

  // Iterate over ALL nodes to ensure folder consistency, not just active ones.
  // The overhead is minimal compared to the IO operations if we check existence first.
  // TODO(harel): Optimally we should only do this for new/changed nodes, but tracking
  // 'changed folder' is hard.
  for (const auto& id_node_pair : objects.nodes()) {
    const auto& node = *id_node_pair.second;
    ++num_changed;
    auto attrs = node.tryAttributes<spark_dsg::ObjectNodeAttributes>();
    if (!attrs) {
      continue;  // not an object
    }

    // Clone attributes first so we can modify them for the backend
    auto new_attrs_ptr = attrs->clone();
    auto* new_attrs =
        dynamic_cast<spark_dsg::ObjectNodeAttributes*>(new_attrs_ptr.get());
    if (!new_attrs) {
      LOG(ERROR) << "Failed to cast cloned attributes to ObjectNodeAttributes";
      continue;
    }

    // Check for image folder update (Move from temp -> final)
    if (auto* khronos_attrs =
            dynamic_cast<spark_dsg::KhronosObjectAttributes*>(new_attrs)) {
      if (!khronos_attrs->image_folder.empty()) {
        std::filesystem::path current_path(khronos_attrs->image_folder);

        NodeSymbol sym(node.id);
        std::string new_dir_name =
            std::string(1, sym.category()) + "_" + std::to_string(sym.categoryId());
        // Expected final relative path
        std::string expected_relative = "images/" + new_dir_name;

        // Check if we are currently in "temp"
        bool is_in_temp = current_path.string().find("/temp/") != std::string::npos ||
                          current_path.string().find("temp/") !=
                              std::string::npos;  // simplistic check

        // If path is not the expected final one
        if (khronos_attrs->image_folder != expected_relative) {
          VLOG(2) << "Updating image folder for node " << sym.str() << ": "
                  << khronos_attrs->image_folder << " -> " << expected_relative;

          std::filesystem::path parent;
          // We need to find the "root" images directory.
          // If current path is absolute, we can try to deduce it.
          if (current_path.is_absolute()) {
            if (is_in_temp) {
              // Structure: .../images/temp/O_track
              // We want:   .../images/O_node
              parent = current_path.parent_path().parent_path();
            } else {
              // Structure: .../images/O_track (legacy/fallback)
              parent = current_path.parent_path();
            }

            std::filesystem::path new_path = parent / new_dir_name;

            bool source_exists = std::filesystem::exists(current_path);

            if (source_exists) {
              try {
                if (!std::filesystem::exists(new_path)) {
                  std::filesystem::create_directories(new_path);
                }

                // Move contents
                for (const auto& entry :
                     std::filesystem::directory_iterator(current_path)) {
                  std::filesystem::path src_file = entry.path();
                  std::filesystem::path dest_file = new_path / src_file.filename();
                  // Overwrite/Rename
                  std::filesystem::rename(src_file, dest_file);
                }

                // Remove source folder
                std::filesystem::remove_all(current_path);
                VLOG(2) << "Moved images from " << current_path << " to " << new_path;

              } catch (const std::exception& e) {
                LOG(WARNING) << "Failed to move object images from " << current_path
                             << " to " << new_path << ": " << e.what();
              }
            } else {
              VLOG(5) << "Source path " << current_path
                      << " does not exist, skipping move.";
            }
          }

          // Update attribute to standard relative path
          khronos_attrs->image_folder = expected_relative;
        }
      }
    }

    MLOG(5) << "processing object " << NodeSymbol(node.id).str()
            << " with attributes:\n"
            << *new_attrs;
    if (new_attrs->mesh_connections.empty()) {
      MLOG(2) << "found empty object node " << NodeSymbol(node.id).str();
      continue;
    }

    if (!updateObjectGeometry(*mesh, *new_attrs)) {
      MLOG(2) << "invalid centroid for object " << NodeSymbol(node.id).str();
    }

    // TODO(nathan) this is sloppy and needs to be cleaned up
    dsg.graph->setNodeAttributes(node.id, std::move(new_attrs_ptr));
  }

  // Garbage Collection for Orphaned Folders is risky if we are mid-move or if temp is
  // used. We should NOT touch 'temp' folder. We should ONLY clean 'images/O_X' where
  // 'O_X' is invalid.
  const char* output_dir_env = std::getenv("ADT4_OUTPUT_DIR");
  if (output_dir_env) {
    std::filesystem::path images_root =
        std::filesystem::path(output_dir_env) / "images";
    if (std::filesystem::exists(images_root)) {
      // Build set of valid directory names from the BACKEND graph (dsg.graph)
      // The unmerged graph might contain tracks that have been merged/pruned in the
      // backend. We want the folders to match the persistable backend state.
      std::set<std::string> valid_names;
      if (dsg.graph && dsg.graph->hasLayer(DsgLayers::OBJECTS)) {
        const auto& backend_objects = dsg.graph->getLayer(DsgLayers::OBJECTS);
        for (const auto& id_node_pair : backend_objects.nodes()) {
          NodeSymbol sym(id_node_pair.first);
          valid_names.insert(std::string(1, sym.category()) + "_" +
                             std::to_string(sym.categoryId()));
        }
      }

      for (const auto& entry : std::filesystem::directory_iterator(images_root)) {
        if (entry.is_directory()) {
          std::string dir_name = entry.path().filename().string();
          // Skip 'temp' directory!
          if (dir_name == "temp") continue;

          // Check if it looks like an object folder O_<digits>
          if (dir_name.size() > 2 && dir_name.rfind("O_", 0) == 0) {
            if (valid_names.find(dir_name) == valid_names.end()) {
              try {
                std::filesystem::remove_all(entry.path());
                LOG(INFO) << "Deleted orphaned object folder: " << entry.path();
              } catch (const std::exception& e) {
                LOG(WARNING) << "Failed to delete orphan: " << e.what();
              }
            }
          }
        }
      }
    }
  }
  MLOG(1) << "object update: " << num_changed << " node(s)";
}

MergeList UpdateObjectsFunctor::findMerges(const DynamicSceneGraph& graph,
                                           const UpdateInfo::ConstPtr& info) const {
  if (!graph.hasLayer(DsgLayers::OBJECTS)) {
    return {};
  }

  const auto new_lcd = info->loop_closure_detected;
  const auto& objects = graph.getLayer(DsgLayers::OBJECTS);
  // freeze layer view to avoid messing with tracker
  LayerView view = new_lcd ? LayerView(objects) : active_tracker.view(objects, true);

  MergeList proposals;
  merge_proposer.findMerges(
      objects,
      view,
      [](const SceneGraphNode& lhs, const SceneGraphNode& rhs) {
        const auto lhs_attrs = lhs.tryAttributes<spark_dsg::ObjectNodeAttributes>();
        const auto rhs_attrs = rhs.tryAttributes<spark_dsg::ObjectNodeAttributes>();
        if (!lhs_attrs || !rhs_attrs) {
          return false;
        }

        return lhs_attrs->bounding_box.contains(rhs_attrs->position) ||
               rhs_attrs->bounding_box.contains(lhs_attrs->position);
      },
      proposals);
  return proposals;
}

void UpdateObjectsFunctor::updateMeshIndices(const DynamicSceneGraph& graph,
                                             const MeshOffsetInfo& offsets) const {
  const auto objects = graph.findLayer(config.layer);
  if (!objects) {
    return;
  }

  MLOG(2) << "remapping indices for " << merged_nodes_.size() << " merged node(s)";
  auto iter = merged_nodes_.begin();
  while (iter != merged_nodes_.end()) {
    auto node = objects->findNode(*iter);
    if (!node) {
      iter = merged_nodes_.erase(iter);
      continue;
    }

    auto attrs = node->tryAttributes<ObjectNodeAttributes>();
    if (!attrs) {
      iter = merged_nodes_.erase(iter);
      continue;
    }

    kimera_pgmo::MeshOffsetInfo::RemapStats stats;
    offsets.remapVertexIndices(attrs->mesh_connections, &stats);
    if (stats.all_archived) {
      iter = merged_nodes_.erase(iter);
    } else {
      ++iter;
    }
  }

  MLOG(2) << merged_nodes_.size() << " merged node(s)";
}

}  // namespace hydra
