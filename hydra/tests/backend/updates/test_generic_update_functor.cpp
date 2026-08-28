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
#include <config_utilities/printing.h>
#include <glog/logging.h>
#include <glog/stl_logging.h>
#include <gtest/gtest.h>
#include <hydra/backend/merge_tracker.h>
#include <hydra/backend/updates/generic_update_functor.h>
#include <kimera_pgmo/deformation_graph.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "hydra_test/resources.h"
#include "hydra_test/shared_dsg_fixture.h"

using namespace spark_dsg;

namespace hydra {
namespace {

MergeList callWithUnmerged(const UpdateFunctor& functor,
                           SharedDsgInfo& dsg,
                           const UpdateInfo::ConstPtr& info,
                           bool enable_merging) {
  const auto unmerged = dsg.graph->clone();
  functor.call(*unmerged, dsg, info);
  const auto hooks = functor.hooks();
  if (enable_merging && hooks.find_merges) {
    return hooks.find_merges(*unmerged, info);
  } else {
    return {};
  }
}

GenericUpdateFunctor::Config defaultConfig() { return {5, "OBJECTS"}; }

// A throwaway directory that cleans itself up, exported as $ADT4_OUTPUT_DIR for the
// duration of a test.
struct ScopedOutputDir {
  std::filesystem::path path;
  std::string previous;
  bool had_previous;
  explicit ScopedOutputDir(const std::string& name)
      : path(std::filesystem::temp_directory_path() / ("hydra_merge_images_" + name)) {
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    const char* prev = std::getenv("ADT4_OUTPUT_DIR");
    had_previous = prev != nullptr;
    previous = prev ? prev : "";
    setenv("ADT4_OUTPUT_DIR", path.string().c_str(), 1);
  }
  ~ScopedOutputDir() {
    if (had_previous) {
      setenv("ADT4_OUTPUT_DIR", previous.c_str(), 1);
    } else {
      unsetenv("ADT4_OUTPUT_DIR");
    }
    std::filesystem::remove_all(path);
  }
};

void writeFile(const std::filesystem::path& p) {
  std::filesystem::create_directories(p.parent_path());
  std::ofstream(p) << "{}\n";
}

std::unique_ptr<KhronosObjectAttributes> makeObject(const std::string& image_folder) {
  auto attrs = std::make_unique<KhronosObjectAttributes>();
  attrs->position << 0.0, 0.0, 0.0;
  attrs->image_folder = image_folder;
  return attrs;
}

}  // namespace

TEST(GenericUpdateFunctor, noUpdate) {
  auto dsg = test::makeSharedDsg();
  auto& graph = *dsg->graph;

  const Eigen::Vector3d expected(1.0, 2.0, 3.0);
  {  // scope limiting moved attrs access
    auto attrs = std::make_unique<NodeAttributes>();
    attrs->position = expected;
    attrs->is_active = true;
    attrs->last_update_time_ns = 10u;
    graph.emplaceNode(DsgLayers::OBJECTS, 0, std::move(attrs));
  }

  UpdateInfo::ConstPtr info(new UpdateInfo{0, nullptr, nullptr, false, {}});
  auto config = defaultConfig();
  config.enable_merging = false;
  GenericUpdateFunctor functor(config);
  callWithUnmerged(functor, *dsg, info, false);

  // No deformation, so nothing should change
  const auto& result = graph.getNode(0).attributes();
  EXPECT_NEAR(0.0, (expected - result.position).norm(), 1.0e-7);
}

TEST(GenericUpdateFunctor, shouldUpdate) {
  auto dsg = test::makeSharedDsg();
  auto& graph = *dsg->graph;

  {  // scope limiting moved attrs access
    auto attrs = std::make_unique<NodeAttributes>();
    attrs->position << 0, 3, 1;
    attrs->is_active = true;
    attrs->last_update_time_ns = 10u;
    graph.emplaceNode(DsgLayers::OBJECTS, 0, std::move(attrs));
  }

  kimera_pgmo::DeformationGraph dgraph;
  dgraph.load(test::get_resource_path() / "graph.dgrf");

  UpdateInfo::ConstPtr info(new UpdateInfo{0, nullptr, nullptr, false, {}, &dgraph});
  auto config = defaultConfig();
  config.enable_merging = false;
  VLOG(1) << "Using config:\n" << config::toString(config);

  GenericUpdateFunctor functor(config);
  // the unmerged graph persists across spins and stays odometric; the functor only
  // ever writes the merged graph
  const auto unmerged = graph.clone();
  functor.call(*unmerged, *dsg, info);

  const auto& result = graph.getNode(0).attributes();
  const Eigen::Vector3d expected(1.0, 2.0, 3.0);
  EXPECT_NEAR(0.0, (expected - result.position).norm(), 1.0e-7);

  {  // the odometric source is never touched
    const Eigen::Vector3d odometric(0.0, 3.0, 1.0);
    const auto& src = unmerged->getNode(0).attributes();
    EXPECT_NEAR(0.0, (odometric - src.position).norm(), 1.0e-7);
  }

  // re-deforming an archived node must not compound
  graph.getNode(0).attributes().is_active = false;
  unmerged->getNode(0).attributes().is_active = false;
  functor.call(*unmerged, *dsg, info);
  EXPECT_NEAR(0.0, (expected - result.position).norm(), 1.0e-7);
}

// The merge hook receives the odometric unmerged graph, whose image_folder attributes
// still point at the frontend's temp dirs; those dirs no longer exist because call()
// already moved the crops to the per-node final path <output>/images/<layer>_<id>.
// The union has to happen between the on-disk final dirs, not the stale attrs.
TEST(GenericUpdateFunctor, mergeUnionsImageFoldersFromFinalPaths) {
  ScopedOutputDir tmp("union");
  const auto images_root = tmp.path / "images";
  writeFile(images_root / "O_0" / "crop_a.png");
  writeFile(images_root / "O_1" / "crop_b.png");

  auto dsg = test::makeSharedDsg();
  auto& unmerged = *dsg->graph;  // plays the odometric source graph
  unmerged.emplaceNode(DsgLayers::OBJECTS,
                       NodeSymbol('O', 0),
                       makeObject((images_root / "temp" / "uuid0").string()));
  unmerged.emplaceNode(DsgLayers::OBJECTS,
                       NodeSymbol('O', 1),
                       makeObject((images_root / "temp" / "uuid1").string()));

  GenericUpdateFunctor functor(defaultConfig());
  const auto hooks = functor.hooks();
  ASSERT_TRUE(hooks.merge != nullptr);

  const std::vector<NodeId> nodes{NodeSymbol('O', 0), NodeSymbol('O', 1)};
  auto attrs = hooks.merge(unmerged, nodes);
  ASSERT_TRUE(attrs != nullptr);
  const auto* merged = dynamic_cast<const KhronosObjectAttributes*>(attrs.get());
  ASSERT_TRUE(merged != nullptr);

  EXPECT_EQ(merged->image_folder, (images_root / "O_0").string());
  EXPECT_TRUE(std::filesystem::exists(images_root / "O_0" / "crop_a.png"));
  EXPECT_TRUE(std::filesystem::exists(images_root / "O_0" / "crop_b.png"));
  EXPECT_FALSE(std::filesystem::exists(images_root / "O_1"));

  // updateAllMergeAttributes re-invokes the hook for every merge set on each loop
  // closure, so a second call must be a stable no-op
  auto attrs2 = hooks.merge(unmerged, nodes);
  const auto* merged2 = dynamic_cast<const KhronosObjectAttributes*>(attrs2.get());
  ASSERT_TRUE(merged2 != nullptr);
  EXPECT_EQ(merged2->image_folder, (images_root / "O_0").string());
  EXPECT_TRUE(std::filesystem::exists(images_root / "O_0" / "crop_a.png"));
  EXPECT_TRUE(std::filesystem::exists(images_root / "O_0" / "crop_b.png"));
}

// Same union, but through the full live path: MergeTracker::applyMerges on a merged
// graph with the functor's hook, exactly as DsgUpdater::callUpdateFunctions drives it.
TEST(GenericUpdateFunctor, mergeUnionsImageFoldersViaMergeTracker) {
  ScopedOutputDir tmp("tracker");
  const auto images_root = tmp.path / "images";
  writeFile(images_root / "O_0" / "crop_a.png");
  writeFile(images_root / "O_1" / "crop_b.png");

  auto dsg = test::makeSharedDsg();
  auto& merged = *dsg->graph;
  merged.emplaceNode(DsgLayers::OBJECTS,
                     NodeSymbol('O', 0),
                     makeObject((images_root / "O_0").string()));
  merged.emplaceNode(DsgLayers::OBJECTS,
                     NodeSymbol('O', 1),
                     makeObject((images_root / "O_1").string()));

  // odometric source graph still carries the stale temp paths
  const auto unmerged = merged.clone();
  unmerged->getNode(NodeSymbol('O', 0)).attributes<KhronosObjectAttributes>().image_folder =
      (images_root / "temp" / "O_0").string();
  unmerged->getNode(NodeSymbol('O', 1)).attributes<KhronosObjectAttributes>().image_folder =
      (images_root / "temp" / "O_1").string();

  GenericUpdateFunctor functor(defaultConfig());
  const auto hooks = functor.hooks();
  ASSERT_TRUE(hooks.merge != nullptr);

  MergeTracker tracker;
  MergeList proposals{{NodeSymbol('O', 1), NodeSymbol('O', 0)}};
  const auto applied = tracker.applyMerges(*unmerged, proposals, *dsg, hooks.merge);
  EXPECT_EQ(applied, 1u);

  const auto& attrs =
      merged.getNode(NodeSymbol('O', 0)).attributes<KhronosObjectAttributes>();
  EXPECT_EQ(attrs.image_folder, (images_root / "O_0").string());
  EXPECT_TRUE(std::filesystem::exists(images_root / "O_0" / "crop_a.png"));
  EXPECT_TRUE(std::filesystem::exists(images_root / "O_0" / "crop_b.png"));
  EXPECT_FALSE(std::filesystem::exists(images_root / "O_1"));
}

// With object archival, the merged-graph copy of an archived node is never refreshed
// by mergeGraph, so the temp->final rename must key off the UNMERGED graph (which the
// backend now syncs with update_archived_attributes) and mirror the final path onto
// the merged copy where it sticks. Regression for box_9: 0 renames, all folders unset.
TEST(GenericUpdateFunctor, renameReadsUnmergedAndWritesMerged) {
  ScopedOutputDir tmp("rename_unmerged");
  const auto images_root = tmp.path / "images";
  writeFile(images_root / "temp" / "O_track7" / "crop_a.png");

  auto dsg = test::makeSharedDsg();
  auto& merged = *dsg->graph;
  // archived merged copy with stale (empty) folder: mergeGraph never refreshed it
  {
    auto attrs = makeObject("");
    attrs->is_active = false;
    merged.emplaceNode(DsgLayers::OBJECTS, NodeSymbol('O', 0), std::move(attrs));
  }
  // unmerged copy carries the frontend's temp pointer
  const auto unmerged = merged.clone();
  {
    auto attrs = makeObject((images_root / "temp" / "O_track7").string());
    attrs->is_active = false;
    unmerged->setNodeAttributes(NodeSymbol('O', 0), std::move(attrs));
  }

  auto config = defaultConfig();
  config.enable_merging = false;
  GenericUpdateFunctor functor(config);
  UpdateInfo::ConstPtr info(new UpdateInfo{0, nullptr, nullptr, false, {}});
  functor.call(*unmerged, *dsg, info);

  EXPECT_TRUE(std::filesystem::exists(images_root / "O_0" / "crop_a.png"));
  EXPECT_FALSE(std::filesystem::exists(images_root / "temp" / "O_track7"));
  const auto& attrs =
      merged.getNode(NodeSymbol('O', 0)).attributes<KhronosObjectAttributes>();
  EXPECT_EQ(attrs.image_folder, (images_root / "O_0").string());
}

// The frontend can DELETE object nodes (GraphUpdater delete updates), so a merge
// parent recorded in the tracker may vanish from the odometric unmerged graph. The
// tracker must drop that merge set instead of letting the hook throw std::out_of_range
// (live crash in box_8: "missing node 'O(15)'" inside updateAllMergeAttributes).
TEST(GenericUpdateFunctor, mergeTrackerSurvivesDeletedParent) {
  ScopedOutputDir tmp("deleted_parent");
  const auto images_root = tmp.path / "images";
  writeFile(images_root / "O_0" / "crop_a.png");
  writeFile(images_root / "O_1" / "crop_b.png");

  auto dsg = test::makeSharedDsg();
  auto& merged = *dsg->graph;
  merged.emplaceNode(DsgLayers::OBJECTS,
                     NodeSymbol('O', 0),
                     makeObject((images_root / "O_0").string()));
  merged.emplaceNode(DsgLayers::OBJECTS,
                     NodeSymbol('O', 1),
                     makeObject((images_root / "O_1").string()));
  const auto unmerged = merged.clone();

  GenericUpdateFunctor functor(defaultConfig());
  const auto hooks = functor.hooks();
  ASSERT_TRUE(hooks.merge != nullptr);

  MergeTracker tracker;
  MergeList proposals{{NodeSymbol('O', 1), NodeSymbol('O', 0)}};
  ASSERT_EQ(tracker.applyMerges(*unmerged, proposals, *dsg, hooks.merge), 1u);

  // frontend deletes the surviving node from the odometric graph
  unmerged->removeNode(NodeSymbol('O', 0));

  // both entry points must not throw and must drop the stale merge set
  EXPECT_NO_THROW(tracker.updateAllMergeAttributes(*unmerged, merged, hooks.merge));
  MergeList repeat{{NodeSymbol('O', 1), NodeSymbol('O', 0)}};
  EXPECT_NO_THROW(tracker.applyMerges(*unmerged, repeat, *dsg, hooks.merge));
}

// A child that never produced crops must not invent folders, and a surviving node
// with no crops of its own still adopts the union of its children.
TEST(GenericUpdateFunctor, mergeUnionsImageFoldersChildOnly) {
  ScopedOutputDir tmp("child_only");
  const auto images_root = tmp.path / "images";
  writeFile(images_root / "O_1" / "crop_b.png");

  auto dsg = test::makeSharedDsg();
  auto& unmerged = *dsg->graph;
  unmerged.emplaceNode(DsgLayers::OBJECTS, NodeSymbol('O', 0), makeObject(""));
  unmerged.emplaceNode(
      DsgLayers::OBJECTS,
      NodeSymbol('O', 1),
      makeObject((images_root / "temp" / "uuid1").string()));
  unmerged.emplaceNode(DsgLayers::OBJECTS, NodeSymbol('O', 2), makeObject(""));

  GenericUpdateFunctor functor(defaultConfig());
  const auto hooks = functor.hooks();
  ASSERT_TRUE(hooks.merge != nullptr);

  const std::vector<NodeId> nodes{
      NodeSymbol('O', 0), NodeSymbol('O', 1), NodeSymbol('O', 2)};
  auto attrs = hooks.merge(unmerged, nodes);
  const auto* merged = dynamic_cast<const KhronosObjectAttributes*>(attrs.get());
  ASSERT_TRUE(merged != nullptr);

  EXPECT_EQ(merged->image_folder, (images_root / "O_0").string());
  EXPECT_TRUE(std::filesystem::exists(images_root / "O_0" / "crop_b.png"));
  EXPECT_FALSE(std::filesystem::exists(images_root / "O_1"));
  EXPECT_FALSE(std::filesystem::exists(images_root / "O_2"));
}

}  // namespace hydra
