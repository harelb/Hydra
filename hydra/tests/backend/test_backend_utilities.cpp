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
#include <gtest/gtest.h>
#include <hydra/backend/backend_utilities.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "hydra_test/shared_dsg_fixture.h"

namespace hydra {

using spark_dsg::AgentNodeAttributes;
using spark_dsg::KhronosObjectAttributes;
using spark_dsg::NodeSymbol;

namespace {

// A throwaway directory that cleans itself up, named per-test to avoid collisions.
struct ScopedTempDir {
  std::filesystem::path path;
  explicit ScopedTempDir(const std::string& name)
      : path(std::filesystem::temp_directory_path() / ("hydra_reconcile_" + name)) {
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~ScopedTempDir() { std::filesystem::remove_all(path); }
};

void writeFile(const std::filesystem::path& p) { std::ofstream(p) << "{}\n"; }

std::unique_ptr<AgentNodeAttributes> makeAgent(int64_t ts_ns,
                                               char prefix,
                                               size_t index,
                                               const std::string& image_folder = "") {
  auto attrs = std::make_unique<AgentNodeAttributes>(
      std::chrono::nanoseconds(ts_ns),
      Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0),
      Eigen::Vector3d(1.0, 2.0, 3.0),
      NodeSymbol(prefix, index));
  attrs->image_folder = image_folder;
  return attrs;
}

}  // namespace

// An empty agent image_folder is reconstructed from the node timestamp (the exact value
// the extractor used for the on-disk filename), only when that *_meta.json exists. The
// agents directory is derived from an already-populated sibling agent node.
TEST(BackendUtilities, ReconcileAgentsFillsEmptyFromTimestamp) {
  ScopedTempDir tmp("agents_fill");
  writeFile(tmp.path / "agent_1000_meta.json");  // a0 (already populated)
  writeFile(tmp.path / "agent_2000_meta.json");  // a1 (empty -> should fill)
  // No file for ts 3000 -> a2 must stay empty.

  auto dsg = test::makeSharedDsg();
  auto& graph = *dsg->graph;
  graph.emplaceNode(
      2, NodeSymbol('a', 0), makeAgent(1000, 'a', 0, (tmp.path / "agent_1000").string()), 'a');
  graph.emplaceNode(2, NodeSymbol('a', 1), makeAgent(2000, 'a', 1), 'a');
  graph.emplaceNode(2, NodeSymbol('a', 2), makeAgent(3000, 'a', 2), 'a');

  const auto filled = utils::reconcileAgentImageFolders(graph);
  EXPECT_EQ(filled, 1u);

  EXPECT_EQ(graph.getNode(NodeSymbol('a', 0)).attributes<AgentNodeAttributes>().image_folder,
            (tmp.path / "agent_1000").string());
  EXPECT_EQ(graph.getNode(NodeSymbol('a', 1)).attributes<AgentNodeAttributes>().image_folder,
            (tmp.path / "agent_2000").string());
  EXPECT_TRUE(
      graph.getNode(NodeSymbol('a', 2)).attributes<AgentNodeAttributes>().image_folder.empty());
}

// Already-populated agent folders are never overwritten.
TEST(BackendUtilities, ReconcileAgentsPreservesExisting) {
  ScopedTempDir tmp("agents_preserve");
  writeFile(tmp.path / "agent_1000_meta.json");
  const std::string custom = (tmp.path / "custom_folder").string();

  auto dsg = test::makeSharedDsg();
  auto& graph = *dsg->graph;
  graph.emplaceNode(2, NodeSymbol('a', 0), makeAgent(1000, 'a', 0, custom), 'a');

  EXPECT_EQ(utils::reconcileAgentImageFolders(graph), 0u);
  EXPECT_EQ(graph.getNode(NodeSymbol('a', 0)).attributes<AgentNodeAttributes>().image_folder,
            custom);
}

// No populated sibling and no $ADT4_OUTPUT_DIR -> nothing to derive from -> no-op.
TEST(BackendUtilities, ReconcileAgentsNoDirIsNoOp) {
  const char* prev = std::getenv("ADT4_OUTPUT_DIR");
  unsetenv("ADT4_OUTPUT_DIR");

  auto dsg = test::makeSharedDsg();
  auto& graph = *dsg->graph;
  graph.emplaceNode(2, NodeSymbol('a', 0), makeAgent(1000, 'a', 0), 'a');  // empty, no sibling

  EXPECT_EQ(utils::reconcileAgentImageFolders(graph), 0u);
  EXPECT_TRUE(
      graph.getNode(NodeSymbol('a', 0)).attributes<AgentNodeAttributes>().image_folder.empty());

  if (prev) {
    setenv("ADT4_OUTPUT_DIR", prev, 1);
  }
}

// Object counterpart: empty image_folder is filled with <images_dir>/<C>_<id> when that
// directory exists, derived from a populated sibling object.
TEST(BackendUtilities, ReconcileObjectsFillsEmptyFromSymbol) {
  ScopedTempDir tmp("objects_fill");
  std::filesystem::create_directories(tmp.path / "O_0");  // O0 (already populated)
  std::filesystem::create_directories(tmp.path / "O_5");  // O5 (empty -> should fill)
  // No directory for O7 -> stays empty.

  auto dsg = test::makeSharedDsg();
  auto& graph = *dsg->graph;
  {
    auto a = std::make_unique<KhronosObjectAttributes>();
    a->image_folder = (tmp.path / "O_0").string();
    graph.emplaceNode(2, NodeSymbol('O', 0), std::move(a));
  }
  graph.emplaceNode(2, NodeSymbol('O', 5), std::make_unique<KhronosObjectAttributes>());
  graph.emplaceNode(2, NodeSymbol('O', 7), std::make_unique<KhronosObjectAttributes>());

  EXPECT_EQ(utils::reconcileObjectImageFolders(graph), 1u);
  EXPECT_EQ(graph.getNode(NodeSymbol('O', 5)).attributes<KhronosObjectAttributes>().image_folder,
            (tmp.path / "O_5").string());
  EXPECT_TRUE(
      graph.getNode(NodeSymbol('O', 7)).attributes<KhronosObjectAttributes>().image_folder.empty());
}

}  // namespace hydra
