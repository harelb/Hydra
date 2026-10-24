#include <gtest/gtest.h>
#include <hydra_multi/input/file_dsg_input.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace hydra_multi {

using spark_dsg::AgentNodeAttributes;
using spark_dsg::DsgLayers;
using spark_dsg::NodeSymbol;
using spark_dsg::ObjectNodeAttributes;
using spark_dsg::SceneGraph;

TEST(FileDsgInputTests, AnnotatesNodes) {
  const auto tmp = std::filesystem::temp_directory_path() / "hydra_multi_file_dsg_input";
  std::filesystem::create_directories(tmp);

  SceneGraph graph;
  const auto objects = graph.getLayerKey(DsgLayers::OBJECTS)->layer;
  const auto agents = graph.getLayerKey(DsgLayers::AGENTS)->layer;
  graph.emplaceNode(objects, NodeSymbol('O', 1), std::make_unique<ObjectNodeAttributes>());
  graph.emplaceNode(objects, NodeSymbol('O', 2), std::make_unique<ObjectNodeAttributes>());
  graph.emplaceNode(agents,
                    NodeSymbol('a', 0),
                    std::make_unique<AgentNodeAttributes>(),
                    'a');
  graph.save(tmp / "dsg.json", false);

  nlohmann::json annotations;
  annotations[std::to_string(NodeSymbol('O', 2))] = {{"awcd_change", {{"type", "removed"}}}};
  annotations[std::to_string(NodeSymbol('O', 9))] = {{"awcd_change", {{"type", "added"}}}};
  std::ofstream(tmp / "node_metadata.json") << annotations.dump();

  FileDsgInput::Config config;
  config.dsg_json = tmp / "dsg.json";
  config.force_robot_id = true;
  config.session = "2026-04-17";
  config.node_metadata_path = tmp / "node_metadata.json";
  auto state = std::make_shared<UnitInterfaceState>();
  FileDsgInput input(config, state, "test", 1);
  input.init();

  const auto& result = *state->dsg_;
  const auto& o1 = result.getNode(NodeSymbol('O', 1)).attributes().metadata.get();
  EXPECT_EQ(o1.at("session"), "2026-04-17");
  EXPECT_FALSE(o1.contains("awcd_change"));

  const auto& o2 = result.getNode(NodeSymbol('O', 2)).attributes().metadata.get();
  EXPECT_EQ(o2.at("session"), "2026-04-17");
  EXPECT_EQ(o2.at("awcd_change").at("type"), "removed");

  // agents are moved to the robot's prefix and keep their metadata
  ASSERT_TRUE(result.hasNode(NodeSymbol('b', 0)));
  EXPECT_EQ(result.getNode(NodeSymbol('b', 0)).attributes().metadata.get().at("session"),
            "2026-04-17");

  std::filesystem::remove_all(tmp);
}

}  // namespace hydra_multi
