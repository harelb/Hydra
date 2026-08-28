#include <gtest/gtest.h>
#include <hydra_ros/utils/change_gate.h>
#include <spark_dsg/scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

namespace hydra {

namespace {

constexpr uint64_t kSecond = 1'000'000'000ull;

spark_dsg::SceneGraph::Ptr makeGraph(size_t num_nodes, double x_offset = 0.0) {
  auto graph = std::make_shared<spark_dsg::SceneGraph>();
  for (size_t i = 0; i < num_nodes; ++i) {
    auto attrs = std::make_unique<spark_dsg::NodeAttributes>();
    attrs->position << x_offset + static_cast<double>(i), 0.0, 0.0;
    graph->emplaceNode(spark_dsg::DsgLayers::OBJECTS,
                       spark_dsg::NodeSymbol('O', i),
                       std::move(attrs));
  }
  return graph;
}

ChangeGate::Config testConfig() {
  ChangeGate::Config config;
  config.displacement_threshold_m = 0.10;
  config.node_delta_threshold = 25;
  config.max_interval_s = 2.0;
  config.min_separation_s = 0.5;
  return config;
}

}  // namespace

TEST(ChangeGate, FirstCallPublishes) {
  ChangeGate gate(testConfig());
  auto graph = makeGraph(3);
  EXPECT_TRUE(gate.shouldPublish(*graph, 0));
}

TEST(ChangeGate, MinSeparationBlocksEverything) {
  ChangeGate gate(testConfig());
  auto graph = makeGraph(3);
  gate.notePublished(*graph, 0);
  // huge change, but only 0.1s elapsed
  auto moved = makeGraph(3, /*x_offset=*/10.0);
  EXPECT_FALSE(gate.shouldPublish(*moved, kSecond / 10));
}

TEST(ChangeGate, UnchangedGraphWaitsForHeartbeat) {
  ChangeGate gate(testConfig());
  auto graph = makeGraph(3);
  gate.notePublished(*graph, 0);
  EXPECT_FALSE(gate.shouldPublish(*graph, 1 * kSecond));  // no change, cap elapsed
  EXPECT_TRUE(gate.shouldPublish(*graph, 2 * kSecond));   // heartbeat
}

TEST(ChangeGate, DisplacementTriggers) {
  ChangeGate gate(testConfig());
  auto graph = makeGraph(3);
  gate.notePublished(*graph, 0);
  auto moved = makeGraph(3, /*x_offset=*/0.5);  // 0.5m >> 0.10m threshold
  EXPECT_TRUE(gate.shouldPublish(*moved, 1 * kSecond));
}

TEST(ChangeGate, NodeDeltaTriggersAndAccumulates) {
  ChangeGate gate(testConfig());
  auto graph = makeGraph(3);
  gate.notePublished(*graph, 0);
  // 10 new nodes: below the 25 threshold
  EXPECT_FALSE(gate.shouldPublish(*makeGraph(13), 1 * kSecond));
  // accumulated 30 new nodes since last publish: triggers (compared vs last PUBLISH)
  EXPECT_TRUE(gate.shouldPublish(*makeGraph(33), 1 * kSecond + kSecond / 2));
}

}  // namespace hydra
