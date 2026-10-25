#include <gtest/gtest.h>
#include <hydra/backend/dsg_updater.h>
#include <hydra/common/shared_dsg_info.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph.h>

#include <numbers>

#include "hydra_multi/backend/place_footprint.h"
#include "hydra_multi/backend/update_multi_traversability_functor.h"

namespace hydra_multi {

using spark_dsg::DsgLayers;
using spark_dsg::NodeId;
using spark_dsg::NodeSymbol;
using spark_dsg::SceneGraph;
using spark_dsg::TravNodeAttributes;
using spark_dsg::TraversabilityState;

namespace {

constexpr size_t kBins = 16;

std::unique_ptr<TravNodeAttributes> makePlace(double x,
                                              double y,
                                              double radius,
                                              uint64_t first_ns,
                                              float feature = 0.0f) {
  auto attrs = std::make_unique<TravNodeAttributes>();
  attrs->position = Eigen::Vector3d(x, y, 0.0);
  attrs->radii.assign(kBins, radius);
  attrs->states.assign(kBins, TraversabilityState::INTRAVERSABLE);
  attrs->min_radius = radius;
  attrs->max_radius = radius;
  attrs->first_observed_ns = first_ns;
  attrs->last_observed_ns = first_ns + 10;
  attrs->is_active = false;
  if (feature != 0.0f) {
    attrs->semantic_feature = Eigen::VectorXf::Zero(2);
    attrs->semantic_feature(0) = feature;
    attrs->semantic_feature(1) = 1.0f - feature;
    attrs->semantic_feature.normalize();
  }
  return attrs;
}

hydra::SharedDsgInfo::Ptr makeDsg() {
  return std::make_shared<hydra::SharedDsgInfo>(hydra::SharedDsgInfo::Config());
}

UpdateMultiTraversabilityFunctor::Config makeConfig() {
  UpdateMultiTraversabilityFunctor::Config config;
  config.layer = DsgLayers::MESH_PLACES;
  return config;
}

}  // namespace

TEST(PlaceFootprint, RotateBoundary) {
  auto attrs = makePlace(0.0, 0.0, 1.0, 0);
  attrs->radii[0] = 3.0;  // long ray along +x
  rotateBoundary(*attrs, std::numbers::pi / 2.0);
  // the long ray now points along +y (bin N / 4)
  EXPECT_NEAR(attrs->radii[kBins / 4], 3.0, 1.0e-9);
  EXPECT_NEAR(attrs->radii[0], 1.0, 1.0e-9);
  EXPECT_TRUE(attrs->contains(Eigen::Vector3d(0.0, 2.5, 0.0)));
  EXPECT_FALSE(attrs->contains(Eigen::Vector3d(2.5, 0.0, 0.0)));
  EXPECT_NEAR(attrs->max_radius, 3.0, 1.0e-9);
}

TEST(PlaceFootprint, UnionAndTrim) {
  const PlaceFootprint a(Eigen::Vector2d(0.0, 0.0), std::vector<double>(kBins, 1.0));
  const PlaceFootprint b(Eigen::Vector2d(1.0, 0.0), std::vector<double>(kBins, 1.0));
  const double overlap = intersectionArea(a, b);
  EXPECT_GT(overlap, 0.5);
  EXPECT_LT(overlap, a.area());

  const auto merged = unionFootprint(a.center, kBins, {a, b});
  EXPECT_NEAR(merged.radii[0], 2.0, 1.0e-9);  // reaches the far side of b
  EXPECT_NEAR(merged.radii[kBins / 2], 1.0, 1.0e-9);
  EXPECT_GE(merged.area(), a.area() + b.area() - overlap - 0.2);

  const auto trimmed = trimFootprints({a, b});
  EXPECT_NEAR(trimmed[0].radii[0], 0.5, 1.0e-9);  // cut at the bisector x = 0.5
  EXPECT_NEAR(trimmed[1].radii[kBins / 2], 0.5, 1.0e-9);
  EXPECT_LT(intersectionArea(trimmed[0], trimmed[1]), 1.0e-6);
}

TEST(UpdateMultiTraversabilityFunctor, MergesOverlappingPlacesAcrossRobots) {
  // robot 0: two neighbouring places; robot 1: a place on the same ground as the first
  // and one that only touches the second
  auto source = makeDsg()->graph;
  const NodeId a0 = NodeSymbol('t', 0), a1 = NodeSymbol('t', 1);
  const NodeId b0 = NodeSymbol('t', 2), b1 = NodeSymbol('t', 3);
  source->emplaceNode(DsgLayers::MESH_PLACES, a0, makePlace(0.0, 0.0, 2.0, 100, 0.9f));
  source->emplaceNode(DsgLayers::MESH_PLACES, a1, makePlace(3.8, 0.0, 2.0, 200, 0.5f));
  source->emplaceNode(DsgLayers::MESH_PLACES, b0, makePlace(0.3, -0.6, 1.5, 50, 0.1f));
  source->emplaceNode(DsgLayers::MESH_PLACES, b1, makePlace(7.4, 0.0, 2.0, 300));
  source->insertEdge(a0, a1);
  source->insertEdge(b0, b1);
  const NodeId object = NodeSymbol('O', 0);
  source->emplaceNode(DsgLayers::OBJECTS, object, std::make_unique<spark_dsg::ObjectNodeAttributes>());
  source->insertEdge(b0, object);  // place -> object
  const std::unordered_map<NodeId, size_t> robots{{a0, 0}, {a1, 0}, {b0, 1}, {b1, 1}};

  hydra::DsgUpdater::Config config;
  config.update_functors.emplace_back(
      "surface_places", hydra::DsgUpdater::Config::FunctorConfig(makeConfig()));
  auto target = makeDsg();
  hydra::DsgUpdater updater(config, source, target);
  hydra::UpdateInfo::ConstPtr info(new hydra::UpdateInfo{
      10, nullptr, nullptr, true, {}, nullptr, &robots});
  updater.callUpdateFunctions(10, info);

  const auto& graph = *target->graph;
  // b0 lies on a0's ground and is merged into the larger a0 (stable id)
  EXPECT_FALSE(graph.hasNode(b0));
  ASSERT_TRUE(graph.hasNode(a0));
  EXPECT_TRUE(graph.hasNode(a1));
  EXPECT_TRUE(graph.hasNode(b1));
  const std::map<NodeId, NodeId> expected_merges{{b0, a0}};
  EXPECT_EQ(target->merges, expected_merges);

  // edges are rewired to the surviving place
  EXPECT_TRUE(graph.hasEdge(a0, a1));
  EXPECT_TRUE(graph.hasEdge(a0, b1));
  EXPECT_TRUE(graph.hasEdge(a0, object));

  // merged attributes: union footprint (trimmed by the neighbours), earliest stamp,
  // area-weighted unit feature
  const auto& merged = graph.getNode(a0).attributes<TravNodeAttributes>();
  EXPECT_EQ(merged.first_observed_ns, 50u);
  EXPECT_EQ(merged.last_observed_ns, 110u);
  EXPECT_NEAR(merged.semantic_feature.norm(), 1.0f, 1.0e-5);
  const Eigen::Vector3d only_b0(0.3, -2.0, 0.0);
  ASSERT_FALSE(makePlace(0.0, 0.0, 2.0, 0)->contains(only_b0));
  EXPECT_TRUE(merged.contains(only_b0));

  // no remaining overlaps between the surviving places
  std::vector<PlaceFootprint> footprints;
  for (const auto& node : graph.getLayer(DsgLayers::MESH_PLACES).nodes()) {
    footprints.emplace_back(node.attributes<TravNodeAttributes>());
  }
  for (size_t i = 0; i < footprints.size(); ++i) {
    for (size_t j = i + 1; j < footprints.size(); ++j) {
      EXPECT_LT(intersectionArea(footprints[i], footprints[j]), 0.05) << i << ", " << j;
    }
  }

  // the trimmed boundary between a0 and a1 is marked traversable
  const auto& neighbor = graph.getNode(a1).attributes<TravNodeAttributes>();
  EXPECT_EQ(neighbor.states[kBins / 2], TraversabilityState::TRAVERSABLE);
}

TEST(UpdateMultiTraversabilityFunctor, SameRobotOptional) {
  auto source = makeDsg()->graph;
  const NodeId a0 = NodeSymbol('t', 0), a1 = NodeSymbol('t', 1);
  source->emplaceNode(DsgLayers::MESH_PLACES, a0, makePlace(0.0, 0.0, 2.0, 100));
  source->emplaceNode(DsgLayers::MESH_PLACES, a1, makePlace(0.5, 0.0, 1.0, 900));
  const std::unordered_map<NodeId, size_t> robots{{a0, 0}, {a1, 0}};

  auto config = makeConfig();
  config.merge_same_robot = false;
  const UpdateMultiTraversabilityFunctor same_off(config);
  EXPECT_EQ(same_off.findPlaceGroups(*source, &robots).size(), 2u);

  config.merge_same_robot = true;
  const UpdateMultiTraversabilityFunctor same_on(config);
  const auto groups = same_on.findPlaceGroups(*source, &robots);
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups.front(), std::vector<NodeId>({a0, a1}));
}

}  // namespace hydra_multi
