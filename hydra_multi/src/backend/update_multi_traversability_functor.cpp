#include "hydra_multi/backend/update_multi_traversability_functor.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/scene_graph.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <numbers>

namespace hydra_multi {

using spark_dsg::NodeAttributes;
using spark_dsg::NodeId;
using spark_dsg::SceneGraph;
using spark_dsg::TravNodeAttributes;
using spark_dsg::TraversabilityState;

namespace {

static const auto registration =
    config::RegistrationWithConfig<hydra::UpdateFunctor,
                                   UpdateMultiTraversabilityFunctor,
                                   UpdateMultiTraversabilityFunctor::Config>(
        "UpdateMultiTraversabilityFunctor");

struct PlaceSet {
  std::vector<NodeId> ids;
  std::vector<PlaceFootprint> footprints;
  std::vector<size_t> robots;
  std::vector<size_t> num_bins;
};

PlaceSet collectPlaces(const spark_dsg::SceneGraphLayer& layer,
                       const std::unordered_map<NodeId, size_t>* node_to_robot) {
  PlaceSet places;
  for (const auto& node : layer.nodes()) {
    const auto attrs = dynamic_cast<const TravNodeAttributes*>(&node.attributes());
    if (!attrs || attrs->radii.size() < 3) {
      continue;
    }
    places.ids.push_back(node.id);
    places.footprints.emplace_back(*attrs);
    places.num_bins.push_back(attrs->radii.size());
    size_t robot = 0;
    if (node_to_robot) {
      const auto iter = node_to_robot->find(node.id);
      robot = iter == node_to_robot->end() ? 0 : iter->second;
    }
    places.robots.push_back(robot);
  }
  return places;
}

//! Pairs (i < j) of footprints whose intersection covers at least min_overlap of the
//! smaller one
std::vector<std::vector<size_t>> overlapGraph(const std::vector<PlaceFootprint>& fps,
                                              const std::vector<double>& areas,
                                              const std::vector<size_t>& robots,
                                              bool same_robot,
                                              double min_overlap) {
  std::vector<std::vector<size_t>> adjacency(fps.size());
  std::vector<size_t> order(fps.size());
  double max_reach = 0.0;
  for (size_t i = 0; i < fps.size(); ++i) {
    order[i] = i;
    max_reach = std::max(max_reach, fps[i].maxRadius());
  }
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return fps[a].center.x() < fps[b].center.x();
  });
  for (size_t oi = 0; oi < order.size(); ++oi) {
    const size_t i = order[oi];
    const double reach = fps[i].maxRadius() + max_reach;
    for (size_t oj = oi + 1; oj < order.size(); ++oj) {
      const size_t j = order[oj];
      if (fps[j].center.x() - fps[i].center.x() > reach) {
        break;
      }
      if (!same_robot && robots[i] == robots[j]) {
        continue;
      }
      if (!fps[i].mayIntersect(fps[j])) {
        continue;
      }
      const double smaller = std::min(areas[i], areas[j]);
      if (smaller <= 0.0) {
        continue;
      }
      if (intersectionArea(fps[i], fps[j]) >= min_overlap * smaller) {
        adjacency[i].push_back(j);
        adjacency[j].push_back(i);
      }
    }
  }
  return adjacency;
}

//! Star clustering: the largest unassigned item becomes an anchor and takes all its
//! unassigned neighbours. Returns the owner (anchor) of every item.
std::vector<size_t> starCluster(const std::vector<std::vector<size_t>>& adjacency,
                                const std::vector<double>& areas,
                                const std::vector<NodeId>& tie_break) {
  const size_t n = adjacency.size();
  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (areas[a] != areas[b]) {
      return areas[a] > areas[b];
    }
    return tie_break[a] < tie_break[b];
  });
  constexpr size_t unassigned = std::numeric_limits<size_t>::max();
  std::vector<size_t> owner(n, unassigned);
  for (const auto i : order) {
    if (owner[i] != unassigned) {
      continue;
    }
    owner[i] = i;
    for (const auto j : adjacency[i]) {
      if (owner[j] == unassigned) {
        owner[j] = i;
      }
    }
  }
  return owner;
}

}  // namespace

void declare_config(UpdateMultiTraversabilityFunctor::Config& config) {
  using namespace config;
  name("UpdateMultiTraversabilityFunctor::Config");
  base<hydra::UpdateRegionGrowingTraversabilityFunctor::Config>(config);
  field(config.merge_places, "merge_places");
  field(config.merge_same_robot, "merge_same_robot");
  field(config.min_overlap, "min_overlap");
  field(config.trim_overlaps, "trim_overlaps");
  field(config.max_merge_iterations, "max_merge_iterations");
  check(config.min_overlap, GT, 0.0, "min_overlap");
  check(config.min_overlap, LE, 1.0, "min_overlap");
}

UpdateMultiTraversabilityFunctor::UpdateMultiTraversabilityFunctor(const Config& config)
    : UpdateRegionGrowingTraversabilityFunctor(config),
      multi_config(config::checkValid(config)) {}

hydra::UpdateFunctor::Hooks UpdateMultiTraversabilityFunctor::hooks() const {
  auto my_hooks = UpdateRegionGrowingTraversabilityFunctor::hooks();
  if (!multi_config.merge_places) {
    return my_hooks;
  }

  my_hooks.find_merges = [this](const SceneGraph& graph,
                                const hydra::UpdateInfo::ConstPtr& info) {
    return findPlaceMerges(graph, info);
  };
  my_hooks.merge = [this](const SceneGraph& graph, const std::vector<NodeId>& ids) {
    return mergePlaces(graph, ids);
  };
  my_hooks.cleanup = [this](const hydra::UpdateInfo::ConstPtr&,
                            SceneGraph&,
                            hydra::SharedDsgInfo* dsg) {
    // cleanup hooks also run after every later functor: trim once per update
    if (!trim_pending_ || !dsg || !dsg->graph) {
      return;
    }
    trim_pending_ = false;
    trimPlaces(*dsg->graph);
  };
  return my_hooks;
}

void UpdateMultiTraversabilityFunctor::call(const SceneGraph& unmerged,
                                            hydra::SharedDsgInfo& dsg,
                                            const hydra::UpdateInfo::ConstPtr& info) const {
  UpdateRegionGrowingTraversabilityFunctor::call(unmerged, dsg, info);
  merges_found_ = false;
  trim_pending_ = multi_config.merge_places && multi_config.trim_overlaps;
}

hydra::MergeList UpdateMultiTraversabilityFunctor::findPlaceMerges(
    const SceneGraph& graph, const hydra::UpdateInfo::ConstPtr& info) const {
  // the groups are computed once per update from the unmerged graph (the first call);
  // later (exhaustive) passes have nothing to add
  if (merges_found_) {
    return {};
  }
  merges_found_ = true;

  hydra::MergeList merges;
  const auto groups = findPlaceGroups(graph, info ? info->node_to_robot_id : nullptr);
  for (const auto& group : groups) {
    for (size_t i = 1; i < group.size(); ++i) {
      merges.push_back({group[i], group.front()});
    }
  }
  return merges;
}

std::vector<std::vector<NodeId>> UpdateMultiTraversabilityFunctor::findPlaceGroups(
    const SceneGraph& graph,
    const std::unordered_map<NodeId, size_t>* node_to_robot) const {
  std::vector<std::vector<NodeId>> result;
  if (!graph.hasLayer(config.layer)) {
    return result;
  }

  const auto places = collectPlaces(graph.getLayer(config.layer), node_to_robot);
  const size_t n = places.ids.size();
  std::vector<double> areas(n);
  for (size_t i = 0; i < n; ++i) {
    areas[i] = places.footprints[i].area();
  }

  // 1. groups of places that overlap their anchor
  const auto adjacency = overlapGraph(places.footprints,
                                      areas,
                                      places.robots,
                                      multi_config.merge_same_robot,
                                      multi_config.min_overlap);
  const auto owner = starCluster(adjacency, areas, places.ids);
  std::map<size_t, std::vector<size_t>> groups;  // anchor index -> member indices
  for (size_t i = 0; i < n; ++i) {
    groups[owner[i]].push_back(i);
  }

  // 2. merge groups that still overlap after the union and the trimming
  size_t iteration = 0;
  for (; multi_config.trim_overlaps && iteration < multi_config.max_merge_iterations;
       ++iteration) {
    std::vector<size_t> anchors;
    std::vector<PlaceFootprint> unions;
    std::vector<double> union_areas;
    std::vector<size_t> robots;
    std::vector<NodeId> ids;
    for (const auto& [anchor, members] : groups) {
      anchors.push_back(anchor);
      robots.push_back(places.robots[anchor]);
      ids.push_back(places.ids[anchor]);
      if (members.size() == 1) {
        unions.push_back(places.footprints[anchor]);
      } else {
        std::vector<PlaceFootprint> member_fps;
        for (const auto m : members) {
          member_fps.push_back(places.footprints[m]);
        }
        unions.push_back(unionFootprint(
            places.footprints[anchor].center, places.num_bins[anchor], member_fps));
      }
      union_areas.push_back(unions.back().area());
    }

    const auto trimmed = trimFootprints(unions);
    std::vector<double> trimmed_areas(trimmed.size());
    for (size_t i = 0; i < trimmed.size(); ++i) {
      trimmed_areas[i] = trimmed[i].area();
    }
    const auto group_adjacency = overlapGraph(trimmed,
                                              trimmed_areas,
                                              robots,
                                              multi_config.merge_same_robot,
                                              multi_config.min_overlap);
    size_t num_pairs = 0;
    for (const auto& neighbors : group_adjacency) {
      num_pairs += neighbors.size();
    }
    if (!num_pairs) {
      break;
    }

    const auto group_owner = starCluster(group_adjacency, union_areas, ids);
    std::map<size_t, std::vector<size_t>> new_groups;
    for (size_t g = 0; g < anchors.size(); ++g) {
      auto& members = new_groups[anchors[group_owner[g]]];
      const auto& old_members = groups.at(anchors[g]);
      members.insert(members.end(), old_members.begin(), old_members.end());
    }
    groups = std::move(new_groups);
  }

  size_t num_merged = 0;
  for (const auto& [anchor, members] : groups) {
    std::vector<NodeId> group{places.ids[anchor]};
    for (const auto m : members) {
      if (m != anchor) {
        group.push_back(places.ids[m]);
      }
    }
    num_merged += group.size() - 1;
    result.push_back(std::move(group));
  }

  LOG(INFO) << "[UpdateMultiTraversabilityFunctor] " << n << " places -> "
            << groups.size() << " (" << num_merged << " merged, " << iteration
            << " extra rounds)";
  return result;
}

NodeAttributes::Ptr UpdateMultiTraversabilityFunctor::mergePlaces(
    const SceneGraph& unmerged, const std::vector<NodeId>& merge_ids) const {
  if (merge_ids.empty()) {
    return nullptr;
  }
  const auto parent_node = unmerged.findNode(merge_ids.front());
  if (!parent_node) {
    return nullptr;
  }
  auto result = parent_node->attributes().clone();
  auto parent = dynamic_cast<TravNodeAttributes*>(result.get());
  if (!parent || parent->radii.size() < 3) {
    return result;
  }

  std::vector<const TravNodeAttributes*> members;
  std::vector<PlaceFootprint> footprints;
  for (const auto id : merge_ids) {
    const auto node = unmerged.findNode(id);
    if (!node) {
      continue;
    }
    const auto attrs = dynamic_cast<const TravNodeAttributes*>(&node->attributes());
    if (!attrs || attrs->radii.size() < 3) {
      continue;
    }
    members.push_back(attrs);
    footprints.emplace_back(*attrs);
  }
  if (members.size() < 2) {
    return result;
  }

  // footprint: union of the members seen from the surviving place's center
  std::vector<int> source;
  const auto merged = unionFootprint(
      parent->position.head<2>(), parent->radii.size(), footprints, &source);
  setBoundary(*parent, merged);
  if (parent->states.size() == parent->radii.size()) {
    const size_t n = parent->radii.size();
    for (size_t i = 0; i < n; ++i) {
      if (source[i] < 0) {
        continue;
      }
      const auto& member = *members[source[i]];
      if (member.states.size() != member.radii.size()) {
        continue;
      }
      const double angle = 2.0 * std::numbers::pi * i / n;
      const Eigen::Vector2d hit =
          merged.center + merged.radii[i] * Eigen::Vector2d(std::cos(angle), std::sin(angle));
      const Eigen::Vector3d hit_L(hit.x() - member.position.x(),
                                  hit.y() - member.position.y(),
                                  0.0);
      parent->states[i] = member.states[member.getBin(hit_L)];
    }
  }

  // observation stamps and the area-weighted mean feature of the members
  Eigen::VectorXf feature;
  double feature_weight = 0.0;
  for (size_t m = 0; m < members.size(); ++m) {
    const auto& attrs = *members[m];
    parent->first_observed_ns = std::min(parent->first_observed_ns, attrs.first_observed_ns);
    parent->last_observed_ns = std::max(parent->last_observed_ns, attrs.last_observed_ns);
    parent->last_update_time_ns =
        std::max(parent->last_update_time_ns, attrs.last_update_time_ns);
    const auto& f = attrs.semantic_feature;
    if (f.size() == 0) {
      continue;
    }
    if (feature.size() == 0) {
      feature = Eigen::VectorXf::Zero(f.size());
    }
    if (f.size() != feature.size()) {
      continue;
    }
    const double weight = std::max(footprints[m].area(), 1.0e-6);
    feature += static_cast<float>(weight) * f;
    feature_weight += weight;
  }
  if (feature_weight > 0.0) {
    const float norm = feature.norm();
    if (norm > 0.0f) {
      parent->semantic_feature = feature / norm;
    }
  }
  return result;
}

void UpdateMultiTraversabilityFunctor::trimPlaces(SceneGraph& graph) const {
  if (!graph.hasLayer(config.layer)) {
    return;
  }
  const auto places = collectPlaces(graph.getLayer(config.layer), nullptr);
  std::vector<std::vector<bool>> trimmed_rays;
  const auto trimmed = trimFootprints(places.footprints, &trimmed_rays);
  size_t num_trimmed = 0;
  for (size_t i = 0; i < places.ids.size(); ++i) {
    const auto& rays = trimmed_rays[i];
    if (std::none_of(rays.begin(), rays.end(), [](bool t) { return t; })) {
      continue;
    }
    ++num_trimmed;
    auto& attrs = graph.getNode(places.ids[i]).attributes<TravNodeAttributes>();
    setBoundary(attrs, trimmed[i]);
    if (attrs.states.size() == rays.size()) {
      for (size_t r = 0; r < rays.size(); ++r) {
        if (rays[r]) {
          // the trimmed boundary borders the neighbouring place
          attrs.states[r] = TraversabilityState::TRAVERSABLE;
        }
      }
    }
  }
  LOG(INFO) << "[UpdateMultiTraversabilityFunctor] trimmed " << num_trimmed << " / "
            << places.ids.size() << " places";
}

}  // namespace hydra_multi
