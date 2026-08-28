#include "hydra_ros/utils/change_gate.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <spark_dsg/node_attributes.h>

#include <algorithm>

namespace hydra {

void declare_config(ChangeGate::Config& config) {
  using namespace config;
  name("ChangeGate::Config");
  field(config.displacement_threshold_m, "displacement_threshold_m");
  field(config.node_delta_threshold, "node_delta_threshold");
  field(config.max_interval_s, "max_interval_s");
  field(config.min_separation_s, "min_separation_s");
  field(config.max_position_samples, "max_position_samples");
  check(config.displacement_threshold_m, GT, 0.0, "displacement_threshold_m");
  check(config.max_interval_s, GE, config.min_separation_s, "max_interval_s");
}

ChangeGate::ChangeGate(const Config& config) : config(config::checkValid(config)) {}

bool ChangeGate::shouldPublish(const spark_dsg::SceneGraph& graph,
                               uint64_t timestamp_ns) {
  if (!last_publish_ns_) {
    return true;
  }

  const auto dt_s = (timestamp_ns - *last_publish_ns_) * 1.0e-9;
  if (dt_s < config.min_separation_s) {
    return false;
  }

  if (dt_s >= config.max_interval_s) {
    return true;
  }

  const auto num_nodes = graph.numNodes();
  const auto node_delta = num_nodes >= last_num_nodes_ ? num_nodes - last_num_nodes_
                                                       : last_num_nodes_ - num_nodes;
  if (node_delta >= config.node_delta_threshold) {
    return true;
  }

  return maxDisplacement(graph) >= config.displacement_threshold_m;
}

void ChangeGate::notePublished(const spark_dsg::SceneGraph& graph,
                               uint64_t timestamp_ns) {
  last_publish_ns_ = timestamp_ns;
  last_num_nodes_ = graph.numNodes();

  last_positions_.clear();
  const auto& layers = graph.layers();
  if (layers.empty()) {
    return;
  }

  const size_t per_layer =
      std::max<size_t>(1, config.max_position_samples / layers.size());
  for (const auto& [layer_id, layer] : layers) {
    const size_t stride = std::max<size_t>(1, layer->numNodes() / per_layer);
    size_t i = 0;
    for (const auto& [node_id, node] : layer->nodes()) {
      if ((i++ % stride) != 0) {
        continue;
      }
      last_positions_[node_id] = node->attributes().position;
    }
  }
}

double ChangeGate::maxDisplacement(const spark_dsg::SceneGraph& graph) const {
  double max_disp = 0.0;
  for (const auto& [node_id, last_pos] : last_positions_) {
    const auto node = graph.findNode(node_id);
    if (!node) {
      continue;
    }
    max_disp = std::max(max_disp, (node->attributes().position - last_pos).norm());
  }
  return max_disp;
}

}  // namespace hydra
