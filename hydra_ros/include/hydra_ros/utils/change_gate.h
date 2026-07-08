#pragma once
#include <spark_dsg/dynamic_scene_graph.h>

#include <Eigen/Geometry>
#include <optional>
#include <unordered_map>

namespace hydra {

// Decides whether the backend DSG changed enough since the LAST ACTUAL PUBLISH to be
// worth re-serializing. Change accumulates across skipped calls by construction (state
// only advances in notePublished).
struct ChangeGate {
  struct Config {
    //! @brief max node displacement since last publish that forces a publish [m]
    double displacement_threshold_m = 0.10;
    //! @brief total node-count change since last publish that forces a publish
    size_t node_delta_threshold = 25;
    //! @brief heartbeat: always publish if this much data-time elapsed [s]
    double max_interval_s = 2.0;
    //! @brief hard rate cap: never publish more often than this [s]
    double min_separation_s = 0.5;
    //! @brief bound on tracked node positions (strided sample per layer)
    size_t max_position_samples = 512;
  } const config;

  explicit ChangeGate(const Config& config);

  bool shouldPublish(const spark_dsg::DynamicSceneGraph& graph, uint64_t timestamp_ns);
  void notePublished(const spark_dsg::DynamicSceneGraph& graph, uint64_t timestamp_ns);

 private:
  double maxDisplacement(const spark_dsg::DynamicSceneGraph& graph) const;

  std::unordered_map<spark_dsg::NodeId, Eigen::Vector3d> last_positions_;
  size_t last_num_nodes_ = 0;
  std::optional<uint64_t> last_publish_ns_;
};

void declare_config(ChangeGate::Config& config);

}  // namespace hydra
