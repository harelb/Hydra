#pragma once

#include <gtsam/inference/Symbol.h>
#include <kimera_pgmo/deformation_graph.h>
#include <kimera_pgmo/utils/common_functions.h>

#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>

#include "hydra_test/temp_directory.h"

namespace hydra::test {

/**
 * @brief Deformation graph whose control points all move by the same rigid
 * transform, so that every interpolated node transform is exactly that transform.
 *
 * @param world_T_world Transform applied to every control point.
 * @param robot_id Robot whose mesh vertex prefix the control points use.
 * @param stamp_ns Timestamp of every control point.
 */
inline kimera_pgmo::DeformationGraph::Ptr makeRigidDeformationGraph(
    const Eigen::Isometry3d& world_T_world,
    size_t robot_id = 0,
    uint64_t stamp_ns = 10) {
  const auto prefix = kimera_pgmo::GetVertexPrefix(robot_id);
  const Eigen::Quaterniond q(world_T_world.linear());
  // start from the serialized form of an empty graph so every field kimera_pgmo expects is
  // present, then add the control points
  const TempDirectory tmp("rigid_deformation");
  const auto path = tmp.path / "graph.json";
  kimera_pgmo::DeformationGraph().save(path.string());
  nlohmann::json root;
  {
    std::ifstream in(path);
    root = nlohmann::json::parse(in);
  }

  const std::string prefix_str(1, prefix);
  auto& vertices = root["vertices"][prefix_str];
  vertices["pos"] = nlohmann::json::array();
  vertices["stamps"] = nlohmann::json::array();
  root["values"] = nlohmann::json::array();
  // control points on a small grid so any query has enough neighbors
  size_t index = 0;
  for (int x = 0; x < 3; ++x) {
    for (int y = 0; y < 3; ++y) {
      const Eigen::Vector3d point(x, y, 0.0);
      // the interpolated transform is R * (p - c) + t with t = R * c + translation
      const Eigen::Vector3d t = world_T_world * point;
      root["values"].push_back(
          {{"key", gtsam::Symbol(prefix, index).key()},
           {"value",
            {{"x", t.x()},
             {"y", t.y()},
             {"z", t.z()},
             {"qw", q.w()},
             {"qx", q.x()},
             {"qy", q.y()},
             {"qz", q.z()}}}});
      vertices["pos"].push_back({point.x(), point.y(), point.z()});
      vertices["stamps"].push_back(stamp_ns);
      ++index;
    }
  }

  {
    std::ofstream out(path);
    out << root;
  }

  return kimera_pgmo::DeformationGraph::load(path);
}

}  // namespace hydra::test
