#pragma once

#include <gtsam/inference/Symbol.h>
#include <kimera_pgmo/deformation_graph.h>
#include <kimera_pgmo/utils/common_functions.h>

#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <memory>

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
inline std::unique_ptr<kimera_pgmo::DeformationGraph> makeRigidDeformationGraph(
    const Eigen::Isometry3d& world_T_world,
    size_t robot_id = 0,
    uint64_t stamp_ns = 10) {
  const auto prefix = kimera_pgmo::GetVertexPrefix(robot_id);
  const Eigen::Quaterniond q(world_T_world.linear());
  const TempDirectory tmp("rigid_deformation");
  const auto path = tmp.path / "graph.dgrf";
  {
    std::ofstream out(path);
    out.precision(17);
    // control points on a small grid so any query has enough neighbors
    size_t index = 0;
    for (int x = 0; x < 3; ++x) {
      for (int y = 0; y < 3; ++y) {
        const Eigen::Vector3d point(x, y, 0.0);
        // the interpolated transform is R * (p - c) + t with t = R * c + translation
        const Eigen::Vector3d t = world_T_world * point;
        out << "NODE " << gtsam::Symbol(prefix, index).key() << " " << t.x() << " "
            << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z()
            << " " << q.w() << "\n";
        ++index;
      }
    }

    index = 0;
    for (int x = 0; x < 3; ++x) {
      for (int y = 0; y < 3; ++y) {
        out << "VERTEX " << gtsam::Symbol(prefix, index).key() << " " << stamp_ns << " "
            << x << " " << y << " 0\n";
        ++index;
      }
    }
  }

  auto dgraph = std::make_unique<kimera_pgmo::DeformationGraph>();
  dgraph->load(path.string());
  return dgraph;
}

}  // namespace hydra::test
