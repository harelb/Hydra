#pragma once
#include <gtsam/geometry/Pose3.h>
#include <pose_graph_tools/pose_graph.h>

#include <Eigen/Dense>
#include <filesystem>
#include <optional>
#include <vector>

#include "hydra_multi/common/types.h"

namespace hydra_multi {

struct UtmCoordinate {
  double easting = 0.0;
  double northing = 0.0;
  int zone = 0;
  bool northern = true;
};

/**
 * @brief Convert WGS84 latitude / longitude to UTM (Krueger series, sub-mm accuracy)
 * @param zone UTM zone to project into (0 picks the zone containing the point)
 */
UtmCoordinate latLonToUtm(double latitude_deg, double longitude_deg, int zone = 0);

//! A single GPS fix in UTM coordinates (easting, northing, altitude)
struct GpsMeasurement {
  Timestamp stamp_ns = 0;
  Eigen::Vector3d utm = Eigen::Vector3d::Zero();
  //! Per-axis variance (east, north, up) in m^2
  Eigen::Vector3d variance = Eigen::Vector3d::Ones();
  //! sensor_msgs/NavSatStatus status (-1 no fix, 0 fix, 1 SBAS, 2 GBAS / RTK)
  int status = 0;
};

using GpsMeasurements = std::vector<GpsMeasurement>;

/**
 * @brief Load GPS fixes from a CSV with header
 * timestamp_ns,latitude,longitude,altitude,status,cov_xx,cov_yy,cov_zz
 * @param zone UTM zone to project into (0 picks the zone of the first fix)
 */
GpsMeasurements loadGpsCsv(const std::filesystem::path& path, int zone = 0);

/**
 * @brief Interpolate the pose of a time-ordered pose graph at a timestamp
 * @param nodes Pose graph nodes sorted by timestamp
 * @param stamp_ns Time to interpolate at
 * @param max_gap_ns Maximum allowed time between the query and the closest node
 * @return Index of the closest node and the interpolated pose, or nullopt if the query
 * is outside of the trajectory or too far from any node
 */
std::optional<std::pair<size_t, gtsam::Pose3>> interpolatePose(
    const std::vector<pose_graph_tools::PoseGraphNode>& nodes,
    Timestamp stamp_ns,
    Timestamp max_gap_ns);

struct GpsAlignment {
  gtsam::Pose3 world_T_robot;
  size_t num_inliers = 0;
  double rmse = 0.0;
};

/**
 * @brief Robustly estimate world_T_robot such that world ≈ world_T_robot * robot
 * (RANSAC on minimal sets followed by a least-squares refit on the inliers)
 * @param robot Points in the robot frame
 * @param world Corresponding points in the world frame
 * @param inlier_threshold_m Residual below which a correspondence is an inlier
 * @param yaw_only Only estimate yaw and translation (gravity-aligned frames)
 */
std::optional<GpsAlignment> alignPoints(const std::vector<Eigen::Vector3d>& robot,
                                        const std::vector<Eigen::Vector3d>& world,
                                        double inlier_threshold_m,
                                        size_t iterations = 200,
                                        bool yaw_only = true);

}  // namespace hydra_multi
