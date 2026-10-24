#include "hydra_multi/common/gps.h"

#include <glog/logging.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>

namespace hydra_multi {

namespace {

inline double deg2rad(double deg) { return deg * M_PI / 180.0; }

// Least-squares fit of world ≈ R * robot + t. Yaw-only fits rotation about z in the xy
// plane and use the median altitude offset (GPS altitude is much noisier than position)
gtsam::Pose3 fitTransform(const std::vector<Eigen::Vector3d>& robot,
                          const std::vector<Eigen::Vector3d>& world,
                          const std::vector<size_t>& indices,
                          bool yaw_only) {
  if (!yaw_only) {
    Eigen::Matrix3Xd src(3, indices.size());
    Eigen::Matrix3Xd dst(3, indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      src.col(i) = robot[indices[i]];
      dst.col(i) = world[indices[i]];
    }

    const Eigen::Matrix4d T = Eigen::umeyama(src, dst, false);
    return gtsam::Pose3(T);
  }

  Eigen::Vector2d robot_mean = Eigen::Vector2d::Zero();
  Eigen::Vector2d world_mean = Eigen::Vector2d::Zero();
  for (const auto idx : indices) {
    robot_mean += robot[idx].head<2>();
    world_mean += world[idx].head<2>();
  }
  robot_mean /= indices.size();
  world_mean /= indices.size();

  double sin_sum = 0.0;
  double cos_sum = 0.0;
  std::vector<double> dz;
  for (const auto idx : indices) {
    const Eigen::Vector2d r = robot[idx].head<2>() - robot_mean;
    const Eigen::Vector2d w = world[idx].head<2>() - world_mean;
    sin_sum += r.x() * w.y() - r.y() * w.x();
    cos_sum += r.x() * w.x() + r.y() * w.y();
    dz.push_back(world[idx].z() - robot[idx].z());
  }

  std::nth_element(dz.begin(), dz.begin() + dz.size() / 2, dz.end());
  const double yaw = std::atan2(sin_sum, cos_sum);
  const Eigen::Rotation2Dd R(yaw);
  const Eigen::Vector2d t = world_mean - R * robot_mean;
  return gtsam::Pose3(gtsam::Rot3::Yaw(yaw), gtsam::Point3(t.x(), t.y(), dz[dz.size() / 2]));
}

double residual(const gtsam::Pose3& world_T_robot,
                const Eigen::Vector3d& robot,
                const Eigen::Vector3d& world,
                bool yaw_only) {
  const Eigen::Vector3d diff = world_T_robot.transformFrom(robot) - world;
  return yaw_only ? diff.head<2>().norm() : diff.norm();
}

}  // namespace

UtmCoordinate latLonToUtm(double latitude_deg, double longitude_deg, int zone) {
  // WGS84
  constexpr double a = 6378137.0;
  constexpr double f = 1.0 / 298.257223563;
  constexpr double k0 = 0.9996;

  UtmCoordinate result;
  result.zone = zone > 0 ? zone : static_cast<int>(std::floor((longitude_deg + 180.0) / 6.0)) + 1;
  result.northern = latitude_deg >= 0.0;

  const double n = f / (2.0 - f);
  const double n2 = n * n;
  const double n3 = n2 * n;
  const double A = a / (1.0 + n) * (1.0 + n2 / 4.0 + n2 * n2 / 64.0);
  const double alpha[3] = {n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0,
                           13.0 * n2 / 48.0 - 3.0 * n3 / 5.0,
                           61.0 * n3 / 240.0};

  const double lon0 = deg2rad((result.zone - 1) * 6.0 - 180.0 + 3.0);
  const double phi = deg2rad(latitude_deg);
  const double dlambda = deg2rad(longitude_deg) - lon0;
  const double c = 2.0 * std::sqrt(n) / (1.0 + n);
  const double t = std::sinh(std::atanh(std::sin(phi)) - c * std::atanh(c * std::sin(phi)));
  const double xi_p = std::atan2(t, std::cos(dlambda));
  const double eta_p = std::atanh(std::sin(dlambda) / std::sqrt(1.0 + t * t));

  double xi = xi_p;
  double eta = eta_p;
  for (int j = 1; j <= 3; ++j) {
    xi += alpha[j - 1] * std::sin(2.0 * j * xi_p) * std::cosh(2.0 * j * eta_p);
    eta += alpha[j - 1] * std::cos(2.0 * j * xi_p) * std::sinh(2.0 * j * eta_p);
  }

  result.easting = 500000.0 + k0 * A * eta;
  result.northing = (result.northern ? 0.0 : 10000000.0) + k0 * A * xi;
  return result;
}

GpsMeasurements loadGpsCsv(const std::filesystem::path& path, int zone) {
  std::ifstream file(path);
  if (!file.good()) {
    LOG(ERROR) << "Unable to open GPS file " << path;
    return {};
  }

  GpsMeasurements measurements;
  std::string line;
  size_t line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#' || std::isalpha(line[0])) {
      continue;  // header or comment
    }

    std::stringstream ss(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(ss, field, ',')) {
      fields.push_back(field);
    }

    if (fields.size() < 8) {
      LOG(WARNING) << "Skipping malformed GPS line " << line_number << " in " << path;
      continue;
    }

    const double lat = std::stod(fields[1]);
    const double lon = std::stod(fields[2]);
    const auto utm = latLonToUtm(lat, lon, zone);
    zone = utm.zone;  // keep every fix in the same zone

    auto& meas = measurements.emplace_back();
    meas.stamp_ns = std::stoull(fields[0]);
    meas.utm << utm.easting, utm.northing, std::stod(fields[3]);
    meas.status = std::stoi(fields[4]);
    meas.variance << std::stod(fields[5]), std::stod(fields[6]), std::stod(fields[7]);
  }

  std::sort(measurements.begin(), measurements.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.stamp_ns < rhs.stamp_ns;
  });
  return measurements;
}

std::optional<std::pair<size_t, gtsam::Pose3>> interpolatePose(
    const std::vector<pose_graph_tools::PoseGraphNode>& nodes,
    Timestamp stamp_ns,
    Timestamp max_gap_ns) {
  if (nodes.empty() || stamp_ns < nodes.front().stamp_ns ||
      stamp_ns > nodes.back().stamp_ns) {
    return std::nullopt;
  }

  auto upper = std::lower_bound(
      nodes.begin(), nodes.end(), stamp_ns, [](const auto& node, Timestamp stamp) {
        return node.stamp_ns < stamp;
      });
  const size_t hi = std::distance(nodes.begin(), upper);
  if (nodes[hi].stamp_ns == stamp_ns) {
    return std::make_pair(hi, gtsam::Pose3(nodes[hi].pose.matrix()));
  }

  const size_t lo = hi - 1;
  const auto t_lo = nodes[lo].stamp_ns;
  const auto t_hi = nodes[hi].stamp_ns;
  const size_t closest = (stamp_ns - t_lo) <= (t_hi - stamp_ns) ? lo : hi;
  const auto gap = closest == lo ? stamp_ns - t_lo : t_hi - stamp_ns;
  if (gap > max_gap_ns) {
    return std::nullopt;
  }

  const double ratio = static_cast<double>(stamp_ns - t_lo) / (t_hi - t_lo);
  const gtsam::Pose3 pose_lo(nodes[lo].pose.matrix());
  const gtsam::Pose3 pose_hi(nodes[hi].pose.matrix());
  return std::make_pair(closest, pose_lo.interpolateRt(pose_hi, ratio));
}

std::optional<GpsAlignment> alignPoints(const std::vector<Eigen::Vector3d>& robot,
                                        const std::vector<Eigen::Vector3d>& world,
                                        double inlier_threshold_m,
                                        size_t iterations,
                                        bool yaw_only) {
  CHECK_EQ(robot.size(), world.size());
  const size_t sample_size = yaw_only ? 2 : 3;
  if (robot.size() < sample_size + 1) {
    return std::nullopt;
  }

  std::mt19937 rng(0);
  std::uniform_int_distribution<size_t> dist(0, robot.size() - 1);
  std::vector<size_t> best_inliers;
  for (size_t iter = 0; iter < iterations; ++iter) {
    std::vector<size_t> sample;
    while (sample.size() < sample_size) {
      const auto idx = dist(rng);
      if (std::find(sample.begin(), sample.end(), idx) == sample.end()) {
        sample.push_back(idx);
      }
    }

    // reject degenerate samples (points too close to define a rotation)
    if ((robot[sample[0]] - robot[sample[1]]).head<2>().norm() < 2.0 * inlier_threshold_m) {
      continue;
    }

    const auto candidate = fitTransform(robot, world, sample, yaw_only);
    std::vector<size_t> inliers;
    for (size_t i = 0; i < robot.size(); ++i) {
      if (residual(candidate, robot[i], world[i], yaw_only) < inlier_threshold_m) {
        inliers.push_back(i);
      }
    }

    if (inliers.size() > best_inliers.size()) {
      best_inliers = inliers;
    }
  }

  if (best_inliers.size() < sample_size + 1) {
    return std::nullopt;
  }

  GpsAlignment result;
  result.world_T_robot = fitTransform(robot, world, best_inliers, yaw_only);
  double sq_error = 0.0;
  for (const auto idx : best_inliers) {
    const double r = residual(result.world_T_robot, robot[idx], world[idx], yaw_only);
    sq_error += r * r;
  }

  result.num_inliers = best_inliers.size();
  result.rmse = std::sqrt(sq_error / best_inliers.size());
  return result;
}

}  // namespace hydra_multi
