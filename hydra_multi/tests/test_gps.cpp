#include <gtest/gtest.h>
#include <hydra_multi/common/gps.h>

#include <filesystem>
#include <fstream>

namespace hydra_multi {

TEST(GpsTests, LatLonToUtm) {
  // reference values from pyproj (EPSG:32618 / EPSG:32756)
  auto utm = latLonToUtm(41.3561627, -74.0523442);
  EXPECT_EQ(18, utm.zone);
  EXPECT_TRUE(utm.northern);
  EXPECT_NEAR(579269.2033, utm.easting, 1.0e-3);
  EXPECT_NEAR(4578728.8910, utm.northing, 1.0e-3);

  utm = latLonToUtm(41.39, -73.95);
  EXPECT_NEAR(587784.6675, utm.easting, 1.0e-3);
  EXPECT_NEAR(4582584.0973, utm.northing, 1.0e-3);

  utm = latLonToUtm(-33.86, 151.21);
  EXPECT_EQ(56, utm.zone);
  EXPECT_FALSE(utm.northern);
  EXPECT_NEAR(334416.3940, utm.easting, 1.0e-3);
  EXPECT_NEAR(6251925.3604, utm.northing, 1.0e-3);
}

TEST(GpsTests, LoadCsv) {
  const auto path = std::filesystem::temp_directory_path() / "hydra_multi_test_gps.csv";
  {
    std::ofstream file(path);
    file << "timestamp_ns,latitude,longitude,altitude,status,cov_xx,cov_yy,cov_zz\n";
    file << "2000,41.39,-73.95,10.0,2,0.04,0.09,1.0\n";
    file << "1000,41.3561627,-74.0523442,184.085,0,0.017689,0.017689,0.009409\n";
  }

  const auto measurements = loadGpsCsv(path);
  std::filesystem::remove(path);
  ASSERT_EQ(2u, measurements.size());
  // sorted by time
  EXPECT_EQ(1000u, measurements[0].stamp_ns);
  EXPECT_NEAR(579269.2033, measurements[0].utm.x(), 1.0e-3);
  EXPECT_NEAR(184.085, measurements[0].utm.z(), 1.0e-9);
  EXPECT_EQ(0, measurements[0].status);
  EXPECT_EQ(2, measurements[1].status);
  EXPECT_NEAR(0.09, measurements[1].variance.y(), 1.0e-12);
}

TEST(GpsTests, InterpolatePose) {
  std::vector<pose_graph_tools::PoseGraphNode> nodes(2);
  nodes[0].stamp_ns = 1000;
  nodes[0].key = 3;
  nodes[1].stamp_ns = 2000;
  nodes[1].key = 4;
  nodes[1].pose.translation() << 10.0, 0.0, 0.0;

  EXPECT_FALSE(interpolatePose(nodes, 500, 1000));
  EXPECT_FALSE(interpolatePose(nodes, 2500, 1000));
  // too far from either node
  EXPECT_FALSE(interpolatePose(nodes, 1500, 100));

  auto result = interpolatePose(nodes, 1200, 1000);
  ASSERT_TRUE(result);
  EXPECT_EQ(0u, result->first);
  EXPECT_NEAR(2.0, result->second.x(), 1.0e-9);

  result = interpolatePose(nodes, 1700, 1000);
  ASSERT_TRUE(result);
  EXPECT_EQ(1u, result->first);
  EXPECT_NEAR(7.0, result->second.x(), 1.0e-9);

  result = interpolatePose(nodes, 2000, 0);
  ASSERT_TRUE(result);
  EXPECT_EQ(1u, result->first);
  EXPECT_NEAR(10.0, result->second.x(), 1.0e-9);
}

TEST(GpsTests, AlignPointsWithOutliers) {
  const gtsam::Pose3 world_T_robot(gtsam::Rot3::Yaw(2.0), gtsam::Point3(500.0, -20.0, 3.0));
  std::vector<Eigen::Vector3d> robot;
  std::vector<Eigen::Vector3d> world;
  for (size_t i = 0; i < 50; ++i) {
    const Eigen::Vector3d p(0.5 * i, 3.0 * std::sin(0.2 * i), 0.01 * i);
    robot.push_back(p);
    world.push_back(world_T_robot.transformFrom(p));
  }

  // gross outliers (multipath)
  world[5] += Eigen::Vector3d(40.0, 0.0, 0.0);
  world[17] += Eigen::Vector3d(0.0, -25.0, 0.0);
  world[30] += Eigen::Vector3d(15.0, 15.0, 0.0);

  const auto result = alignPoints(robot, world, 1.0);
  ASSERT_TRUE(result);
  EXPECT_EQ(47u, result->num_inliers);
  EXPECT_NEAR(0.0, result->rmse, 1.0e-6);
  EXPECT_TRUE(gtsam::assert_equal(world_T_robot, result->world_T_robot, 1.0e-6));

  // too few points
  EXPECT_FALSE(alignPoints({robot[0], robot[1]}, {world[0], world[1]}, 1.0));
}

}  // namespace hydra_multi
