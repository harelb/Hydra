#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "hydra/places/region_growing_traversability_clustering.h"

namespace hydra::places {

// Test shim: expose the protected static helpers.
struct RegionGrowingTest : public RegionGrowingTraversabilityClustering {
  using RegionGrowingTraversabilityClustering::RegionGrowingTraversabilityClustering;
  using RegionGrowingTraversabilityClustering::growRegion;
  using RegionGrowingTraversabilityClustering::erodeCandidates;
  using RegionGrowingTraversabilityClustering::growConnectedWithMinWidth;
  using RegionGrowingTraversabilityClustering::enclosedUnknownFill;
};

using VoxelSet = RegionGrowingTraversabilityClustering::VoxelSet;

namespace {
VoxelSet makeSet(const std::vector<std::array<int, 2>>& pts) {
  VoxelSet s;
  for (const auto& p : pts) {
    s.insert(VoxelIndex(p[0], p[1], 0));
  }
  return s;
}
}  // namespace

TEST(RegionGrowingConnectivity, DiagonalGapBlockedBy4Connectivity) {
  // Two 1-voxel rooms touching only diagonally: (0,0) and (1,1).
  const VoxelSet candidates = makeSet({{0, 0}, {1, 1}});

  // 8-connected: diagonal counts -> both reachable.
  const auto c8 = RegionGrowingTest::growRegion(candidates, VoxelIndex(0, 0, 0), 8u);
  EXPECT_EQ(c8.size(), 2u);

  // 4-connected: diagonal does NOT count -> only the seed room.
  const auto c4 = RegionGrowingTest::growRegion(candidates, VoxelIndex(0, 0, 0), 4u);
  EXPECT_EQ(c4.size(), 1u);
  EXPECT_TRUE(c4.count(VoxelIndex(0, 0, 0)));
  EXPECT_FALSE(c4.count(VoxelIndex(1, 1, 0)));
}

namespace {
VoxelSet makeRoom(int x0, int x1, int y0, int y1) {
  VoxelSet s;
  for (int x = x0; x <= x1; ++x) {
    for (int y = y0; y <= y1; ++y) {
      s.insert(VoxelIndex(x, y, 0));
    }
  }
  return s;
}
}  // namespace

TEST(RegionGrowingConnectivity, MinWidthSeversThinBridgeKeepsRoom) {
  // Room A [0..2]x[0..2], Room B [4..6]x[0..2], joined by a 1-voxel bridge (3,1).
  VoxelSet candidates = makeRoom(0, 2, 0, 2);
  for (const auto& v : makeRoom(4, 6, 0, 2)) candidates.insert(v);
  candidates.insert(VoxelIndex(3, 1, 0));

  const int radius = 1;  // min_connection_width_voxels = 2 -> radius = 1
  const VoxelSet core = RegionGrowingTest::erodeCandidates(candidates, radius);
  EXPECT_FALSE(core.count(VoxelIndex(3, 1, 0)));  // 1-wide bridge eroded away
  EXPECT_TRUE(core.count(VoxelIndex(1, 1, 0)));   // room-A interior survives

  const auto result = RegionGrowingTest::growConnectedWithMinWidth(
      candidates, core, VoxelIndex(1, 1, 0), 4u);
  // Room B is unreachable across the thin bridge.
  EXPECT_FALSE(result.count(VoxelIndex(4, 1, 0)));
  EXPECT_FALSE(result.count(VoxelIndex(5, 1, 0)));
  // Room A preserved (center + edges reachable through core).
  EXPECT_TRUE(result.count(VoxelIndex(1, 1, 0)));
  EXPECT_TRUE(result.count(VoxelIndex(2, 1, 0)));
  EXPECT_TRUE(result.count(VoxelIndex(0, 1, 0)));
}

TEST(RegionGrowingConnectivity, WideDoorwayConnects) {
  // Same rooms but a full 3-wide doorway at x=3 (y=0,1,2).
  VoxelSet candidates = makeRoom(0, 2, 0, 2);
  for (const auto& v : makeRoom(4, 6, 0, 2)) candidates.insert(v);
  for (int y = 0; y <= 2; ++y) candidates.insert(VoxelIndex(3, y, 0));

  const int radius = 1;
  const VoxelSet core = RegionGrowingTest::erodeCandidates(candidates, radius);
  EXPECT_TRUE(core.count(VoxelIndex(3, 1, 0)));  // wide doorway survives erosion

  const auto result = RegionGrowingTest::growConnectedWithMinWidth(
      candidates, core, VoxelIndex(1, 1, 0), 4u);
  EXPECT_TRUE(result.count(VoxelIndex(5, 1, 0)));  // room B reached via doorway
}

TEST(RegionGrowingConnectivity, EnclosedUnknownHoleFilled) {
  // 5x5 connected traversable region with the center (2,2) carved out as UNKNOWN.
  VoxelSet connected = makeRoom(0, 4, 0, 4);
  connected.erase(VoxelIndex(2, 2, 0));
  const VoxelSet unknown = makeSet({{2, 2}});

  const auto fill = RegionGrowingTest::enclosedUnknownFill(
      connected, unknown, /*max_hole_voxels=*/10, 4u);
  EXPECT_TRUE(fill.count(VoxelIndex(2, 2, 0)));  // small enclosed pocket -> filled
}

TEST(RegionGrowingConnectivity, LargeUnknownExteriorNotFilled) {
  // A small connected patch beside a big UNKNOWN region: the exterior must NOT fill.
  const VoxelSet connected = makeSet({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
  VoxelSet unknown;
  for (int x = 2; x <= 9; ++x)
    for (int y = 0; y <= 9; ++y) unknown.insert(VoxelIndex(x, y, 0));  // 80 voxels

  const auto fill = RegionGrowingTest::enclosedUnknownFill(
      connected, unknown, /*max_hole_voxels=*/10, 4u);
  EXPECT_TRUE(fill.empty());  // component (80) exceeds cap -> left as exterior
}

TEST(RegionGrowingConnectivity, DetachedUnknownNotFilled) {
  // Small UNKNOWN pocket not adjacent to the connected region -> not filled.
  const VoxelSet connected = makeRoom(0, 2, 0, 2);
  const VoxelSet unknown = makeSet({{10, 10}});  // isolated, far away

  const auto fill = RegionGrowingTest::enclosedUnknownFill(
      connected, unknown, /*max_hole_voxels=*/10, 4u);
  EXPECT_TRUE(fill.empty());
}

}  // namespace hydra::places
