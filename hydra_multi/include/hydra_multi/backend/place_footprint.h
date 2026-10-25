#pragma once

#include <spark_dsg/node_attributes.h>

#include <Eigen/Core>
#include <vector>

namespace hydra_multi {

/**
 * @brief 2D footprint of a region-growing traversability place: a star polygon around
 * the place center with one boundary vertex per radius, at angle 2 * pi * i / N in the
 * world frame (see spark_dsg::TravNodeAttributes::getBoundaryPoint).
 */
struct PlaceFootprint {
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  std::vector<double> radii;

  PlaceFootprint() = default;
  PlaceFootprint(const Eigen::Vector2d& center, const std::vector<double>& radii);
  explicit PlaceFootprint(const spark_dsg::TravNodeAttributes& attrs);

  bool empty() const { return radii.empty(); }
  double maxRadius() const;
  //! Boundary vertex i in world coordinates
  Eigen::Vector2d vertex(size_t i) const;
  //! Same interpolated boundary test as TravNodeAttributes::contains
  bool contains(const Eigen::Vector2d& point) const;
  //! Polygon area
  double area() const;
  //! Largest t >= 0 at which the ray origin + t * dir crosses the polygon boundary
  //! (negative if the ray misses the polygon)
  double farthestRayHit(const Eigen::Vector2d& origin, const Eigen::Vector2d& dir) const;
  //! Whether the bounding circles of the two footprints overlap
  bool mayIntersect(const PlaceFootprint& other) const;
};

//! Area of the intersection of two footprints
double intersectionArea(const PlaceFootprint& a, const PlaceFootprint& b);

/**
 * @brief Rotate the boundary of a traversability place about its center by yaw (the
 * radii are indexed by world-frame angle, so a frame rotation has to shift them).
 * Radii are linearly interpolated, states take the closest bin.
 */
void rotateBoundary(spark_dsg::TravNodeAttributes& attrs, double yaw);

/**
 * @brief Star polygon around `center` (with `num_bins` rays) whose ray i reaches the
 * farthest boundary crossing of any member along that ray: the union of the members
 * as seen from the center.
 * @param member_index Optional output: index of the member that sets each ray (-1 if
 * none)
 */
PlaceFootprint unionFootprint(const Eigen::Vector2d& center,
                              size_t num_bins,
                              const std::vector<PlaceFootprint>& members,
                              std::vector<int>* member_index = nullptr);

/**
 * @brief Trim each footprint at the bisector with every other footprint that covers
 * the crossing point, so overlapping footprints split the shared ground between them
 * (a Voronoi split restricted to the overlap). All footprints are trimmed against the
 * untrimmed others.
 * @param trimmed Optional output: per footprint, which rays were trimmed
 */
std::vector<PlaceFootprint> trimFootprints(
    const std::vector<PlaceFootprint>& footprints,
    std::vector<std::vector<bool>>* trimmed = nullptr);

//! Write radii (and min/max radius) of a footprint into place attributes
void setBoundary(spark_dsg::TravNodeAttributes& attrs, const PlaceFootprint& footprint);

}  // namespace hydra_multi
