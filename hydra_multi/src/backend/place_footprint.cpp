#include "hydra_multi/backend/place_footprint.h"

#include <glog/logging.h>

#include <algorithm>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <cmath>
#include <deque>
#include <limits>
#include <numbers>

namespace hydra_multi {

namespace bg = boost::geometry;
using BgPoint = bg::model::d2::point_xy<double>;
using BgPolygon = bg::model::polygon<BgPoint, false>;  // counter-clockwise, closed

namespace {

BgPolygon toPolygon(const PlaceFootprint& footprint) {
  BgPolygon poly;
  auto& ring = poly.outer();
  ring.reserve(footprint.radii.size() + 1);
  for (size_t i = 0; i < footprint.radii.size(); ++i) {
    const auto v = footprint.vertex(i);
    ring.emplace_back(v.x(), v.y());
  }
  if (!ring.empty()) {
    ring.push_back(ring.front());
  }
  bg::correct(poly);
  return poly;
}

inline double cross(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  return a.x() * b.y() - a.y() * b.x();
}

}  // namespace

PlaceFootprint::PlaceFootprint(const Eigen::Vector2d& center,
                               const std::vector<double>& radii)
    : center(center), radii(radii) {}

PlaceFootprint::PlaceFootprint(const spark_dsg::TravNodeAttributes& attrs)
    : center(attrs.position.head<2>()), radii(attrs.radii) {}

double PlaceFootprint::maxRadius() const {
  return radii.empty() ? 0.0 : *std::max_element(radii.begin(), radii.end());
}

Eigen::Vector2d PlaceFootprint::vertex(size_t i) const {
  const double angle = 2.0 * std::numbers::pi * static_cast<double>(i) /
                       static_cast<double>(radii.size());
  return center + radii[i] * Eigen::Vector2d(std::cos(angle), std::sin(angle));
}

bool PlaceFootprint::contains(const Eigen::Vector2d& point) const {
  if (radii.empty()) {
    return false;
  }
  const Eigen::Vector2d p = point - center;
  const double distance = p.norm();
  if (distance < 1.0e-9) {
    return true;
  }
  double bin = std::atan2(p.y(), p.x()) / (2.0 * std::numbers::pi);
  bin = (bin >= 0.0 ? bin : bin + 1.0) * radii.size();
  const size_t left = static_cast<size_t>(std::floor(bin)) % radii.size();
  const size_t right = (left + 1) % radii.size();
  const double frac = bin - std::floor(bin);
  return distance <= radii[left] + (radii[right] - radii[left]) * frac;
}

double PlaceFootprint::area() const {
  if (radii.size() < 3) {
    return 0.0;
  }
  return std::abs(bg::area(toPolygon(*this)));
}

double PlaceFootprint::farthestRayHit(const Eigen::Vector2d& origin,
                                      const Eigen::Vector2d& dir) const {
  double best = -1.0;
  const size_t n = radii.size();
  for (size_t i = 0; i < n; ++i) {
    const Eigen::Vector2d a = vertex(i);
    const Eigen::Vector2d e = vertex((i + 1) % n) - a;
    const double den = cross(dir, e);
    if (std::abs(den) < 1.0e-12) {
      continue;
    }
    const Eigen::Vector2d w = a - origin;
    const double t = cross(w, e) / den;    // along the ray
    const double s = cross(w, dir) / den;  // along the edge
    if (t >= 0.0 && s >= -1.0e-9 && s <= 1.0 + 1.0e-9) {
      best = std::max(best, t);
    }
  }
  return best;
}

bool PlaceFootprint::mayIntersect(const PlaceFootprint& other) const {
  const double reach = maxRadius() + other.maxRadius();
  return (other.center - center).squaredNorm() <= reach * reach;
}

double intersectionArea(const PlaceFootprint& a, const PlaceFootprint& b) {
  if (a.radii.size() < 3 || b.radii.size() < 3 || !a.mayIntersect(b)) {
    return 0.0;
  }
  try {
    std::deque<BgPolygon> output;
    bg::intersection(toPolygon(a), toPolygon(b), output);
    double area = 0.0;
    for (const auto& poly : output) {
      area += std::abs(bg::area(poly));
    }
    return area;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Footprint intersection failed: " << e.what();
    return 0.0;
  }
}

void rotateBoundary(spark_dsg::TravNodeAttributes& attrs, double yaw) {
  const size_t n = attrs.radii.size();
  if (n == 0) {
    return;
  }
  const double shift = yaw / (2.0 * std::numbers::pi) * static_cast<double>(n);
  std::vector<double> radii(n);
  spark_dsg::TraversabilityStates states(attrs.states.size());
  for (size_t i = 0; i < n; ++i) {
    // new bin i looks at the old angle (i - shift)
    double x = std::fmod(static_cast<double>(i) - shift, static_cast<double>(n));
    x = x < 0.0 ? x + n : x;
    const size_t left = static_cast<size_t>(std::floor(x)) % n;
    const size_t right = (left + 1) % n;
    const double frac = x - std::floor(x);
    radii[i] = attrs.radii[left] + (attrs.radii[right] - attrs.radii[left]) * frac;
    if (states.size() == n) {
      states[i] = attrs.states[frac < 0.5 ? left : right];
    }
  }
  attrs.radii = radii;
  attrs.states = states;
  if (!radii.empty()) {
    attrs.min_radius = *std::min_element(radii.begin(), radii.end());
    attrs.max_radius = *std::max_element(radii.begin(), radii.end());
  }
}

PlaceFootprint unionFootprint(const Eigen::Vector2d& center,
                              size_t num_bins,
                              const std::vector<PlaceFootprint>& members,
                              std::vector<int>* member_index) {
  PlaceFootprint result(center, std::vector<double>(num_bins, 0.0));
  if (member_index) {
    member_index->assign(num_bins, -1);
  }
  for (size_t i = 0; i < num_bins; ++i) {
    const double angle =
        2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_bins);
    const Eigen::Vector2d dir(std::cos(angle), std::sin(angle));
    for (size_t m = 0; m < members.size(); ++m) {
      const double t = members[m].farthestRayHit(center, dir);
      if (t > result.radii[i]) {
        result.radii[i] = t;
        if (member_index) {
          (*member_index)[i] = static_cast<int>(m);
        }
      }
    }
  }
  return result;
}

std::vector<PlaceFootprint> trimFootprints(const std::vector<PlaceFootprint>& footprints,
                                           std::vector<std::vector<bool>>* trimmed) {
  std::vector<PlaceFootprint> result = footprints;
  if (trimmed) {
    trimmed->clear();
    for (const auto& footprint : footprints) {
      trimmed->emplace_back(footprint.radii.size(), false);
    }
  }

  // sort by x to only visit footprints whose bounding circles can overlap
  std::vector<size_t> order(footprints.size());
  std::vector<double> reach(footprints.size());
  double max_reach = 0.0;
  for (size_t i = 0; i < footprints.size(); ++i) {
    order[i] = i;
    reach[i] = footprints[i].maxRadius();
    max_reach = std::max(max_reach, reach[i]);
  }
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return footprints[a].center.x() < footprints[b].center.x();
  });

  const auto trim_against = [&](size_t a, size_t b) {
    const auto& from = footprints[a];
    const auto& other = footprints[b];
    auto& radii = result[a].radii;
    const Eigen::Vector2d d = other.center - from.center;
    const double dd = d.squaredNorm();
    const size_t n = radii.size();
    for (size_t i = 0; i < n; ++i) {
      const double angle =
          2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(n);
      const Eigen::Vector2d dir(std::cos(angle), std::sin(angle));
      const double s = dir.dot(d);
      if (s <= 1.0e-9) {
        continue;  // the bisector is behind this ray
      }
      const double t_bisector = dd / (2.0 * s);
      if (t_bisector < radii[i] && other.contains(from.center + t_bisector * dir)) {
        radii[i] = t_bisector;
        if (trimmed) {
          (*trimmed)[a][i] = true;
        }
      }
    }
  };

  for (size_t oi = 0; oi < order.size(); ++oi) {
    const size_t a = order[oi];
    for (size_t oj = oi + 1; oj < order.size(); ++oj) {
      const size_t b = order[oj];
      if (footprints[b].center.x() - footprints[a].center.x() > reach[a] + max_reach) {
        break;
      }
      if (!footprints[a].mayIntersect(footprints[b])) {
        continue;
      }
      if (footprints[a].empty() || footprints[b].empty()) {
        continue;
      }
      trim_against(a, b);
      trim_against(b, a);
    }
  }
  return result;
}

void setBoundary(spark_dsg::TravNodeAttributes& attrs, const PlaceFootprint& footprint) {
  attrs.radii = footprint.radii;
  if (footprint.radii.empty()) {
    attrs.min_radius = 0.0;
    attrs.max_radius = 0.0;
    return;
  }
  attrs.min_radius = *std::min_element(footprint.radii.begin(), footprint.radii.end());
  attrs.max_radius = *std::max_element(footprint.radii.begin(), footprint.radii.end());
}

}  // namespace hydra_multi
