#pragma once

#include <hydra/backend/updates/update_region_growing_traversability_functor.h>

#include "hydra_multi/backend/place_footprint.h"

namespace hydra_multi {

/**
 * @brief Region-growing traversability update for fused multi-robot graphs.
 *
 * Runs the per-robot update (deformation, place-place edges) and, when merging is
 * enabled, replaces its merge rule (merge only when a place center lies inside the
 * other place, keep one of the two footprints) with:
 * 1. Places whose footprints overlap by at least `min_overlap` of the smaller one
 *    (across robots, optionally also within one robot) are grouped around the largest
 *    place of each group (no chaining: every member overlaps the group's anchor).
 * 2. Each group becomes one place: the anchor (stable id) with the union of the member
 *    footprints, the earliest / latest observation stamps and the area-weighted mean
 *    semantic feature. Merges go through the DsgUpdater merge tracking, so edges are
 *    rewired and the merges are reported like any other node merge.
 * 3. If `trim_overlaps` is set, the merged footprints split the ground they still share
 *    with neighbours at their bisectors, and groups that still overlap by at least
 *    `min_overlap` are merged again (up to `max_merge_iterations` rounds).
 */
struct UpdateMultiTraversabilityFunctor
    : public hydra::UpdateRegionGrowingTraversabilityFunctor {
  struct Config : public hydra::UpdateRegionGrowingTraversabilityFunctor::Config {
    //! Merge overlapping places (false keeps the per-robot merge rule)
    bool merge_places = true;
    //! Also merge overlapping places of the same robot
    bool merge_same_robot = true;
    //! Minimum intersection area over the smaller footprint area to merge two places
    double min_overlap = 0.3;
    //! Split remaining overlaps between neighbouring places at their bisectors
    bool trim_overlaps = true;
    //! Rounds of merging after trimming
    size_t max_merge_iterations = 5;
  } const multi_config;

  explicit UpdateMultiTraversabilityFunctor(const Config& config);

  Hooks hooks() const override;

  void call(const spark_dsg::SceneGraph& unmerged,
            hydra::SharedDsgInfo& dsg,
            const hydra::UpdateInfo::ConstPtr& info) const override;

  //! Merge groups (anchor first) of the places in the layer of `graph`
  std::vector<std::vector<spark_dsg::NodeId>> findPlaceGroups(
      const spark_dsg::SceneGraph& graph,
      const std::unordered_map<spark_dsg::NodeId, size_t>* node_to_robot) const;

  //! Attributes of the merged place: the first id is the surviving place
  spark_dsg::NodeAttributes::Ptr mergePlaces(
      const spark_dsg::SceneGraph& unmerged,
      const std::vector<spark_dsg::NodeId>& merge_ids) const;

  //! Split overlaps between the places in the layer of `graph`
  void trimPlaces(spark_dsg::SceneGraph& graph) const;

 protected:
  hydra::MergeList findPlaceMerges(const spark_dsg::SceneGraph& graph,
                                   const hydra::UpdateInfo::ConstPtr& info) const;

  mutable bool merges_found_ = false;
  mutable bool trim_pending_ = false;
};

void declare_config(UpdateMultiTraversabilityFunctor::Config& config);

}  // namespace hydra_multi
