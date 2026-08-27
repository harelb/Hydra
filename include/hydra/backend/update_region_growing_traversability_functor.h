/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#pragma once

#include "hydra/backend/deformation_interpolator.h"
#include "hydra/backend/update_functions.h"
#include "hydra/utils/logging.h"
#include "hydra/utils/nearest_neighbor_utilities.h"

namespace hydra {

/**
 * @brief Functor to update traversability places in the DSG. This functor should be
 * called with exhaustive merging enabled.
 */
struct UpdateRegionGrowingTraversabilityFunctor : public UpdateFunctor {
  struct Config : public VerbosityConfig {
    //! Layer to update traversability in
    std::string layer = DsgLayers::TRAVERSABILITY;

    //! Maximum centroid-to-centroid distance [m] at which two places may be linked by
    //! the proximity fallback. 0 (the default) disables the fallback entirely, leaving
    //! only the star-polygon intersection test. The fallback exists because
    //! TravNodeAttributes::fromExteriorPoints takes the MINIMUM radius per angular bin
    //! (and fills empty bins with the global minimum), so the stored polygon is a
    //! heavily shrunk INNER approximation of the region; since the region-growing
    //! clustering partitions voxels disjointly, adjacent regions abut rather than
    //! overlap and intersects() almost never fires across an active-window boundary.
    double max_connection_distance_m = 0.0;

    //! Maximum gap [m] that may remain between the two boundaries along the connecting
    //! ray for the proximity fallback to fire, i.e. distance - reach_1 - reach_2 where
    //! reach_i is place i's own boundary radius in the direction of the other place.
    //! This is what keeps the fallback from linking two places that merely happen to be
    //! close: they must very nearly touch along the direction we are linking them in.
    double max_connection_gap_m = 1.0;

    //! Require the boundary bins facing the other place to be TRAVERSABLE on both
    //! sides. The exterior boundary voxel states record why the region stopped growing
    //! in that direction (a wall yields INTRAVERSABLE, unobserved space UNKNOWN), so
    //! this is the available evidence that the ray between the two centroids is not
    //! crossing an obstacle. Disabling it allows linking through walls.
    bool require_traversable_boundary = true;

    DeformationInterpolator::Config deformation;
  } const config;

  using EdgeSet = std::set<EdgeKey>;
  using NodeSet = std::set<NodeId>;
  using State = spark_dsg::TraversabilityState;

  explicit UpdateRegionGrowingTraversabilityFunctor(const Config& config);

  Hooks hooks() const override;

  void call(const DynamicSceneGraph& unmerged,
            SharedDsgInfo& dsg,
            const UpdateInfo::ConstPtr& info) const override;

 protected:
  // Hook callbacks.
  MergeList findNodeMerges(const DynamicSceneGraph& dsg,
                           const UpdateInfo::ConstPtr& info) const;

  NodeAttributes::Ptr mergeNodes(const DynamicSceneGraph& dsg,
                                 const std::vector<NodeId>& merge_ids) const;

  void cleanup(const UpdateInfo::ConstPtr& /* info */, SharedDsgInfo* /* dsg */) const;

  // Processing Steps.
  /**
   * @brief Update the positions of all traversability nodes in the DSG. Propagates to
   * the complete DSG in case of new loop closures.
   */
  void updateDeformation(const DynamicSceneGraph& unmerged,
                         SharedDsgInfo& dsg,
                         const UpdateInfo::ConstPtr& info) const;

  /**
   * @brief Remove all active window and inactive edges from the graph.
   */
  void resetAddedEdges(DynamicSceneGraph& dsg) const;

  /**
   * @brief Compute edges between overlapping inactive nodes globally.
   */
  void findInactiveEdges(DynamicSceneGraph& dsg) const;

  /**
   * @brief Find and add edges from active to inactive nodes.
   */
  void findActiveWindowEdges(DynamicSceneGraph& dsg) const;

  /**
   * @brief Remove active window edges that no longer have active overlap and move
   * designate archived ones as inactive edges.
   */
  void pruneActiveWindowEdges(DynamicSceneGraph& dsg) const;

  // Helper functions.

  /**
   * @brief Find places that are inactive and spatially but not temporally overlap with
   * the given node.
   */
  std::vector<NodeId> findConnections(const DynamicSceneGraph& dsg,
                                      const TravNodeAttributes& from_attrs) const;

  /**
   * @brief Check whether two traversability nodes should be connected by an edge, i.e.
   * their boundaries overlap or (optionally) they nearly touch along the ray joining
   * them. Symmetric in its arguments.
   */
  bool areConnected(const TravNodeAttributes& attrs1,
                    const TravNodeAttributes& attrs2) const;

  /**
   * @brief Proximity fallback for areConnected: two places are linked if their
   * centroids are within max_connection_distance_m and the gap left between their two
   * boundaries along the connecting ray is at most max_connection_gap_m.
   */
  bool isNearlyTouching(const TravNodeAttributes& attrs1,
                        const TravNodeAttributes& attrs2) const;

  /**
   * @brief Distance from a node's centroid to its boundary in the given (local frame)
   * direction, interpolated between angular bins the same way contains() does. Returns
   * a negative value if the boundary in that direction is not traversable, or if there
   * is no boundary information at all.
   */
  double boundaryReach(const TravNodeAttributes& attrs,
                       const Eigen::Vector3d& direction_L) const;

  /**
   * @brief Check if two traversability nodes have active window (temporal) overlap.
   */
  static bool hasActiveOverlap(const TravNodeAttributes& attrs1,
                               const TravNodeAttributes& attrs2);

  /**
   * @brief View on all active nodes in a layer.
   */
  static LayerView activeNodes(const SceneGraphLayer& layer);

 protected:
  // Members.
  const DeformationInterpolator deformation_interpolator_;

  // State.
  mutable EdgeSet active_edges_;      // Active window edges in the current update.
  mutable EdgeSet merge_candidates_;  // List of nodes that have inactive edges and
                                      // could thus be merged this iteration.
};

void declare_config(UpdateRegionGrowingTraversabilityFunctor::Config& config);

}  // namespace hydra
