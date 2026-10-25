#pragma once
#include <config_utilities/factory.h>
#include <config_utilities/virtual_config.h>
#include <hydra/backend/dsg_updater.h>
#include <hydra/backend/external_loop_closure_receiver.h>
#include <hydra/backend/merge_tracker.h>
#include <hydra/backend/update_functions.h>
#include <hydra/common/output_sink.h>
#include <kimera_pgmo/kimera_pgmo_interface.h>

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <unordered_map>

#include "hydra_multi/backend/initial_align.h"
#include "hydra_multi/common/multi_dsg_info.h"
#include "hydra_multi/common/types.h"
#include "hydra_multi/interface/interface_container.h"

namespace hydra_multi {

struct LoopClosureLog {
  gtsam::Symbol src;        // factor edge "source"
  gtsam::Symbol dest;       // factor edge "dest"
  gtsam::Pose3 src_T_dest;  // src_frame.between(dest_frame)
  bool dsg;
  int64_t level;
  bool added_to_dgraph = false;  // track if already added to deformation graph
};

struct MultiBackendModuleStatus {
  size_t new_loop_closures = 0;
  size_t total_loop_closures = 0;
  size_t total_interrobot_loop_closures = 0;
  size_t inlier_loop_closures = 0;
  size_t inlier_interrobot_loop_closures = 0;
  size_t total_factors = 0;
  size_t total_values = 0;
  size_t new_factors = 0;
  size_t new_graph_factors = 0;
  size_t trajectory_len = 0;
  size_t num_merges_undone = 0;
  std::optional<double> spin_s;
  std::optional<double> opt_s;
  std::optional<double> mesh_s;
  void reset();
};

std::ostream& operator<<(std::ostream& os, const MultiBackendModuleStatus& status);

class MultiBackendModule : public kimera_pgmo::KimeraPgmoInterface {
 public:
  using Ptr = std::shared_ptr<MultiBackendModule>;
  using Sink = hydra::OutputSink<uint64_t,
                                 const spark_dsg::SceneGraph&,
                                 const kimera_pgmo::DeformationGraph&,
                                 const MultiBackendModuleStatus&>;
  using NodeToRobotMap = std::unordered_map<spark_dsg::NodeId, size_t>;

  //! Absolute position (GPS) priors on the agent pose graphs
  struct GpsConfig {
    //! Add GPS priors and align each robot to the GPS world frame
    bool enable = false;
    //! UTM coordinate (easting, northing, altitude) of the world frame origin. Empty
    //! uses the first fix of the lowest robot id
    std::vector<double> origin_utm;
    //! Lower bound on the standard deviation of each fix
    double min_sigma_m = 0.1;
    //! Scale applied to the reported standard deviation of each fix
    double sigma_scale = 1.0;
    //! Constrain altitude (GPS altitude is usually much worse than position)
    bool use_altitude = false;
    //! Standard deviation applied to altitude when use_altitude is false
    double altitude_sigma_m = 100.0;
    //! Maximum time between a fix and the closest agent pose
    double max_time_diff_s = 1.0;
    //! Fixes with horizontal standard deviation below this are known inliers for GNC
    //! (negative disables)
    double known_inlier_max_sigma_m = -1.0;
    //! Minimum number of fixes required to align a robot to the GPS frame
    size_t min_alignment_fixes = 10;
    //! Inlier threshold for the initial alignment
    double alignment_inlier_threshold_m = 3.0;
    //! Only estimate yaw and translation in the initial alignment
    bool align_yaw_only = true;
    //! Standard deviation of a gravity (roll / pitch) prior on every pose of a robot with
    //! GPS priors, assuming gravity-aligned odometry (non-positive disables). Without it,
    //! position priors with an unconstrained altitude leave the trajectory free to tilt
    double gravity_sigma = 0.02;
    //! Robots without a GPS alignment are initialized from at least this many loop
    //! closures to robots that have one (0 disables)
    size_t min_alignment_loop_closures = 3;
    //! Inter-robot loop closures between robots in the GPS frame that disagree with the
    //! aligned initial guess by more than this are discarded (non-positive disables)
    double loop_closure_gate_m = 5.0;
    //! Measure the loop-closure gate and the loop-closure alignment consensus horizontally
    //! (world xy) only, so odometric height drift of unanchored robots does not count
    bool gate_horizontal_only = false;
    //! A loop-closure alignment is rejected when the median horizontal distance to the
    //! robot's coarse GPS fixes exceeds this (non-positive disables)
    double coarse_gate_m = 20.0;
  };

  struct Config : hydra::DsgUpdater::Config {
    //! Actually perform PGO on every detected loop closure
    bool optimize_on_lc = true;
    //! Specialized PGMO configuration that includes scene graph factor covariances
    kimera_pgmo::KimeraPgmoConfig pgmo;
    //! Add prior to robot ID (negative disables anchor)
    int anchor_robot_id = 0;
    //! PGO versus PGMO
    bool include_mesh_factors = true;
    //! Drop mesh faces torn by the deformation: a face is dropped when one of its edges
    //! is longer than torn_face_stretch_ratio times its undeformed length and longer
    //! than torn_face_min_edge_m after deformation (non-positive ratio disables)
    double torn_face_stretch_ratio = 0.0;
    double torn_face_min_edge_m = 0.5;
    //! Configuration for associating external loop closures
    hydra::ExternalLoopClosureReceiver::Config external_loop_closures;
    //! Scene graph loop closure variance
    double sg_loop_closure_variance = 0.1;
    //! Output sinks that process that latest backed scene graph and state
    std::vector<Sink::Factory> sinks;
    //! Initial alginment module
    config::VirtualConfig<InitialAlignModule> initial_align;
    //! Partial optimizer
    config::VirtualConfig<kimera_pgmo::Optimizer> partial_optimizer{
        kimera_pgmo::KimeraRpgoOptimizer::Config()};
    //! GPS priors
    GpsConfig gps;
  } const config;

  MultiBackendModule(const Config& config,
                     const InterfaceContainer::StatesPtr& shared_states);

  virtual ~MultiBackendModule();

  MultiBackendModule(const MultiBackendModule& other) = delete;

  MultiBackendModule& operator=(const MultiBackendModule& other) = delete;

  void start();

  void stop();

  void save(const DataDirectory& output);

  std::string printInfo() const;

  void addSink(const Sink::Ptr& sink);

  void addLoopClosures(const pose_graph_tools::PoseGraph& loop_closures);

  bool spinOnce(Timestamp timestamp_ns);

  bool optimizeOnce(Timestamp timestamp_ns);

  //! Whether the last update added any GPS priors
  bool hasGpsPriors() const { return num_gps_priors_ > 0; }

 protected:
  void spin();

  void optimizeSpin();

  void stopImpl();

  bool pollSharedStates(Timestamp& stamp, bool mark_updated = true) const;

  bool updateFromStates(Timestamp timestamp_ns);

  void combineAndDeformMesh(Timestamp timestamp_ns, bool force_mesh_update = false);

  //! Drop the faces of a deformed mesh torn by the deformation (returns number dropped)
  size_t dropTornFaces(MeshData& mesh_data) const;

  bool processLoopClosures(Timestamp timestamp_ns);

  void addLoopClosures();

  void initialAlignment();

  void optimizeStartup() override;

  void optimizeCleanup(const gtsam::Values& results) override;

  void optimize(size_t timestamp_ns);

  void partialOptimization(const Timestamp& timestamp_ns);

  gtsam::Pose3 getRobotWorldFrame(RobotId robot_id, gtsam::Values& optimized);

  void rebaseStates(size_t timestamps_ns);

  void logStatus();

  void alignRobotToGps(RobotId robot_id, UnitInterfaceState& state);

  size_t addGpsPriors(RobotId robot_id, const UnitInterfaceState& state);

  void alignRobotsWithLoopClosures();

  void gateLoopClosures();

 protected:
  std::unique_ptr<std::thread> spin_thread_;
  std::unique_ptr<std::thread> optimization_thread_;

  std::atomic<bool> should_shutdown_{false};
  std::atomic<bool> should_deform_{false};
  bool have_loopclosures_ = false;
  bool have_new_mesh_ = false;
  uint64_t last_sequence_number_ = 0;

  struct GuardedValues {
    std::mutex mutex;
    gtsam::Values values;
  } optimized_;

  std::mutex status_mutex_;
  MultiBackendModuleStatus status_;
  std::vector<MultiBackendModuleStatus> status_log_;

  std::list<LoopClosureLog> loop_closures_;
  hydra::MessageQueue<pose_graph_tools::PoseGraph> lc_queue_;
  hydra::MessageQueue<LoopClosureLog> interrobot_lc_queue_;
  hydra::ExternalLoopClosureReceiver lc_receiver_;

  std::map<pose_graph_tools::PoseGraphEdge::Type, double> edge_variance_map_;

  Sink::List sinks_;

  MultiDsgInfo::Ptr unmerged_dsg_;
  MultiDsgInfo::Ptr merged_dsg_;
  InterfaceContainer::StatesPtr shared_states_;
  std::map<size_t, MeshData::Ptr> staged_mesh_data_;
  std::map<size_t, gtsam::Pose3> curr_W_T_robot_;

  InitialAlignModule::Ptr initial_align_;
  hydra::DsgUpdater::Ptr dsg_updater_;
  std::optional<Eigen::Vector3d> gps_origin_utm_;
  std::set<RobotId> gps_aligned_robots_;
  size_t num_gps_priors_ = 0;
  kimera_pgmo::Optimizer::Ptr partial_pgo_;
};

void declare_config(MultiBackendModule::GpsConfig& config);

void declare_config(MultiBackendModule::Config& config);

}  // namespace hydra_multi
