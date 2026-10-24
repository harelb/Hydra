#include "hydra_multi/backend/module.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/backend/backend_utilities.h>
#include <hydra/backend/mst_factors.h>
#include <hydra/common/launch_callbacks.h>
#include <hydra/common/pipeline_queues.h>
#include <hydra/utils/pgmo_mesh_traits.h>
#include <hydra/utils/timing_utilities.h>
#include <kimera_pgmo/utils/mesh_io.h>
#include <kimera_rpgo/utils/g2o.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ranges>

#include "hydra_multi/backend/loop_closure_initial_align.h"
#include "hydra_multi/backend/utils.h"
#include "hydra_multi/common/multi_global_info.h"

namespace hydra_multi {

using hydra::DsgUpdater;
using hydra::timing::ScopedTimer;
using spark_dsg::AgentNodeAttributes;
using spark_dsg::DsgLayers;
using spark_dsg::Mesh;
using spark_dsg::NodeId;
using spark_dsg::NodeSymbol;
using spark_dsg::SceneGraphLayer;

namespace {

inline gtsam::Pose3 projectToClosestSE3(const Eigen::Matrix4d& T) {
  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R_proj = svd.matrixU() * svd.matrixV().transpose();

  if (R_proj.determinant() < 0) {
    Eigen::Matrix3d U = svd.matrixU();
    U.col(2) *= -1;
    R_proj = U * svd.matrixV().transpose();
  }

  Eigen::Vector3d t = T.block<3, 1>(0, 3);

  return gtsam::Pose3(gtsam::Rot3(R_proj), gtsam::Point3(t));
}

inline static const auto registration_ =
    config::RegistrationWithConfig<MultiBackendModule,
                                   MultiBackendModule,
                                   MultiBackendModule::Config,
                                   InterfaceContainer::StatesPtr>("MultiBackendModule");

void writeBackendStatus(const std::vector<MultiBackendModuleStatus>& entries,
                        const std::filesystem::path& filename) {
  std::ofstream file(filename, std::ofstream::out);
  // file format
  file << "total_lc,new_lc,total_factors,total_values,new_factors,new_graph_"
          "factors,trajectory_len,run_time,optimize_time,mesh_update_time,num_"
          "merges_"
          "undone\n";

  const auto nan = std::numeric_limits<double>::quiet_NaN();
  for (const auto& entry : entries) {
    file << entry.total_loop_closures << "," << entry.new_loop_closures << ","
         << entry.total_factors << "," << entry.total_values << "," << entry.new_factors
         << "," << entry.new_graph_factors << "," << entry.trajectory_len << ","
         << entry.spin_s.value_or(nan) << "," << entry.opt_s.value_or(nan) << ","
         << entry.mesh_s.value_or(nan) << "," << entry.num_merges_undone << "\n";
  }

  file.close();
}

void saveTrajectory(const SceneGraphLayer& layer,
                    const std::filesystem::path& output_path) {
  std::ofstream fout(output_path);
  fout << "#timestamp_kf,x,y,z,qw,qx,qy,qz\n";
  for (const auto& node : layer.nodes()) {
    const auto attrs = node.tryAttributes<AgentNodeAttributes>();
    if (!attrs) {
      LOG(ERROR) << "Invalid agent layer, node " << NodeSymbol(node.id).str()
                 << " is not agent node";
      continue;
    }

    fout << attrs->timestamp.count() << "," << attrs->position.x() << ","
         << attrs->position.y() << "," << attrs->position.z() << ","
         << attrs->world_R_body.w() << "," << attrs->world_R_body.x() << ","
         << attrs->world_R_body.y() << "," << attrs->world_R_body.z() << "\n";
  }

  fout.flush();
  fout.close();
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const MultiBackendModuleStatus& status) {
  os << "new_loop_closures: " << status.new_loop_closures << ", "
     << "total_loop_closures: " << status.total_loop_closures << ", "
     << "total_interrobot_loop_closures: " << status.total_interrobot_loop_closures
     << ", " << "inlier_loop_closures: " << status.inlier_loop_closures << ", "
     << "inlier_interrobot_loop_closures: " << status.inlier_interrobot_loop_closures
     << ", " << "total_factors: " << status.total_factors << ", "
     << "total_values: " << status.total_values << ", "
     << "new_factors: " << status.new_factors << ", "
     << "new_graph_factors: " << status.new_graph_factors << ", "
     << "trajectory_len: " << status.trajectory_len << ", "
     << "num_merges_undone: " << status.num_merges_undone << ", " << "spin_s: "
     << (status.spin_s ? std::to_string(status.spin_s.value()) + " [s]" : "n/a") << ", "
     << "opt_s: "
     << (status.opt_s ? std::to_string(status.opt_s.value()) + " [s]" : "n/a") << ", "
     << "mesh_s: "
     << (status.mesh_s ? std::to_string(status.mesh_s.value()) + " [s]" : "n/a");
  return os;
}

void MultiBackendModuleStatus::reset() { *this = MultiBackendModuleStatus(); }

void declare_config(MultiBackendModule::GpsConfig& config) {
  using namespace config;
  name("MultiBackendModule::GpsConfig");
  field(config.enable, "enable");
  field(config.origin_utm, "origin_utm");
  field(config.min_sigma_m, "min_sigma_m");
  field(config.sigma_scale, "sigma_scale");
  field(config.use_altitude, "use_altitude");
  field(config.altitude_sigma_m, "altitude_sigma_m");
  field(config.max_time_diff_s, "max_time_diff_s");
  field(config.known_inlier_max_sigma_m, "known_inlier_max_sigma_m");
  field(config.min_alignment_fixes, "min_alignment_fixes");
  field(config.alignment_inlier_threshold_m, "alignment_inlier_threshold_m");
  field(config.align_yaw_only, "align_yaw_only");
  field(config.gravity_sigma, "gravity_sigma");
  field(config.min_alignment_loop_closures, "min_alignment_loop_closures");
  field(config.loop_closure_gate_m, "loop_closure_gate_m");
  field(config.coarse_gate_m, "coarse_gate_m");

  checkCondition(config.origin_utm.empty() || config.origin_utm.size() == 3,
                 "origin_utm must be empty or have 3 entries");
  check(config.min_sigma_m, GT, 0.0, "min_sigma_m");
  check(config.sigma_scale, GT, 0.0, "sigma_scale");
  check(config.altitude_sigma_m, GT, 0.0, "altitude_sigma_m");
  check(config.max_time_diff_s, GT, 0.0, "max_time_diff_s");
  check(config.alignment_inlier_threshold_m, GT, 0.0, "alignment_inlier_threshold_m");
}

void declare_config(MultiBackendModule::Config& config) {
  using namespace config;
  base<DsgUpdater::Config>(config);
  name("MultiBackendModule::Config");
  field(config.pgmo, "pgmo");
  field(config.optimize_on_lc, "optimize_on_lc");
  field(config.anchor_robot_id, "anchor_robot_id");
  field(config.include_mesh_factors, "include_mesh_factors");
  field(config.external_loop_closures, "external_loop_closures");
  field(config.sg_loop_closure_variance, "sg_loop_closure_variance");
  config.initial_align.setOptional();
  field(config.initial_align, "initial_align");
  config.partial_optimizer.setOptional();
  field(config.partial_optimizer, "partial_optimizer");
  field(config.gps, "gps");
}

MultiBackendModule::MultiBackendModule(
    const Config& config, const InterfaceContainer::StatesPtr& shared_states)
    : KimeraPgmoInterface(config.pgmo),
      config(config::checkValid(config)),
      lc_receiver_(config.external_loop_closures, &lc_queue_),
      unmerged_dsg_(MultiGlobalInfo::instance().createMultiDsg()),
      merged_dsg_(MultiGlobalInfo::instance().createMultiDsg()),
      shared_states_(shared_states),
      initial_align_(config.initial_align.create()),
      dsg_updater_(new hydra::DsgUpdater(config, unmerged_dsg_->graph, merged_dsg_)),
      partial_pgo_(config.partial_optimizer.create()) {
  using EdgeType = pose_graph_tools::PoseGraphEdge::Type;
  edge_variance_map_[EdgeType::ODOM] = KimeraPgmoInterface::config_.odom_variance;
  edge_variance_map_[EdgeType::LOOPCLOSE] = KimeraPgmoInterface::config_.lc_variance;
  edge_variance_map_[EdgeType::MESH] = KimeraPgmoInterface::config_.mesh_edge_variance;
  edge_variance_map_[EdgeType::POSE_MESH] =
      KimeraPgmoInterface::config_.pose_mesh_variance;
  edge_variance_map_[EdgeType::MESH_POSE] =
      KimeraPgmoInterface::config_.pose_mesh_variance;
  edge_variance_map_[EdgeType::PRIOR] = KimeraPgmoInterface::config_.prior_variance;

  // make sure there's always a valid status measurement in the log
  status_.reset();
  status_log_.push_back(status_);
  if (!config.include_mesh_factors) {
    LOG(WARNING) << "Mesh factors are OFF! Mesh will not be updated!";
  }

  if (config.gps.enable) {
    if (config.gps.origin_utm.size() == 3) {
      gps_origin_utm_ = Eigen::Vector3d(
          config.gps.origin_utm[0], config.gps.origin_utm[1], config.gps.origin_utm[2]);
    }

    if (config.anchor_robot_id >= 0) {
      LOG(WARNING) << "GPS priors are enabled but robot " << config.anchor_robot_id
                   << " is anchored at the origin! Set anchor_robot_id to -1";
    }

    if (initial_align_) {
      LOG(WARNING) << "GPS priors are enabled with a loop-closure initial alignment, "
                      "which re-anchors robots to the reference robot frame";
    }
  }
}

MultiBackendModule::~MultiBackendModule() { stopImpl(); }

void MultiBackendModule::start() {
  spin_thread_.reset(new std::thread(&MultiBackendModule::spin, this));
  optimization_thread_.reset(new std::thread(&MultiBackendModule::optimizeSpin, this));
  LOG(INFO) << " [Hydra-Multi Backend] started!";
}

void MultiBackendModule::stop() { stopImpl(); }

void MultiBackendModule::save(const DataDirectory& output) {
  const auto backend_path = output.path("backend");
  const auto pgmo_path = output.path("backend/pgmo");
  merged_dsg_->graph->save(backend_path / "dsg.json", false);
  merged_dsg_->graph->save(backend_path / "dsg_with_mesh.json");

  const auto& graph = *merged_dsg_->graph;
  const auto desired_layer = graph.getLayerKey(DsgLayers::AGENTS)->layer;
  for (const auto& layer : graph.layer_partition(desired_layer)) {
    const auto robot_id = kimera_pgmo::robot_prefix_to_id.at(layer.id.partition);
    std::string filename = "robot_" + std::to_string(robot_id) + "_trajectory.csv";
    const std::filesystem::path trajectory_path = backend_path / filename;
    ::hydra_multi::saveTrajectory(layer, trajectory_path);
  }

  const auto mesh = merged_dsg_->graph->mesh();
  if (mesh && !mesh->empty()) {
    // mesh implements vertex and face traits
    kimera_pgmo::WriteMesh(backend_path / "mesh.ply", *mesh, *mesh);
  }

  // node id bookkeeping so per-robot data keyed by node id (e.g. image stores) can be
  // re-keyed into the fused graph: robot -> [original id, fused id], fused merges
  {
    std::ofstream map_file(backend_path / "fusion_node_map.json");
    map_file << "{\"robots\": {";
    bool first_robot = true;
    for (const auto& [robot_id, node_map] : unmerged_dsg_->robot_node_map) {
      map_file << (first_robot ? "" : ", ") << "\"" << robot_id << "\": [";
      first_robot = false;
      std::map<NodeId, NodeId> sorted(node_map.begin(), node_map.end());
      bool first = true;
      for (const auto& [original, fused] : sorted) {
        map_file << (first ? "" : ", ") << "[" << original << ", " << fused << "]";
        first = false;
      }
      map_file << "]";
    }

    map_file << "}, \"merges\": [";
    bool first = true;
    for (const auto& [child, parent] : merged_dsg_->merges) {
      map_file << (first ? "" : ", ") << "[" << child << ", " << parent << "]";
      first = false;
    }
    map_file << "]}\n";
  }

  if (gps_origin_utm_) {
    int zone = 0;
    for (const auto& [id, state] : *shared_states_) {
      zone = zone ? zone : state->gps_utm_zone;
    }

    std::ofstream origin_file(backend_path / "gps_origin_utm.yaml");
    origin_file << std::setprecision(12) << "utm_zone: " << zone << "\n"
                << "easting: " << gps_origin_utm_->x() << "\n"
                << "northing: " << gps_origin_utm_->y() << "\n"
                << "altitude: " << gps_origin_utm_->z() << "\n";
  }

  writeBackendStatus(status_log_, pgmo_path / "dsg_pgmo_status.csv");
  deformation_graph_->save(pgmo_path / "deformation_graph.json");

  const std::string output_csv = backend_path / "loop_closures.csv";
  std::ofstream output_file;
  output_file.open(output_csv);

  output_file << "time_from_ns,time_to_ns,x,y,z,qw,qx,qy,qz,type,level" << std::endl;
  for (const auto& loop_closure : loop_closures_) {
    // pose = src.between(dest) or to_T_from
    auto time_from = hydra::utils::getTimeNs(*merged_dsg_->graph, loop_closure.dest);
    auto time_to = hydra::utils::getTimeNs(*merged_dsg_->graph, loop_closure.src);
    if (!time_from || !time_to) {
      continue;
    }

    const gtsam::Point3 pos = loop_closure.src_T_dest.translation();
    const gtsam::Quaternion quat = loop_closure.src_T_dest.rotation().toQuaternion();

    output_file << *time_from << "," << *time_to << ",";
    output_file << pos.x() << "," << pos.y() << "," << pos.z() << ",";
    output_file << quat.w() << ", " << quat.x() << "," << quat.y() << "," << quat.z()
                << ",";
    output_file << (loop_closure.dsg ? 1 : 0) << "," << loop_closure.level;
    output_file << std::endl;
  }
}

std::string MultiBackendModule::printInfo() const {
  std::stringstream ss;
  ss << config::toString(config_);
  return ss.str();
}

void MultiBackendModule::addSink(const Sink::Ptr& sink) {
  if (sink) {
    sinks_.push_back(sink);
  }
}

void MultiBackendModule::addLoopClosures(const pose_graph_tools::PoseGraph& info) {
  lc_queue_.push(info);
}

bool MultiBackendModule::optimizeOnce(Timestamp stamp) {
  const bool new_loop_closures = processLoopClosures(stamp);
  if (new_loop_closures || num_gps_priors_ > 0) {
    optimize(stamp);
  }

  return true;
}

void MultiBackendModule::spin() {
  bool should_shutdown = false;
  while (!should_shutdown) {
    Timestamp stamp;
    bool has_update = pollSharedStates(stamp);
    if (MultiGlobalInfo::instance().force_shutdown() || !has_update) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }

    if (!has_update) {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(10ms);
      continue;
    }

    spinOnce(stamp);
  }
}

void MultiBackendModule::optimizeSpin() {
  bool should_shutdown = false;
  while (!should_shutdown) {
    Timestamp stamp;
    pollSharedStates(stamp, false);

    bool should_optimize = processLoopClosures(stamp) && config.optimize_on_lc;
    if (!should_optimize || MultiGlobalInfo::instance().force_shutdown()) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }

    if (should_optimize && !should_shutdown) {
      optimize(stamp);
    } else {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(10ms);
    }
  }
}

void MultiBackendModule::stopImpl() {
  VLOG(1) << "Backend stopping...";
  should_shutdown_ = true;

  if (optimization_thread_) {
    VLOG(3) << "[Hydra-Multi Backend] joining optimizer thread and stopping";
    optimization_thread_->join();
    optimization_thread_.reset();
    VLOG(3) << "[Hydra-Multi Backend] optimizer thread stopped!";
  }

  if (spin_thread_) {
    VLOG(3) << "[Hydra-Multi Backend] joining spin thread and stopping";
    spin_thread_->join();
    spin_thread_.reset();
    VLOG(3) << "[Hydra-Multi Backend] spin stopped!";
  }
}

bool MultiBackendModule::pollSharedStates(Timestamp& stamp, bool mark_updated) const {
  bool updated = false;
  stamp = 0;
  for (const auto& [robot_id, state] : *shared_states_) {
    std::lock_guard<std::mutex> graph_lock(state->mutex);
    if (state->updated) {
      updated = true;
      if (mark_updated) {
        state->updated = false;  // Toggle back to false
      }

      if (state->stamp > stamp) {
        stamp = state->stamp;
      }
    }
  }

  return updated;
}

bool MultiBackendModule::processLoopClosures(Timestamp timestamp_ns) {
  ScopedTimer timer("backend/process_external_lcs", timestamp_ns);

  size_t processed = 0;
  std::lock_guard<std::mutex> unmerged_lock(unmerged_dsg_->mutex);
  lc_receiver_.update(
      *unmerged_dsg_->graph,
      [this, &processed](
          NodeId to_node, NodeId from_node, const gtsam::Pose3 to_T_from) {
        LoopClosureLog lc{to_node, from_node, to_T_from, true, 1, false};
        loop_closures_.push_back(lc);

        // TODO(Yun) currently interrobot non-external lcs not added to initial_align
        const auto to_prefix = NodeSymbol(to_node).category();
        const auto from_prefix = NodeSymbol(from_node).category();
        if (initial_align_ && to_prefix != from_prefix) {
          interrobot_lc_queue_.push(lc);
        }

        ++num_loop_closures_;
        ++processed;
      });

  have_loopclosures_ = (loop_closures_.size() > 0);
  return processed > 0;
}

void MultiBackendModule::partialOptimization(const Timestamp& timestamp_ns) {
  gtsam::Values optimized;
  {  // critical section for values
    std::lock_guard<std::mutex> lock(optimized_.mutex);
    optimized = optimized_.values;
  }

  if (optimized.empty()) {
    return;
  }

  if (!partial_pgo_) {
    deformation_graph_->updateValues(optimized);
    return;
  }

  ScopedTimer timer("backend/partial_optimization", timestamp_ns);
  auto initial = deformation_graph_->getValuesCopy();

  bool has_all_keys = true;
  for (const auto& key : initial.keys()) {
    if (!optimized.exists(key)) {
      has_all_keys = false;
      break;
    }
  }

  if (has_all_keys) {
    // every node already has an optimized value (the graph was rebuilt from the states)
    VLOG(1) << "No work to do in partial optimization!";
    deformation_graph_->updateValues(optimized);
    return;
  }

  auto factors = deformation_graph_->getFactorsCopy();
  auto temp_factors = deformation_graph_->getTempFactorsCopy();
  auto temp_initial = deformation_graph_->getTempValuesCopy();

  // TODO(Yun): remove hard coded value?
  static const gtsam::SharedNoiseModel& prior_noise =
      gtsam::noiseModel::Isotropic::Variance(6, 1.0e-3);

  for (const auto& key : optimized.keys()) {
    auto value = optimized.at<gtsam::Pose3>(key);
    factors.add(gtsam::PriorFactor<gtsam::Pose3>(key, value, prior_noise));
    initial.update(key, value);
  }

  // Set all factors as inliers
  auto known_indices = std::views::iota(size_t(0), factors.size());
  std::set<size_t> known_inliers(known_indices.begin(), known_indices.end());
  auto temp_known_indices = std::views::iota(size_t(0), temp_factors.size());
  std::set<size_t> temp_known_inliers(temp_known_indices.begin(),
                                      temp_known_indices.end());
  LOG(INFO) << "Starting partial optimization";
  try {
    partial_pgo_->update(factors,
                         initial,
                         known_inliers,
                         temp_factors,
                         temp_initial,
                         temp_known_inliers);
    LOG(INFO) << "Finished partial optimization";
  } catch (const gtsam::IndeterminantLinearSystemException& e) {
    LOG(ERROR) << "Failed optimization: '" << gtsam::Symbol(e.nearbyVariable());
    std::vector<gtsam::NonlinearFactorGraph::sharedFactor> nearby_factors;
    for (const auto& factor : factors) {
      for (const auto& key : factor->keys()) {
        if (key == e.nearbyVariable()) {
          nearby_factors.push_back(factor);
          break;
        }
      }
    }

    for (size_t i = 0; i < nearby_factors.size(); ++i) {
      const auto& factor = nearby_factors[i];
      std::stringstream ss;
      ss << "[";
      auto iter = factor->keys().begin();
      while (iter != factor->keys().end()) {
        ss << gtsam::Symbol(*iter).string();
        ++iter;
        if (iter != factor->keys().end()) {
          ss << ", ";
        }
      }
      ss << "]";
      LOG(ERROR) << "factor " << i << ": " << ss.str();
    }
  }

  auto estimates = partial_pgo_->getEstimates();
  auto temp_estimates = partial_pgo_->getTempEstimates();
  // Only update values
  deformation_graph_->updateValues(estimates);
  deformation_graph_->updateTempValues(temp_estimates);
}

bool MultiBackendModule::spinOnce(Timestamp timestamp_ns) {
  ScopedTimer timer("backend/spin", timestamp_ns);
  {
    auto lock = deformation_graph_->acquireLock();
    std::lock_guard<std::mutex> graph_lock(unmerged_dsg_->mutex);

    VLOG(2) << "Checking for updates @ " << timestamp_ns << " [ns]";
    if (!updateFromStates(timestamp_ns)) {
      VLOG(2) << "Waiting for states...";
      return true;
    }

    VLOG(1) << "Spinning once for update @ " << timestamp_ns << " [ns]";
    partialOptimization(timestamp_ns);

    combineAndDeformMesh(timestamp_ns, should_deform_);

    hydra::UpdateInfo::ConstPtr info(
        new hydra::UpdateInfo{timestamp_ns,
                              nullptr,
                              deformation_graph_->getValues(),
                              should_deform_,
                              {},
                              deformation_graph_.get(),
                              &unmerged_dsg_->node_robot_map});
    dsg_updater_->callUpdateFunctions(timestamp_ns, info);

    // should_deform_ = false;
    rebaseStates(timestamp_ns);
  }

  VLOG(2) << "Calling backend sinks @ " << timestamp_ns << " [ns]";
  ScopedTimer sink_timer("backend/sinks", timestamp_ns);
  MultiBackendModuleStatus status;
  {  // status critical section
    std::lock_guard<std::mutex> lock(status_mutex_);
    status = status_log_.back();
  }  // status critical section

  Sink::callAll(sinks_, timestamp_ns, *merged_dsg_->graph, *deformation_graph_, status);
  return true;
}

void MultiBackendModule::addLoopClosures() {
  // Add Loop Closures (only those not yet added to deformation graph)
  for (auto& lc : loop_closures_) {
    if (lc.added_to_dgraph) {
      continue;
    }
    deformation_graph_->processNewBetween(
        lc.src,
        lc.dest,
        lc.src_T_dest,
        (lc.level ? KimeraPgmoInterface::config_.lc_variance
                  : config.sg_loop_closure_variance));
    lc.added_to_dgraph = true;
  }
}

void MultiBackendModule::optimizeStartup() {
  if (!config.gps.enable) {
    addLoopClosures();
    initialAlignment();
    return;
  }

  // align robots without GPS and drop loop closures that contradict the GPS frame before
  // they reach the factor graph
  alignRobotsWithLoopClosures();
  gateLoopClosures();
  addLoopClosures();
}

void MultiBackendModule::optimizeCleanup(const gtsam::Values& results) {
  std::lock_guard<std::mutex> lock(optimized_.mutex);
  optimized_.values = gtsam::Values(results);
}

void MultiBackendModule::optimize(size_t timestamp_ns) {
  ScopedTimer timer("backend/optimize", timestamp_ns);
  LOG(INFO) << "Starting full optimization";
  kimera_pgmo::OptimizeStats stats;
  {  // timer scope
    ScopedTimer timer("backend/optimization", timestamp_ns, true, 0, false);
    stats = KimeraPgmoInterface::optimize();
  }  // timer scope
  LOG(INFO) << "Finish full optimization";

  {  // status scope
    std::lock_guard<std::mutex> lock(status_mutex_);
    status_.total_loop_closures = stats.total_loop_closures;
    status_.total_interrobot_loop_closures = stats.total_interrobot_loop_closures;
    status_.inlier_loop_closures = stats.inlier_loop_closures;
    status_.inlier_interrobot_loop_closures = stats.inlier_interrobot_loop_closures;
    status_.total_factors = stats.total_factors;
    status_.total_values = stats.total_values;
    status_.opt_s = stats.elapsed_s;
  }

  logStatus();
  should_deform_ = true;
}

gtsam::Pose3 MultiBackendModule::getRobotWorldFrame(RobotId robot_id,
                                                    gtsam::Values& optimized) {
  const char robot_prefix = kimera_pgmo::robot_id_to_prefix.at(robot_id);
  if (!deformation_graph_->hasPrefixPoses(robot_prefix)) {
    return gtsam::Pose3();
  }

  auto trajectory = deformation_graph_->getTrajectory(robot_prefix);
  auto last_key = trajectory.size() - 1;

  auto Wglobal_T_pose = trajectory.back();

  // TODO(nathan) think about this more?
  // Try to use instead the optimized value
  auto it = optimized.upper_bound(gtsam::Symbol(robot_prefix, last_key));
  if (it != optimized.begin()) {
    --it;
    if (gtsam::Symbol(it->key).chr() == robot_prefix) {
      last_key = gtsam::Symbol(it->key).index();
      Wglobal_T_pose = optimized.at<gtsam::Pose3>(it->key);
    }
  }

  const auto curr_W_T_pose = deformation_graph_->getInitialPose(robot_prefix, last_key);
  const auto Wrobot_T_pose = curr_W_T_robot_.at(robot_id).between(curr_W_T_pose);
  const auto Wglobal_T_Wrobot = Wglobal_T_pose.compose(Wrobot_T_pose.inverse());
  // NOTE(Yun): Weirdly numerical pile up and quickly diverges if we do not keep
  // projecting back...
  return projectToClosestSE3(Wglobal_T_Wrobot.matrix());
}

void MultiBackendModule::rebaseStates(size_t timestamp_ns) {
  ScopedTimer timer("backend/rebase_states", timestamp_ns);
  // TODO(Yun): Account for how to propagate merged nodes or map to robot / unit
  // interface.
  gtsam::Values optimized;
  {
    std::lock_guard<std::mutex> lock(optimized_.mutex);
    optimized = optimized_.values;
  }

  for (const auto& [id, state] : *shared_states_) {
    std::lock_guard<std::mutex> state_lock(state->mutex);
    state->world_T_robot = getRobotWorldFrame(id, optimized);
  }
}

bool MultiBackendModule::updateFromStates(Timestamp timestamp_ns) {
  ScopedTimer timer("backend/update_from_states", timestamp_ns);
  bool updates = false;
  unmerged_dsg_->clear();

  // Accumulate factor graphs from each robot into one
  deformation_graph_->clear();
  num_gps_priors_ = 0;

  // Reset loop closure flags since the deformation graph was cleared
  // They will be re-added by addLoopClosures() at the end of updateFromStates
  for (auto& lc : loop_closures_) {
    lc.added_to_dgraph = false;
  }

  for (const auto& [id, state] : *shared_states_) {
    VLOG(3) << "Saw robot " << id << " in update @ " << timestamp_ns << " [ns]";
    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (config.gps.enable) {
      alignRobotToGps(id, *state);
    }

    // Append transform
    Eigen::Isometry3d transform(state->world_T_robot.matrix());
    curr_W_T_robot_[id] = gtsam::Pose3(transform.matrix());
    // Unfortunately, "transforming" the deformation graph when adding makes initial
    // alignment a bit more complicated. Hence requiring us to store the transform to
    // account for it in initial alignment.

    //// Process pose graph and mesh graph
    // Check and construct remapping ids then process pose graph and mesh graph
    // Implicit assumption that each state here only have a single id
    if (state->pose_graph_->nodes.size() > 0) {
      std::map<size_t, size_t> id_mapping;
      id_mapping[state->pose_graph_->nodes[0].robot_id] = id;
      VLOG(3) << "Remapping " << state->pose_graph_->nodes[0].robot_id << " -> " << id
              << " for pg with " << state->pose_graph_->nodes.size() << " nodes";
      deformation_graph_->processPoseGraph(
          *state->pose_graph_, edge_variance_map_, id_mapping, &transform);
      if (config.anchor_robot_id >= 0 &&
          id == static_cast<size_t>(config.anchor_robot_id)) {
        auto prefix = kimera_pgmo::GetRobotPrefix(id);
        deformation_graph_->addNewNode(gtsam::Symbol(prefix, 0),
                                       0,
                                       gtsam::Pose3(),
                                       true,
                                       config.pgmo.prior_variance);
      }

      if (config.gps.enable) {
        num_gps_priors_ += addGpsPriors(id, *state);
      }
    }

    if (state->mesh_graph_->nodes.size() > 0 && config.include_mesh_factors) {
      std::map<size_t, size_t> id_mapping;
      id_mapping[state->mesh_graph_->nodes[0].robot_id] = id;
      deformation_graph_->processMeshGraph(
          *state->mesh_graph_, edge_variance_map_, id_mapping, &transform);
    }

    if (!state->dsg_) {
      continue;
    }

    // TODO(nathan) this could be handled better
    updates = true;
    unmerged_dsg_->addRobotGraph(id, *state->dsg_, &transform);

    if (!state->mesh_data_ || !state->mesh_data_->mesh ||
        state->mesh_data_->mesh->empty()) {
      continue;
    }

    staged_mesh_data_[id] = state->mesh_data_->clone();
    staged_mesh_data_[id]->transform(transform);
  }

  // Add objects or places to deformation graph
  deformation_graph_->clearTemporaryStructures();

  // Add loop closures so they appear in the published pose graph
  addLoopClosures();

  return updates;
}

void MultiBackendModule::combineAndDeformMesh(Timestamp timestamp_ns,
                                              bool force_mesh_update) {
  ScopedTimer timer("backend/deform_mesh", timestamp_ns);

  Mesh::Ptr combined_mesh;
  for (const auto& [id, mesh_data] : staged_mesh_data_) {
    if (!mesh_data) {
      LOG(WARNING) << "No mesh for robot " << id;
      continue;
    }

    if (have_loopclosures_ || num_gps_priors_ > 0 || force_mesh_update) {
      VLOG(3) << "Deforming mesh for robot " << id;
      // TODO(Yun): cannot use pre archived vertices as 'start_hint'
      // Note that here the mesh in the state is actually getting modified!
      char vertex_prefix = kimera_pgmo::robot_id_to_vertex_prefix.at(id);
      deformation_graph_->deformPoints(*mesh_data->mesh,
                                       mesh_data->original,
                                       vertex_prefix,
                                       KimeraPgmoInterface::config_.num_interp_pts,
                                       KimeraPgmoInterface::config_.interp_horizon);
    }

    VLOG(5) << "Copying mesh to combined mesh for robot " << id;
    if (!combined_mesh) {
      combined_mesh = mesh_data->mesh->clone();
      unmerged_dsg_->robot_vertex_offset[id] = 0;
    } else {
      unmerged_dsg_->robot_vertex_offset[id] = combined_mesh->numVertices();
      *combined_mesh += *mesh_data->mesh;
    }

    unmerged_dsg_->robot_num_vertices[id] = mesh_data->mesh->numVertices();
  }

  if (combined_mesh) {
    unmerged_dsg_->graph->setMesh(combined_mesh);
    reindexMeshConnections(*unmerged_dsg_->graph,
                           unmerged_dsg_->node_robot_map,
                           unmerged_dsg_->robot_vertex_offset);
  } else {
    VLOG(3) << "No mesh for any robot!";
  }

  // TODO(nathan) fix this!!!!!!!!!!!!!!!!
  // Merged DSG is actually reset twice here (but resetBackendDsg necessary to reset
  // MergeTracker)
  dsg_updater_->resetBackendDsg(timestamp_ns);
  std::lock_guard<std::mutex> merged_graph_lock(merged_dsg_->mutex);
  merged_dsg_->update(*unmerged_dsg_);
}

void MultiBackendModule::initialAlignment() {
  if (!initial_align_) {
    LOG(WARNING) << "Initial alignment not set.";
    return;
  }

  // TODO(nathan) figure out how to handle inheritance
  auto info = std::make_shared<LoopClosureInitialAlignModule::Info>();
  // W_T_robot only updated during initial
  info->prev_W_T_robot = curr_W_T_robot_;
  info->dgraph = deformation_graph_;
  while (!interrobot_lc_queue_.empty()) {
    const auto lc = interrobot_lc_queue_.pop();
    info->loop_closures.push_back({lc.src, lc.dest, lc.src_T_dest});
  }

  VLOG(10) << "Running initial alignment";
  initial_align_->update(info);

  // TODO(Yun): temp values not aligned. Might be fine for now
  auto pgmo_values = deformation_graph_->getValuesCopy();
  const auto initial_values = initial_align_->computeInitialGuess(
      info, pgmo_values, merged_dsg_->node_robot_map);
  deformation_graph_->updateValues(initial_values);
}

namespace {

std::vector<pose_graph_tools::PoseGraphNode> sortedNodes(
    const pose_graph_tools::PoseGraph& pose_graph) {
  auto nodes = pose_graph.nodes;
  std::sort(nodes.begin(), nodes.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.stamp_ns < rhs.stamp_ns;
  });
  return nodes;
}

}  // namespace

void MultiBackendModule::alignRobotToGps(RobotId robot_id, UnitInterfaceState& state) {
  if (gps_aligned_robots_.count(robot_id) || state.gps_measurements.empty() ||
      !state.pose_graph_ || state.pose_graph_->nodes.empty()) {
    return;
  }

  if (!gps_origin_utm_) {
    gps_origin_utm_ = state.gps_measurements.front().utm;
    LOG(INFO) << "GPS origin set to first fix of robot " << robot_id << ": "
              << std::setprecision(12) << gps_origin_utm_->transpose();
  }

  const auto nodes = sortedNodes(*state.pose_graph_);
  const auto max_gap_ns = static_cast<Timestamp>(config.gps.max_time_diff_s * 1e9);
  std::vector<Eigen::Vector3d> robot_points;
  std::vector<Eigen::Vector3d> world_points;
  for (const auto& meas : state.gps_measurements) {
    const auto result = interpolatePose(nodes, meas.stamp_ns, max_gap_ns);
    if (!result) {
      continue;
    }

    robot_points.push_back(result->second.transformFrom(state.body_t_gps));
    world_points.push_back(meas.utm - *gps_origin_utm_);
  }

  if (robot_points.size() < config.gps.min_alignment_fixes) {
    LOG(WARNING) << "Robot " << robot_id << " only has " << robot_points.size()
                 << " GPS fixes overlapping its trajectory; not aligning to GPS";
    return;
  }

  const auto alignment = alignPoints(robot_points,
                                     world_points,
                                     config.gps.alignment_inlier_threshold_m,
                                     200,
                                     config.gps.align_yaw_only);
  if (!alignment) {
    LOG(WARNING) << "Failed to align robot " << robot_id << " to GPS";
    return;
  }

  LOG(INFO) << "Aligned robot " << robot_id << " to GPS with "
            << alignment->num_inliers << " / " << robot_points.size()
            << " inliers (rmse: " << alignment->rmse << " m)";
  state.world_T_robot = alignment->world_T_robot;
  gps_aligned_robots_.insert(robot_id);
}

size_t MultiBackendModule::addGpsPriors(RobotId robot_id,
                                        const UnitInterfaceState& state) {
  if (!gps_origin_utm_ || state.gps_measurements.empty() || !state.pose_graph_) {
    return 0;
  }

  const auto& gps = config.gps;
  const auto prefix = kimera_pgmo::GetRobotPrefix(robot_id);
  const auto nodes = sortedNodes(*state.pose_graph_);
  const auto max_gap_ns = static_cast<Timestamp>(gps.max_time_diff_s * 1e9);
  kimera_pgmo::DeformationGraph::PositionMeasurements measurements;
  for (const auto& meas : state.gps_measurements) {
    const auto result = interpolatePose(nodes, meas.stamp_ns, max_gap_ns);
    if (!result) {
      continue;
    }

    // express the antenna at the fix time relative to the closest pose node
    const auto& node = nodes[result->first];
    const gtsam::Pose3 robot_T_node(node.pose.matrix());
    const auto node_T_fix = robot_T_node.between(result->second);

    const Eigen::Vector3d sigma = meas.variance.cwiseMax(0.0).cwiseSqrt();
    const double horizontal_sigma = std::max(sigma.x(), sigma.y());
    Eigen::Vector3d variance;
    for (int i = 0; i < 3; ++i) {
      const double s = std::max(sigma(i) * gps.sigma_scale, gps.min_sigma_m);
      variance(i) = s * s;
    }

    if (!gps.use_altitude) {
      variance.z() = gps.altitude_sigma_m * gps.altitude_sigma_m;
    }

    auto& prior = measurements.emplace_back();
    prior.key = gtsam::Symbol(prefix, node.key);
    prior.world_t_sensor = meas.utm - *gps_origin_utm_;
    prior.covariance = variance.asDiagonal();
    prior.body_t_sensor = node_T_fix.transformFrom(state.body_t_gps);
    prior.known_inlier =
        gps.known_inlier_max_sigma_m > 0.0 && horizontal_sigma <= gps.known_inlier_max_sigma_m;
  }

  const auto num_added = deformation_graph_->processPositionMeasurements(measurements);
  VLOG(1) << "Added " << num_added << " GPS priors for robot " << robot_id;
  if (num_added && gps.gravity_sigma > 0.0) {
    std::vector<std::pair<gtsam::Key, gtsam::Point3>> body_ups;
    for (const auto& node : nodes) {
      const gtsam::Rot3 robot_R_node(node.pose.linear());
      body_ups.emplace_back(gtsam::Symbol(prefix, node.key),
                            robot_R_node.unrotate(gtsam::Point3::UnitZ()));
    }

    deformation_graph_->processUpVectorPriors(body_ups, gps.gravity_sigma);
  }

  return num_added;
}

void MultiBackendModule::alignRobotsWithLoopClosures() {
  const auto min_lcs = config.gps.min_alignment_loop_closures;
  if (!min_lcs || gps_aligned_robots_.empty()) {
    return;
  }

  // For each loop closure between a robot without a GPS frame and one with it: the pose
  // the unaligned node should have (from the aligned side and the loop closure) and its
  // current pose. Each one proposes a correction world_T_world' = target * current^-1.
  struct Constraint {
    gtsam::Pose3 target;
    gtsam::Pose3 current;
  };

  // Coarse GPS fixes (too noisy for priors) of each robot paired with the antenna position
  // in the current world frame; they reject alignments that are grossly wrong (e.g. a
  // self-consistent set of false loop closures from repetitive structure)
  std::map<RobotId, std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>> coarse;
  if (config.gps.coarse_gate_m > 0.0 && gps_origin_utm_) {
    const auto max_gap_ns = static_cast<Timestamp>(config.gps.max_time_diff_s * 1e9);
    for (const auto& [robot_id, state] : *shared_states_) {
      std::lock_guard<std::mutex> state_lock(state->mutex);
      if (gps_aligned_robots_.count(robot_id) || state->gps_coarse_measurements.empty() ||
          !state->pose_graph_) {
        continue;
      }

      const auto nodes = sortedNodes(*state->pose_graph_);
      const auto world_T_robot = curr_W_T_robot_.count(robot_id)
                                     ? curr_W_T_robot_.at(robot_id)
                                     : gtsam::Pose3();
      for (const auto& meas : state->gps_coarse_measurements) {
        const auto result = interpolatePose(nodes, meas.stamp_ns, max_gap_ns);
        if (!result) {
          continue;
        }

        const auto antenna = world_T_robot.transformFrom(
            result->second.transformFrom(state->body_t_gps));
        coarse[robot_id].emplace_back(antenna, meas.utm - *gps_origin_utm_);
      }
    }
  }

  const auto coarse_error = [&](RobotId robot_id, const gtsam::Pose3& correction) {
    const auto iter = coarse.find(robot_id);
    if (iter == coarse.end() || iter->second.size() < config.gps.min_alignment_fixes) {
      return -1.0;
    }

    std::vector<double> errors;
    for (const auto& [antenna, fix] : iter->second) {
      errors.push_back((correction.transformFrom(antenna) - fix).head<2>().norm());
    }

    std::nth_element(errors.begin(), errors.begin() + errors.size() / 2, errors.end());
    return errors[errors.size() / 2];
  };

  // align the best supported robot first, so robots only connected through other
  // unaligned robots can follow
  while (true) {
    const auto values = deformation_graph_->getValuesCopy();
    std::map<RobotId, std::vector<Constraint>> constraints;
    for (const auto& lc : loop_closures_) {
      const auto& prefixes = kimera_pgmo::robot_prefix_to_id;
      if (!prefixes.count(lc.src.chr()) || !prefixes.count(lc.dest.chr()) ||
          !values.exists(lc.src) || !values.exists(lc.dest)) {
        continue;
      }

      const RobotId src_robot = prefixes.at(lc.src.chr());
      const RobotId dest_robot = prefixes.at(lc.dest.chr());
      const bool src_aligned = gps_aligned_robots_.count(src_robot);
      const bool dest_aligned = gps_aligned_robots_.count(dest_robot);
      if (src_aligned == dest_aligned) {
        continue;
      }

      const auto world_T_src = values.at<gtsam::Pose3>(lc.src);
      const auto world_T_dest = values.at<gtsam::Pose3>(lc.dest);
      if (src_aligned) {
        constraints[dest_robot].push_back({world_T_src * lc.src_T_dest, world_T_dest});
      } else {
        constraints[src_robot].push_back(
            {world_T_dest * lc.src_T_dest.inverse(), world_T_src});
      }
    }

    // consensus per robot: the correction that brings the most loop closures within
    // tolerance (residuals are evaluated locally at each loop closure, so the lever arm
    // to the world origin does not inflate rotation differences)
    std::optional<RobotId> best_robot;
    gtsam::Pose3 best_correction;
    size_t best_inliers = 0;
    size_t best_total = 0;
    for (const auto& [robot_id, robot_constraints] : constraints) {
      for (const auto& candidate : robot_constraints) {
        const auto correction = candidate.target * candidate.current.inverse();
        size_t inliers = 0;
        for (const auto& c : robot_constraints) {
          const auto residual = c.target.between(correction * c.current);
          if (residual.translation().norm() < config.gps.alignment_inlier_threshold_m &&
              gtsam::Rot3::Logmap(residual.rotation()).norm() < 0.2) {
            ++inliers;
          }
        }

        if (inliers <= best_inliers) {
          continue;
        }

        const auto gps_error = coarse_error(robot_id, correction);
        if (gps_error > config.gps.coarse_gate_m) {
          continue;
        }

        best_robot = robot_id;
        best_correction = correction;
        best_inliers = inliers;
        best_total = robot_constraints.size();
      }
    }

    if (!best_robot || best_inliers < min_lcs) {
      if (best_robot) {
        LOG(WARNING) << "Robot " << *best_robot << ": only " << best_inliers << " / "
                     << best_total << " loop closures agree; not aligning";
      }
      return;
    }

    const auto robot_id = *best_robot;
    const auto agent_prefix = kimera_pgmo::robot_id_to_prefix.at(robot_id);
    const auto vertex_prefix = kimera_pgmo::robot_id_to_vertex_prefix.at(robot_id);
    gtsam::Values aligned;
    for (const auto& key : values.keys()) {
      const gtsam::Symbol symbol(key);
      if (symbol.chr() == agent_prefix || symbol.chr() == vertex_prefix) {
        aligned.insert(key, best_correction * values.at<gtsam::Pose3>(key));
      }
    }

    deformation_graph_->updateValues(aligned);
    curr_W_T_robot_[robot_id] = best_correction * curr_W_T_robot_[robot_id];
    gps_aligned_robots_.insert(robot_id);
    // the scene graph and mesh are re-transformed by the state frame on the next update
    const auto state = shared_states_->find(robot_id);
    if (state != shared_states_->end()) {
      std::lock_guard<std::mutex> state_lock(state->second->mutex);
      state->second->world_T_robot = best_correction * state->second->world_T_robot;
    }

    const auto gps_error = coarse_error(robot_id, best_correction);
    LOG(INFO) << "Aligned robot " << robot_id << " to the GPS frame with " << best_inliers
              << " / " << best_total << " loop closures"
              << (gps_error >= 0.0 ? " (coarse GPS median error " + std::to_string(gps_error) + " m)"
                                   : std::string())
              << " (correction: "
              << best_correction.translation().norm() << " m, "
              << gtsam::Rot3::Logmap(best_correction.rotation()).norm() << " rad)";
  }
}

void MultiBackendModule::gateLoopClosures() {
  const auto gate = config.gps.loop_closure_gate_m;
  if (gate <= 0.0) {
    return;
  }

  const auto values = deformation_graph_->getValuesCopy();
  const auto& prefixes = kimera_pgmo::robot_prefix_to_id;
  size_t num_rejected = 0;
  auto iter = loop_closures_.begin();
  while (iter != loop_closures_.end()) {
    const auto& lc = *iter;
    if (lc.added_to_dgraph || lc.src.chr() == lc.dest.chr() || !prefixes.count(lc.src.chr()) ||
        !prefixes.count(lc.dest.chr()) || !values.exists(lc.src) || !values.exists(lc.dest) ||
        !gps_aligned_robots_.count(prefixes.at(lc.src.chr())) ||
        !gps_aligned_robots_.count(prefixes.at(lc.dest.chr()))) {
      ++iter;
      continue;
    }

    const auto predicted = values.at<gtsam::Pose3>(lc.src) * lc.src_T_dest;
    const auto residual = predicted.between(values.at<gtsam::Pose3>(lc.dest));
    if (residual.translation().norm() > gate) {
      iter = loop_closures_.erase(iter);
      ++num_rejected;
      continue;
    }

    ++iter;
  }

  if (num_rejected) {
    LOG(INFO) << "Rejected " << num_rejected << " inter-robot loop closures that disagree "
              << "with the GPS frame by more than " << gate << " m";
  }
}

void MultiBackendModule::logStatus() {
  std::lock_guard<std::mutex> lock(status_mutex_);

  const auto& timer = hydra::timing::ElapsedTimeRecorder::instance();
  status_.spin_s = timer.getLastElapsed("backend/spin");
  status_.mesh_s = timer.getLastElapsed("backend/mesh_update");
  status_log_.push_back(status_);
  status_.reset();
}

}  // namespace hydra_multi
