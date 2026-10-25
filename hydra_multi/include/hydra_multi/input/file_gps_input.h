#pragma once
#include <filesystem>

#include "hydra_multi/input/input.h"

namespace hydra_multi {

class FileGpsInput : public Input {
 public:
  struct Config {
    //! CSV with header timestamp_ns,latitude,longitude,altitude,status,cov_xx,cov_yy,cov_zz
    std::filesystem::path gps_path;
    //! Antenna position in the robot body frame
    std::vector<double> body_t_gps{0.0, 0.0, 0.0};
    //! UTM zone to project into (0 = zone of the first fix)
    int utm_zone = 0;
    //! Minimum NavSatStatus status to keep a fix (0 = any fix)
    int min_status = 0;
    //! Discard fixes whose horizontal standard deviation exceeds this
    double max_sigma_m = 2.0;
    //! Fixes up to this horizontal standard deviation are kept as coarse fixes that only
    //! validate alignments (non-positive disables)
    double coarse_max_sigma_m = 15.0;
    //! Fixes too uncertain horizontally for either set are kept as altitude-only priors
    //! when their altitude standard deviation is at most this (non-positive disables)
    double altitude_max_sigma_m = 0.0;
  } const config;

  FileGpsInput(const Config& config,
               UnitInterfaceState::Ptr state,
               std::string name,
               size_t id);

  ~FileGpsInput() = default;

  void init() override;

  void stop() override;
};

void declare_config(FileGpsInput::Config& config);

}  // namespace hydra_multi
