# Codes to design in situ sampling strategies 

### This repository includes the codes generated to evaluate different in situ sampling strategies to reconstruct fine-scale (~20 km) ocean currents in the context of SWOT satellite mission (subtask 2.3.1 of the H2020 EuroSea project)
In **1_simulate_observations** you can find codes to design the sampling strategy and to extract the simulated observations. In **2_reconstruct_observations_spatiotemporal_OI** the spatio-temporal optimal interpolation is applied to simulated observations of temperature and salinity. The dynamic height and geostrophic velocity are calculated from the reconstructed fields. In **3_evaluation** you can find the codes to compare reconstructed fields with the ocean truth. Below there is a step-by-step description of all codes.

* _H2020 EuroSea project:_<br>
The H2020 EuroSea project aims at improving and integrating the European Ocean Observing and Forecasting System (see official website: [https://eurosea.eu/](https://eurosea.eu/)). It has received funding from the European Union’s Horizon 2020  research and innovation programme under grant agreement No 862626.

* _Task 2.3:_<br>
Task 2.3 has the objective to improve the design of multi-platform experiments aimed to validate the Surface Water and Ocean Topography (SWOT) satellite observations with the goal to optimize the utility of these observing platforms. Observing System Simulation Experiments (OSSEs) have been conducted to evaluate different configurations of the in situ observing system, including rosette and underway CTD, gliders, conventional satellite nadir altimetry and velocities from drifters. High-resolution models have been used to simulate the observations and to represent the “ocean truth”. Several methods of reconstruction have been tested: spatio-temporal optimal interpolation, machine-learning techniques, model data assimilation and the MIOST tool. Contributors to Task 2.3 are CSIC (Spain), CLS (France), SOCIB (Spain), IMT-Atlantique (France) and Ocean-Next (France). **The planned OSSEs are detailed in this public report [Barceló-Llull et al., 2020](https://doi.org/10.3289/eurosea_d2.1) and the complete analysis is available here [Barceló-Llull et al., 2022](https://doi.org/10.3289/eurosea_d2.3).**

* _Subtask 2.3.1:_<br>
Subtask 2.3.1 aims to **evaluate different in situ sampling strategies to reconstruct fine-scale ocean currents (~20 km) in the context of SWOT**. An advanced version of the classic optimal interpolation used in field experiments, which considers the spatial and temporal variability of the observations, has been applied to reconstruct different configurations with the objective to evaluate the best sampling strategy to validate SWOT. 

* _Where?_<br>
The analysis  focuses on two regions of interest: **(i) the western Mediterranean Sea and (ii) the subpolar northwest Atlantic**. In the western Mediterranean Sea, the target area is located within a swath of SWOT, while in the northwest Atlantic the region of study includes a crossover of  SWOT during the fast-sampling phase. 

## Report with the full analysis

The complete analysis can be found in this report: [Barceló-Llull et al., 2022](https://doi.org/10.3289/eurosea_d2.3).

## Dataset

The dataset generated in Subtask 2.3.1 is available here: Bàrbara Barceló-Llull, Ananda Pascual, Aurélie Albert, Jaime Hernández-Lasheras, Stephanie Leroux, & Baptiste Mourre. (2022). Dataset generated to evaluate in situ sampling strategies to reconstruct fine-scale ocean currents in the context of SWOT satellite mission (H2020 EuroSea project) (V1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6798018


## The codes

The codes are organized into 3 folders:
- 1_simulate_observations
- 2_reconstruct_observations_spatiotemporal_OI
- 3_evaluation

Here there is a step-by-step description with the corresponding codes. 

### 1_simulate_observations

*Objective: Simulate CTD observations from different platforms (rosette, underway CTD, gliders) and ADCP observations to evaluate different in situ sampling strategies*

(1) `Step00a_interpolate_eNATL60_4D_outputs.py` 

Interpolate eNATL60 model outputs onto a regular grid with a horizontal resolution of 1/60º for each time step and at each depth layer. Before starting the interpolation, we mask the 0 unreal values in the original data to exclude them from the interpolation. Then, we save the interpolated model data in a netcdf file for each variable (T, S, U and V), for each region (Atlantic or Mediterranean) and for each period (January or September). We use the interpolated data to extract the corresponding pseudo-observations in Step 2. NOTE: U and V are not zonal and meridional velocities. They are velocities along the original x and y axis. Then, if the objective is to extract ADCP velocity, they need to be rotated. 

(2) Define sampling strategies:

`Step01_define_sampling_strategy_CTD_ADCP.py`

Define sampling strategy with rosette CTD: get (time, lon, lat, dep) of each cast. Valid for reference configuration and configurations 1, 2 and 4. MedSea and Atlantic.

`Step01_define_sampling_strategy_gliders.py`

Define sampling strategy with gliders: get (time, lon, lat, dep) of each profile. Valid for configuration 5. MedSea and Atlantic.

`Step01_define_sampling_strategy_uCTD.py`

Define sampling strategy with underway CTD: get (time, lon, lat, dep) of each profile. Valid for configuration 3. MedSea and Atlantic.

(3) `Step02_extract_pseudo-obs_from_models.py`

Extract pseudo-observations of temperature, salinity and ADCP horizontal velocities.

(4) `Step03_plot_pseudo-obs.py`

Plot pseudo-observations.

(5) `Step31_simulate_pseudo-obs_error.py`

Simulate random error for CTD and ADCP pseudo-observations, following a Gaussian distribution with a standard deviation defined by the instrumental error. Different (uncorrelated) for each observations. Following Gasparin et al., 2019.

Toolbox: `EuroSea_toolbox.py`

### 2_reconstruct_observations_spatiotemporal_OI

*Objective: Reconstruct observations with the spatio-temporal optimal interpolation*

(1) `Step05b_spatio-temporal_optimal_interpolation_T_and_S.py`

Reconstruct pseudo-observations of T and S applying linear interpolation on the vertical and
the spatio-temporal optimal interpolation horizontally. For all configurations, in the Atlantic and Mediterranean, for all models.

For T and S: noise-to-signal error 3%, Lx = Ly = 20 km, Lt = 10 days.
Not done for U and V but the code is ready. 

Time of the resulting OI map: central date of each configuration.


(2) `Step07b_calculate_dh_vgeo_spatio-temporal_OI.py` 

From the reconstructed T and S 3D fields: compute DH, (ug, vg), and Rog. For all configurations, in the Atlantic and Mediterranean, for all models.
    
(3) `Step08b_figures_to_check_reconstructed_fields_spatio-temporal_OI.py` 

Plot figures to check reconstructed fields with the spatio-temporal OI.

Plot for each configuration: T (with observations) + S (with observations) + dh+vgeo

Toolboxes: `EuroSea_toolbox.py`, `deriv_tools.py`, `Tools_OI.py`

### 3_evaluation

*Objective: Compare reconstructed fields to the ocean truth and find the best sampling strategies*

(1) `Step00b_interpolate_eNATL60_2D_outputs.py`

Interpolate eNATL60 2D (surface) fields onto a regular grid.

(1) `Step09b_extract_reconstructed_data_for_comparison_spatio-temporal_OI.py`

Extract reconstructed data at the upper layer to do comparisons in Step 12. Save data with the same format as the ocean truth .pkl file.

(2) `Step09c_figures_reconstructed_fields_all_spatio-temporal_OI.py`

Reconstructed fields: Plot 1 figure for each region, model and variable, including all configurations. Variables: dh, geostrophic velocity magnitude and geostrophic Ro. For comparison between configurations and with the ocean truth. [Figures published in D2.3]

(3) `Step10b_extract_model_data_for_figures_spatio-temporal_OI.py` 

Extract 2D model data (ocean truth) within the configuration domain and linearly interpolate model fields to the time of the OI map. Save fields.

(4) `Step11b_figures_model_SSH_speed_Ro_spatio-temporal_OI.py` 

Ocean truth: Plot 1 figure for each region, model and variable, including all configurations. Variables: SSH, total horizontal velocity magnitude (speed) and Ro computed from total horizontal velocity. For comparison with reconstructed fields. [Figures published in D2.3]

(5) `Step10c_extract_model_data_for_comparison_spatio-temporal_OI_bigger_domain.py`

Extract 2D model data within a domain bigger than the corresponding configuration and linearly interpolate model fields to the time of the OI map. Save fields, to be used in Step12f.

(6) `Step12f_compute_statistics_spatio-temporal_OI_limit_domain.py`

This codes does:

1.	Open reconstructed and model fields (saved in a bigger domain than the corresponding configuration in Step10c).
2.	Interpolate model fields (ssh, ut, vt, speed, Ro) onto the reconstruction grid.
3.	Limit model and reconstructed data within the configuration 2a domain.
4.	Compute DH anomaly and SSH anomaly. Spatial average over configuration 2a domain. 
5.	Calculate RMSE-based score between reconstructed and model fields.                  https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60/blob/master/notebooks/example_data_eval.ipynb
6.	Save RMSE-based score in a .pkl file for each field.
7.	Plot table and figure of the RMSE-based score. (Only plot figures for DH and geostrophic velocity magnitude.)

(7) `Step13a_comparison_statistics_spatio-temporal_OI_best_configurations_ranking_all.py`

Leaderboard based on the RMSEs calculated in Step12f for the Mediterranean and Atlantic [D2.3].

Toolboxes: `EuroSea_toolbox.py`, `deriv_tools.py`

