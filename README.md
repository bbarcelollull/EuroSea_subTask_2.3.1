# H2020 EuroSea project - Task 2.3

### This repository includes the codes generated in subtask 2.3.1 of the EuroSea project

* _H2020 EuroSea project:_<br>
The H2020 EuroSea project aims at improving and integrating the European Ocean Observing and Forecasting System (see official website: [https://eurosea.eu/](https://eurosea.eu/)). It has received funding from the European Union’s Horizon 2020  research and innovation programme under grant agreement No 862626).

* _Task 2.3:_<br>
Task 2.3 has the objective to improve the design of multi-platform experiments aimed to validate the Surface Water and Ocean Topography (SWOT) satellite observations with the goal to optimize the utility of these observing platforms. Observing System Simulation Experiments (OSSEs) have been conducted to evaluate different configurations of the in situ observing system, including rosette and underway CTD, gliders, conventional satellite nadir altimetry and velocities from drifters. High-resolution models have been used to simulate the observations and to represent the “ocean truth”. Several methods of reconstruction have been tested: spatio-temporal optimal interpolation, machine-learning techniques, model data assimilation and the MIOST tool. **The planned OSSEs are detailed in this public report: [Barceló-Llull et al., 2020](https://doi.org/10.3289/eurosea_d2.1), and the full analysis will be published soon.**
Contributors to task 2.3 are CSIC (Lead), CLS, SOCIB, IMT and Ocean-Next. 

* _Subtask 2.3.1:_<br>
Subtask 2.3.1 aims to **evaluate different in situ sampling strategies to reconstruct fine-scale ocean currents (~20 km) in the context of SWOT**. An advanced version of the classic optimal interpolation used in field experiments, which considers the spatial and temporal variability of the observations, has been applied to reconstruct different configurations with the objective to evaluate the best sampling strategy to validate SWOT. 

* _Where?_<br>
The analysis  focuses on two regions of interest: **(i) the western Mediterranean Sea and (ii) the Subpolar North West Atlantic**. In the western Mediterranean Sea, the target area is located within a swath of SWOT, while in the North West Atlantic the region of study includes a crossover of  SWOT during the fast-sampling phase. 


## The codes:

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

