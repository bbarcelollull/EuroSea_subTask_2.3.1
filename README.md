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

### Simulation of CTD observations from different platforms (rosette, underway CTD, gliders) and ADCP observations

- 2 subregions of about 15ºx10º  (WESTMED and NANFL) 
- 2D surface fields extracted:  SSH, SST, SSS, U, V, lon, lat
- Currents: U,V at 15 m and at the surface.
- Period: 1 year (between 1-july-2009 and 30-june-2010)
- Fields available at native resolution (1/60º, hourly) 
- Fields also available at downgraded resolution (1/20º, daily) 
- +(NEMO mask,mesh files to describe the model grid)
- as of today this data is available upon request, contact us or the [MEOM group@IGE, Grenoble](https://meom-group.github.io/).

**Pseudo-obs  generated from the above model outputs and for the 2 subregions:**

- pseudo-SWOT (swath and nadir, both for the sampling and science phase) [[data and info here](./swot_pseudoobs.md)],
- pseudo alongtrack SENTINEL3,  SARAL, ENVISAT [[data and info here](./nadir_alongtrack.md)],
- Lagrangian particles (pseudo-drifters) in the MEDWEST subregion [[data and info here](./lagrangian_traj.md)].
