## Qopen Finland

This repository contains the source code for the reproduction of results and figures of the following publication:

Tom Eulenfeld, Gregor Hillers, Tommi A. T. Vuorinen and Ulrich Wegler (2023),
Induced earthquake source parameters, attenuation, and site effects from waveform envelopes in the Fennoscandian Shield,
*Journal of Geophysical Research: Solid Earth*,
doi: [10.1029/2022JB025162](https://doi.org/10.1029/2022JB025162).

Additionally, this repository contains result files, figures and metadata files.

### Structure

#### Result files

*`qopen/01_go/results.json`* Attenuation results (g, b)\
*`data/Q.json`* Alternatively, Q values for this study can be looked up here under key Eulenfeld2013\
*`qopen/02_sites/results.json`* Site amplifications\
*`qopen/04_source_nconst/results.json`* Source displacement spectra, fc, M0 for 2018 earthquakes\
*`qopen/07_source_2020_nconst/results.json`* Source displacement spectra, fc, M0 for 2020 earthquakes\
*`data/eq_params_20??.csv`* Earthquake parameters for 2018/2020 analyzed earthquakes in CSV format

#### Metadata files

*`data/events*.*`* CSV and CSZ (including picks) files with 2018/2020 events\
*`data/*stations*.*`* Used station metadata\
*`data/wellpath20??.txt`* Well paths of 2018/2020 boreholes\
*`data/Q.json`* Scattering and intrinsic Q values from different studies


#### Other files

Qopen configuration, results, logs, plots and the calling script are located in the `qopen` folder.
Python scripts are located in the `scripts` folder.
Figures used in the publication and additional figures are located in the `figs` folder.


### Preparation for running codes

1. Download or clone this repository.
2. Install the the relevant python packages: `qopen>=4.4 obspy>=1.4.0 obspycsv>=1.0.0 cartopy shapely pyproj`. The minimum version also specifies the versions I used.
   obspycsv is not needed for ObsPy>=1.5.
   The following is working for me when using conda
   ```
   conda --add channels conda-forge
   conda create -n eqspec obspy statsmodels cartopy shapely pyproj
   conda activate eqspec
   pip install qopen obspycsv
   ```
3. Download the waveforms needed to run Qopen and put them into `data` folder:
   [ISM_2018](https://doi.org/10.23729/6d15a5ea-7671-4bab-88a1-71f4ed962276),
   [ISUH_2018](https://doi.org/10.23729/39cfac4f-4d0d-4fb4-83dc-6f67e8ba8dce),
   [ISUH_2020](https://doi.org/10.23729/cdfd937c-37d5-46b0-9c16-f6e0c10bc81f).


### Running Qopen

Switch to the `qopen` folder with the Qopen configuration and results. Check out the `run_qopen.sh` script which runs a set of Qopen commands.
Run these one by one or the script itself to recreate the results. Additionaly, you will find plots of envelope fits and intermediate results which are not in the repository.


### Run scripts

Switch to the `scripts` directory and run the python scripts to recreate the plots in the `figs` folder, metadata and result files in the `data` folder.
