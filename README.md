# Code for Anderson et al. (2023) Earthquake Infrasound Paper in Nature Communications Earth and Environment

### Contents
This repository includes Python code and data to reproduce figures 4-6 in the paper.

Folder 'code/' contains the code to make these figures.

Folder 'cleanbf/' contains the package for running clean beamforming. This is the particular version of cleanbf used in this paper; the package as available elsewhere may eventually be updated in a way that breaks compatibility with this paper's code, so be sure to use the package version included here.

Folder 'data/' mostly contains empty placeholder folders that will eventually contain files that are downloaded or generated by code. 'data/seismic/' includes small seismic traces that are used to estimate seismic wave speeds for figure 6. 'data/state_lines' includes files used to plot state lines on figure 6.

Folder 'figures/' is an empty folder where the code will save figures.

### Preparing to run code
To run the code, an appropriate environment with Python and dependency packages must be installed. We recommend installing Anaconda or Miniconda, then creating the environment with the following commands:
```
conda deactivate
conda create -y -n earthquake_infrasound python=3.10.5 matplotlib=3.5.1 numpy=1.23.1 obspy=1.3.0 pandas=1.3.5 scipy=1.8.0 spyder=5.4.0 pygmt=0.7.0 geopandas=0.11.0
conda activate earthquake_infrasound
```
These commands also install Spyder, a convenient interface for scientific programming, but you can use a different IDE or even a plain ipython terminal if you want. Spyder supports breaking code into cells using the '#%%' symbol, and we strongly recommend that you run those cells one at a time and in order. Whatever interface you use, you must open it from within the 'supplement/' folder (or set that to your working directory); otherwise, the paths in the code will not work. 

### Re-creating paper figures
First, open and run 'download_waveforms.py' to download all the infrasound waveform data from IRIS-Data Management Center. The waveforms will be stored in 'data/downloads/waveforms/', and minimally-preprocessed data for each event will be written to files in 'data/'.

Then, for each of figures 3-6 and supplementary figures 1-2, run the script with the corresponding name. To save time, expensive analyses are run in 'if True:' blocks, and their results are saved in pickle files. You can use pre-calculated results by leaving it as 'if False:' or change it to 'if True:' to recalculate (some analyses will take a long time).
