# DAS_Compression

## Description

This repository is a collection of compression methods tested on HDF5 data retrieved from distributed acoustic sensing (DAS) devices. The objective of this project is to find a compression method that encodes these data for long-term storage in order to reduce the frequency of trips to sensor locations to collect the data and reset the storage device. Ideally, this method should have a high compression ratio (highest priority), low reconstruction error, and relatively fast compression times (<1min).

## Prerequisites

### <ins>HDF5 C++ API</ins>

#### Download Page

https://www.hdfgroup.org/downloads/hdf5/

## Methods

### <ins>ZFP</ins>

#### Source Code

https://github.com/LLNL/zfp

#### Running from Terminal

File Name: DAS_Compression_ZFP

Description: Compresses + decompresses target file

./DAS_Compression_ZFP [INPUT_FILE] [DATA_DIRECTORY] -r [COMPRESSION_BIT_RATE (8 <= rate <= 16)] [OPTIONAL -d (Data Generation Flag)]

Example: ./DAS_Compression_ZFP /data/input/westSac_180103000031.tdms.h5 /data/ -r 8

Note: Data directory structure is assumed to contain an "output" directory which contains directories "compressed" and "decompressed" to hold the output files.

### <ins>SZ</ins>

#### Source Code

https://github.com/szcompressor/SZ

#### Running from Terminal

File Name: DAS_Compression_SZ

Description: Compresses + decompresses target file

./DAS_Compression_SZ [INPUT_FILE] [OUTPUT_DIRECTORY] [ERROR_BOUND_TYPE (ABS or REL)] [ERROR_BOUND] [COMPRESSION_MODE (-c for optimal compression ratio, -s for optimal compression speed] [OPTIONAL -d (Data Generation Flag)]

Example: ./DAS_Compression_ZFP /data/input/westSac_180103000031.tdms.h5 /data/output/ REL 0.003 -c

Note: Output directory structure is assumed to contain directories "compressed" and "decompressed" to hold the output files.

### <ins>FastPFOR</ins>

#### Source Code

https://github.com/lemire/FastPFor

#### Running from Terminal

File Name: DAS_Compress

Description: Compresses target file

./DAS_Compress -i [INPUT_FILE] -o [OUTPUT_DIRECTORY] -d [Dataset Name (DataCT or Acoustic)] [OPTIONAL -g (Data Generation Flag)]

Example: ./DAS_Compress -i /data/input/westSac_180103000031.tdms.h5 -o /data/output/ -d DataCT

### <ins>MGARD</ins>

#### Source Code

https://github.com/CODARcode/MGARD

#### Running from Terminal

File Name: DAS_Compress_MGARD

Description: Compresses target file

./DAS_Compress_MGARD -i [INPUT_FILE] -o [COMPRESSION_OUTPUT_DIRECTORY] -d [Dataset Name (DataCT or Acoustic)] [OPTIONAL -t (Transpose Flag)] [OPTIONAL -g (Data Generation Flag)]

Example: ./DAS_Compress_MGARD -i /data/input/westSac_180103000031.tdms.h5 -o /data/output/compressed/ -d DataCT

#### Continuing Work

MGARD compression code is fully written and commented. In order to implement the GPU version, clone the GitHub repo from above and follow the instructions in README_MGARD_GPU.md in order to build using GPU.

Prerequisites: NVIDIA GPU, NVCOMP Library, CUDA 11.0+

### <ins>TiLib</ins>

#### Source Code

https://github.com/Upliner/tilib

#### Running from Terminal

Note: Wavelet code and makefile are contained in src/wavelet/

File Name: DAS_Compress_wavelet

Description: Compresses target file

./DAS_Compress_wavelet -i [INPUT_FILE] -o [COMPRESSION_OUTPUT_DIRECTORY] -d [Dataset Name (DataCT or Acoustic)] [OPTIONAL -t (Transpose Flag)] [OPTIONAL -g (Data Generation Flag)]

Example: ./DAS_Compress_wavelet -i /data/input/westSac_180103000031.tdms.h5 -o /data/output/compressed/ -d DataCT

#### Continuing Work

Compression code is written and commented, but would not run on my local machine: the process would terminate with the signal "killed". After moving it to Cori, it appears that there is some issue with the HDF5 library upon attempting to compile.
