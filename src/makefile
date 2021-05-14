HDF5_DIR = /opt/cray/pe/hdf5-parallel/1.10.5.2/GNU/8.2
IDEALEM_DIR=/global/project/projectdirs/m1248/alex/media/sf_das_compression/include/idealem-package/build
FASTPFOR_DIR=/global/project/projectdirs/m1248/alex/media/sf_das_compression/include/FastPFor/build-install
MGARD_DIR=/global/project/projectdirs/m1248/alex/media/sf_das_compression/include/MGARD_BUILD
ZFP_DIR=/global/project/projectdirs/m1248/alex/media/sf_das_compression/include/zfp
SZ_DIR=/global/project/projectdirs/m1248/alex/media/sf_das_compression/include/SZ
CC       = CC


Other_FLAG=-g -lz
HDF5_FLAG=-I$(HDF5_DIR)/include -L$(HDF5_DIR)/lib -lhdf5  -lhdf5_hl
IDEALEM_FLAG=-I$(IDEALEM_DIR)/include -L$(IDEALEM_DIR)/lib -lidealem_c -lm
FASTPFOR_FLAG=-I$(FASTPFOR_DIR)/include/fastpfor -L$(FASTPFOR_DIR)/lib64 -lFastPFOR
MGARD_FLAG=-I$(MGARD_DIR)/include/mgard -L$(MGARD_DIR)/lib64 -lmgard
ZFP_FLAG=-I$(ZFP_DIR)/include -L$(ZFP_DIR)/build/lib64 -lzfp
SZ_FLAG=-I$(SZ_DIR)/include/sz -L$(SZ_DIR)/lib64 -lSZ -lzstd -lm
OPENMP_FLAG=-fopenmp

all: DAS_Compression_IDEALEM DAS_Compression_FastPFOR DAS_Decompress DAS_Compress_MGARD DAS_Compression_ZFP DAS_Compression_SZ xcorr_test_sz hdf52txt

DAS_Compression_IDEALEM: DAS_Compression_IDEALEM.cpp
	$(CC) -o DAS_Compression_IDEALEM DAS_Compression_IDEALEM.cpp $(HDF5_FLAG) $(Other_FLAG) $(IDEALEM_FLAG)
DAS_Compression_FastPFOR: DAS_Compression_FastPFOR.cpp
	$(CC) -o DAS_Compression_FastPFOR DAS_Compression_FastPFOR.cpp $(HDF5_FLAG) $(FASTPFOR_FLAG) $(Other_FLAG) #$(FASTPFOR_DIR)/lib64/libFastPFOR.a
DAS_Compress: DAS_Compress.cpp
	$(CC) -o DAS_Compress DAS_Compress.cpp $(HDF5_FLAG) $(FASTPFOR_FLAG) $(Other_FLAG)
DAS_Decompress: DAS_Decompress.cpp
	$(CC) -o DAS_Decompress DAS_Decompress.cpp $(HDF5_FLAG) $(FASTPFOR_FLAG) $(Other_FLAG)
DAS_Compress_MGARD: DAS_Compress_MGARD.cpp
	$(CC) -o DAS_Compress_MGARD DAS_Compress_MGARD.cpp $(HDF5_FLAG) $(MGARD_FLAG) $(OPENMP_FLAG) $(Other_FLAG)
DAS_Compression_ZFP: DAS_Compression_ZFP.cpp
	$(CC) -o DAS_Compression_ZFP DAS_Compression_ZFP.cpp $(HDF5_FLAG) $(ZFP_FLAG) $(Other_FLAG)
DAS_Compression_SZ: DAS_Compression_SZ.cpp
	$(CC) -o DAS_Compression_SZ DAS_Compression_SZ.cpp $(HDF5_FLAG) $(SZ_FLAG) $(Other_FLAG)
xcorr_test_sz: xcorr_test_sz.cpp
	$(CC) -o xcorr_test_sz xcorr_test_sz.cpp $(HDF5_FLAG) $(Other_FLAG)
hdf52txt: hdf52txt.cpp
	$(CC) -o hdf52txt hdf52txt.cpp $(HDF5_FLAG) $(FASTPFOR_FLAG) $(Other_FLAG)

clean: 
	rm -f *.h5 *.o \
        	DAS_Compression_IDEALEM
	rm -f *.h5 *.o \
		DAS_Compression_FastPFOR
	rm -f *.h5 *.o \
		DAS_Compress
	rm -f *.h5 *.o \
		DAS_Decompress
	rm -f *.h5 *.o \
		hdf52txt
	rm -f *.h5 *.o \
		DAS_Compress_MGARD
	rm -f *.h5 *.o \
		DAS_Compression_ZFP
	rm -f *.h5 *.o \
		DAS_Compression_SZ
	rm -f *.h5 *.o \
		xcorr_test_sz
