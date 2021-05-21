HDF5_DIR 	 = /opt/cray/pe/hdf5-parallel/1.10.5.2/GNU/8.2
TILIB_DIR = /global/project/projectdirs/m1248/alex/media/sf_das_compression/include/tilib
CC       	= g++


Other_FLAG=-g -lz -Wall
HDF5_FLAG=-I$(HDF5_DIR)/include -L$(HDF5_DIR)/lib -lhdf5  -lhdf5_hl
TILIB_FLAG=-I$(TILIB_DIR)/install/include/tilib

ti_SOURCES = \
	$(TILIB_DIR)/src/ari.o\
	$(TILIB_DIR)/src/bitio.o\
	$(TILIB_DIR)/src/butterworth.o\
	$(TILIB_DIR)/src/color.o\
	$(TILIB_DIR)/src/daub97.o\
	$(TILIB_DIR)/src/extend.o\
	$(TILIB_DIR)/src/nodelist.o\
	$(TILIB_DIR)/src/pbm.o\
	$(TILIB_DIR)/src/spiht.o\
	$(TILIB_DIR)/src/split.o\
	$(TILIB_DIR)/src/tilib.o

all: DAS_Compress_wavelet
 
DAS_Compress_wavelet: DAS_Compress_wavelet.cpp
	gcc -c $(TILIB_DIR)/src/ari.c -o $(TILIB_DIR)/src/ari.o $(Other_FLAG)
	gcc -c $(TILIB_DIR)/src/bitio.c -o $(TILIB_DIR)/src/bitio.o $(Other_FLAG)
	gcc -c $(TILIB_DIR)/src/butterworth.c -o $(TILIB_DIR)/src/butterworth.o $(Other_FLAG)
	gcc -c $(TILIB_DIR)/src/color.c -o $(TILIB_DIR)/src/color.o $(Other_FLAG)
	gcc -c $(TILIB_DIR)/src/daub97.c -o $(TILIB_DIR)/src/daub97.o $(Other_FLAG)
	gcc -c $(TILIB_DIR)/src/extend.c -o $(TILIB_DIR)/src/extend.o $(Other_FLAG)
	gcc -c $(TILIB_DIR)/src/nodelist.c -o $(TILIB_DIR)/src/nodelist.o $(Other_FLAG)
	gcc -c $(TILIB_DIR)/src/pbm.c -o $(TILIB_DIR)/src/pbm.o $(Other_FLAG)
	gcc -c $(TILIB_DIR)/src/spiht.c -o $(TILIB_DIR)/src/spiht.o $(Other_FLAG)
	gcc -c $(TILIB_DIR)/src/split.c -o $(TILIB_DIR)/src/split.o $(Other_FLAG)
	gcc -c $(TILIB_DIR)/src/tilib.c -o $(TILIB_DIR)/src/tilib.o $(Other_FLAG)
	g++ -c DAS_Compress_wavelet.cpp -o DAS_Compress_wavelet.o $(HDF5_FLAG) $(TILIB_FLAG) $(Other_FLAG)
	$(CC) -o DAS_Compress_wavelet DAS_Compress_wavelet.o $(ti_SOURCES) $(HDF5_FLAG) $(TILIB_FLAG) $(Other_FLAG)

clean: 
	rm -f *.h5 *.o \
		DAS_Compress_wavelet
