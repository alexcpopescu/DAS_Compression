#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <ctime>
#include <regex> 
#include <iterator>
#include <dirent.h>
#include <unistd.h>
#include "hdf5.h"
#include "codecfactory.h"
#include "deltautil.h"

using namespace FastPForLib;

/*
FUNCTION: get_input_dim
DESCRIPTION:
    Accesses the dataset stored at file_name and dataset_name and returns the size of 
    dimension [dim].
INPUT: 
    string file_name: file path
    string dataset_name: name of hdf5 dataset in file
    int dim: which dimension size you want to query: i.e. x = 0, y = 1, z = 2, etc.
OUTPUT:
    RETURNS: size of dimension dim
*/

int get_input_dim(std::string file_name, std::string dataset_name, int dim)
{
    // open the file, dataset, and dataspace using default params
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen(file, dataset_name.c_str(), H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset);

    // query the rank and dimension sizes from the dataspace
    std::vector<unsigned long long> dim_size;
    int rank = H5Sget_simple_extent_ndims(dataspace);
    dim_size.resize(rank);
    H5Sget_simple_extent_dims(dataspace, &dim_size[0], NULL);

    // close the dataset and file
    H5Dclose(dataset);
    H5Fclose(file);

    return dim_size[dim];
}

/*
FUNCTION: read_data_int32
DESCRIPTION: 
    Reads the data stored at file_name and dataset_name as int16 and returns an array
    containing the data. Typically used to read in uncompressed/decompressed data (2D).
INPUT: 
    string file_name: file path
    string dataset_name: name of hdf5 dataset in file
    int* dims: integer array containing the size of each dimension
    int rank: number of dimensions of dataset (must be 1 or 2)
OUTPUT:
    array containing data from dataset within specified file
    RETURNS: 0
*/

std::vector<uint32_t> read_data_uint32(std::string file_name, std::string dataset_name, int* dims, int rank)
{
    // open the file and dataset using default params
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen(file, dataset_name.c_str(), H5P_DEFAULT);

    // calculate the product of the dimension sizes
    int N = 1;
    for (int i = 0; i < rank; i++)
    {
        N = N * dims[i];
    }

    // allocated memory to store the read data
    std::vector<uint32_t> data(N);

    // read in the data and store it to the array
    int status = H5Dread(dataset, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    assert(status >= 0);                                                                            // ensure the read executed successfully

    // close the dataset and file
    status = H5Dclose(dataset);
    status = H5Fclose(file);

    return data;
}

/*
FUNCTION: read_data_int16
DESCRIPTION: 
    Reads the data stored at file_name and dataset_name as int16 and returns an array
    containing the data. Typically used to read in uncompressed/decompressed data (2D).
INPUT: 
    string file_name: file path
    string dataset_name: name of hdf5 dataset in file
    int* dims: integer array containing the size of each dimension
    int rank: number of dimensions of dataset (must be 1 or 2)
OUTPUT:
    array containing data from dataset within specified file
    RETURNS: 0
*/

std::vector<int16_t> read_data_int16(std::string file_name, std::string dataset_name, int* dims, int rank)
{
    // open the file and dataset using default params
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen(file, dataset_name.c_str(), H5P_DEFAULT);

    // calculate the product of the dimension sizes
    int N = 1;
    for (int i = 0; i < rank; i++)
    {
        N = N * dims[i];
    }

    // allocated memory to store the read data
    std::vector<int16_t> data(N);

    // read in the data and store it to the array
    int status = H5Dread(dataset, H5T_NATIVE_SHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    assert(status >= 0);                                                                            // ensure the read executed successfully

    // close the dataset and file
    status = H5Dclose(dataset);
    status = H5Fclose(file);

    return data;
}

/*
FUNCTION: read_data_uint8
DESCRIPTION: 
    Reads the data stored at file_name and dataset_name as uint8 and returns an array
    containing the data. Typically used to read in compressed data (1D).
INPUT: 
    string file_name: file path
    string dataset_name: name of hdf5 dataset in file
    int* dims: integer array containing the size of each dimension
    int rank: number of dimensions of dataset (must be 1 or 2)
OUTPUT:
    array containing data from dataset within specified file
    RETURNS: 0
*/

std::vector<uint32_t> read_data_uint8(std::string file_name, std::string dataset_name, int* dims, int rank)
{
    // open the file and dataset using default params
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen(file, dataset_name.c_str(), H5P_DEFAULT);

    // calculate the product of the dimension sizes
    int N = 1;
    for (int i = 0; i < rank; i++)
    {
        N = N * dims[i];
    }

    // allocated memory to store the read data
    std::vector<uint8_t> data(N);

    // read in the data and store it to the array
    int status = H5Dread(dataset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    assert(status >= 0);                                                                            // ensure the read executed successfully

    // close the dataset and file
    status = H5Dclose(dataset);
    status = H5Fclose(file);

    // convert data to uint32 for decompression
    
    std::vector<uint32_t> output(data.size());
    
    for (int i = 0; i < N; i++)
    {
        output[i] = data[i];
    }
    
    
    /*
    for (int i = 0; i < data.size(); i++)
    {
	uint32_t val;
	std::memcpy(&val, &data[i], sizeof(std::uint32_t));
	output[i] = val;
    }
    */

    //std::copy(data.begin(), data.begin()+data.size(), output);

    return output;
}

/*
FUNCTION: write_data_int16
Description:
    Writes the int16 data stored in arr to a new int16 file/dataset with name file_name, 
    dataset_name. Dimension sizes are stored in dims. Typically used to write decompressed data to
    file. Data must be 2-dimensional and stored as int16.
INPUT: 
    void* arr: array containing data to be written to file
    string file_name: file path to desired write file
    string dataset_name: name of hdf5 dataset in desired write file
    const hsize_t* dims: array containing dimension sizes
OUTPUT:
    array written to file
    RETURNS: 0
*/

int write_data_int16(std::vector<int32_t> input, std::string file_name, std::string dataset_name, 
                     const hsize_t* dims)
{
    // create new file to store compressed data
    hid_t output_file = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // create dataspace, where
    // dims[0] = rows
    // dims[1] = cols
    hid_t dataspace = H5Screate_simple(2, dims, NULL);
    hid_t datatype = H5Tcopy(H5T_NATIVE_SHORT);                                                     // type of data on disk (int16)

    // create new dataset
    hid_t output_dataset = H5Dcreate(output_file, dataset_name.c_str(), datatype, dataspace, 
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // write data to file
    int status = H5Dwrite(output_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input.data());    // type of data in buffer (int32)
    assert(status >= 0);                                                                            // ensure the write executed successfully

    // close the dataset and file
    status = H5Dclose(output_dataset);
    status = H5Fclose(output_file);

    return 0;
}

/*
FUNCTION: write_data_uint8
Description:
    Writes the uint8 data stored in arr to a new file/dataset with name file_name, dataset_name. 
    Dimension sizes are stored in dims. Typically used to write compressed data to file.
    Data must be 1-dimensional.
INPUT: 
    void* arr: array containing data to be written to file
    string file_name: file path to desired write file
    string dataset_name: name of hdf5 dataset in desired write file
    const hsize_t* dims: singleton array containing dimension size
OUTPUT:
    array written to file
    RETURNS: 0
*/

int write_data_uint8(std::vector<uint32_t> input, std::string file_name, std::string dataset_name, 
                     const hsize_t* dims)
{
    // create new file to store compressed data
    hid_t output_file = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // create dataspace, where
    // dims[0] = rows * cols
    hid_t dataspace = H5Screate_simple(1, dims, NULL);
    hid_t datatype = H5Tcopy(H5T_NATIVE_UCHAR);                                                     // type of data on disk (uint8)

    // create new dataset
    hid_t output_dataset = H5Dcreate(output_file, dataset_name.c_str(), datatype, dataspace, 
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // write data to file
    int status = H5Dwrite(output_dataset, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT,           // type of data in buffer (uint32)
                          input.data());
    assert(status >= 0);                                                                            // ensure the write executed successfully

    // close the dataset and file
    status = H5Dclose(output_dataset);
    status = H5Fclose(output_file);

    return 0;
}

std::vector<int32_t> vector_diff_uncomp(std::vector<int32_t> &input)
{
    std::vector<int32_t> output;
    output.push_back(input[0]);
    for (int i = 0; i < input.size() - 1; i++)
    {
	output.push_back(input[i + 1] - input[i]);
    }
    return output;
}

std::vector<int32_t> vector_diff_decomp(std::vector<int32_t> &input)
{
    std::vector<int32_t> output;
    output.push_back(input[0]);
    for (int i = 1; i < input.size(); i++)
    {
	output.push_back(output[i - 1] + input[i]);
    }
    return output;
}

std::vector<uint32_t> shift_vector_unsigned(std::vector<int32_t> &input, int32_t shift_val)
{
    std::vector<uint32_t> output;
    for (int i = 0; i < input.size(); i++)
    {
        output.push_back(input[i] + shift_val);
    }
    return output;
}

std::vector<int32_t> shift_vector_signed(std::vector<uint32_t> &input, int32_t shift_val)
{
    std::vector<int32_t> output;
    for (int i = 1; i < input.size(); i++)
    {
        output.push_back(input[i] - shift_val);
    }
    return output;
}

void transpose_data(std::vector<int32_t> &input, std::vector<int32_t> &output, const int N, const int M)
{
    for (int n = 0; n < N * M; n++)
    {
	int i = n / N;
	int j = n % N;
	output[n] = input[M * j + i];
    }
}

/*
FUNCTION: rmse
DESCRIPTION:
    Calculates the root mean square error (RMSE) between two int16 arrays
INPUT: 
    int16_t* arr1, int16_t* arr2: arrays to calculate the rmse between
    int arr_size: size of the arrays
OUTPUT:
    RETURNS: RMSE
*/

double rmse(std::vector<int32_t> arr1, std::vector<int32_t> arr2)
{
    double rmse_val = 0;

    for (int i = 0; i < arr1.size(); i++)
    {
        double sq_diff = pow((double) arr1[i] - (double) arr2[i], 2.0);                                 // calculate and square the difference (error) between elements at arr1[i], arr2[i]
        rmse_val += sq_diff;
    }

    rmse_val = rmse_val / arr1.size();                                                                     // calculate the mean
    rmse_val = sqrt(rmse_val);                                                                          // square root of the mean

    return rmse_val;
}

/*
FUNCTION: min
DESCRIPTION:
    Calculate the minimum of the given int16 array
INPUT: 
    int16_t* arr: input array
    int arr_size: size of the array
OUTPUT:
    RETURNS: minumum value of arr
*/

int32_t min(std::vector<int32_t> arr) 
{
    int32_t min_val = 0;

    for (int i=0; i<arr.size(); i++)
    {
        int32_t curr_val = arr[i];
        if (curr_val < min_val)
        {
            min_val = curr_val;
        }
    }

    return min_val;
}

/*
FUNCTION: max
DESCRIPTION:
    Calculate the maximum of the given int16 array
INPUT: 
    int16_t* arr: input array
    int arr_size: size of the array
OUTPUT:
    RETURNS: maximum value of arr
*/

int32_t max(std::vector<int32_t> arr) 
{
    int32_t max_val = 0;
    
    for (int i=0; i<arr.size(); i++)
    {
        int32_t curr_val = arr[i];
        if (curr_val > max_val)
        {
            max_val = curr_val;
        }
    }

    return max_val;
}

/*
FUNCTION: mean
DESCRIPTION:
    Calculate the mean of the given int16 array
INPUT: 
    int16_t* arr: input array
    int arr_size: size of the array
OUTPUT:
    RETURNS: mean value of arr
*/

double mean(std::vector<int32_t> arr) 
{
    double sum = 0;
    
    for (int i=0; i<arr.size(); i++)
    {
        sum += arr[i];
    }

    double mean = sum / arr.size();

    return mean;
}

/*
FUNCTION: absolute_mean
DESCRIPTION:
    Calculate the mean of the absolute values stored in given int16 array
INPUT: 
    int16_t* arr: input array
    int arr_size: size of the array
OUTPUT:
    RETURNS: mean value of arr
*/

double absolute_mean(std::vector<int32_t> arr) 
{
    double sum = 0;
    
    for (int i=0; i<arr.size(); i++)
    {
        sum += abs(arr[i]);
    }

    double mean = sum / arr.size();

    return mean;
}

/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
 * windows and linux. 
 * 
 * CREDIT : Thomas Bonini, https://stackoverflow.com/a/1861337
 * 
 * */

uint64_t GetTimeMs64()
{
    /* Linux */
    struct timeval tv;

    gettimeofday(&tv, NULL);

    uint64_t ret = tv.tv_usec;
    /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
    ret /= 1000;

    /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
    ret += (tv.tv_sec * 1000);

    return ret;
}

/*
FUNCTION: test_compression
DESCRIPTION: 
    Compresses data stored at file_path, then decompresses the data. Calculates min, max, mean
    of uncompressed and decompressed files, as well as the RMSE between them. Also calculates the 
    absolute mean of the uncompressed data to normalize the RMSE data. Typically used for
    debugging.
INPUT: 
    string file_path: file path to input (uncompressed) data --> should be /root/.../data/input/file_name
    string compressed_file_path: file path to output (compressed) data --> should be 
                                 ./root/.../data/output/compressed/file_name_compressed
    string decompressed_file_path: file path to output (decompressed) data --> should be 
                                 /root/.../data/output/decompressed/file_name_decompressed
OUTPUT:
    Compressed and uncompressed data files in /root/.../data/output/compressed and 
    /root/.../data/output/decompressed
    PRINTS: 
        - Min, max, mean of uncompressed and decompressed files, as well as the RMSE between them
          and abs mean of the uncompressed file
        - Read, write, compression, decompression time
    RETURNS: 0
*/

int decompress(std::string file_path, std::string decomp_file_path, 
               std::string DATASET_NAME, int N_ROWS, int N_COLS)
{
    uint64_t read_start_time = GetTimeMs64();

    // get input dimensions
    int nx_comp_input = get_input_dim(file_path, DATASET_NAME, 0);                                 // size of compressed data         
    int comp_input_dims [1] = {nx_comp_input};

    // create and fill uint8 array that contains the 1D compressed data
    std::vector<uint32_t> comp_input = read_data_uint32(file_path, DATASET_NAME, 
                                                       comp_input_dims, 1);

    uint64_t read_end_time = GetTimeMs64();
    std::cout << "Read Time: " << (read_end_time - read_start_time) << "ms\n\n";

    // initialize output vector to store decompressed data
    std::vector<uint32_t> decomp_output_unsigned(N_ROWS * N_COLS + 2);
    size_t output_size = decomp_output_unsigned.size();

    std::cout << "Output Size: " << output_size << "\n";

    // decompress the data
    std::cout << "Decompressing Data...\n";
    uint64_t decompression_start_time = GetTimeMs64();
    IntegerCODEC &codec = *CODECFactory::getFromName("simdfastpfor256");
    codec.decodeArray(comp_input.data(), comp_input.size(), decomp_output_unsigned.data(), output_size);
    uint64_t decompression_end_time = GetTimeMs64();
    std::cout << "Decompression Time: " << (decompression_end_time - decompression_start_time) 
              << "ms\n\n"; 

    // read metadata from compressed data and initialize decompression parameters
    int32_t shift_val = decomp_output_unsigned[0];
    int32_t TRANSPOSE = decomp_output_unsigned[1];
    decomp_output_unsigned.erase(decomp_output_unsigned.begin(), decomp_output_unsigned.begin() + 1);

    // transform data back to original uncompressed data
    std::vector<int32_t> decomp_output;

    int64_t transpose_start_time = 0;
    int64_t transpose_end_time = 0;

    if (TRANSPOSE == 0)
    {
        decomp_output = shift_vector_signed(decomp_output_unsigned, shift_val);
    }
    else
    {
	std::vector<int32_t> decomp_output_unsigned_transposed = shift_vector_signed(decomp_output_unsigned, shift_val);
        decomp_output.resize(decomp_output_unsigned.size());
	transpose_start_time = GetTimeMs64();
	transpose_data(decomp_output_unsigned_transposed, decomp_output, N_COLS,
                       N_ROWS);      
	transpose_end_time = GetTimeMs64();	
       	std::cout << "Transpose Time: " << (transpose_end_time - transpose_start_time) << "ms\n\n";
    }                                                                                                                                                                            

    // write decompressed data to file   
    hsize_t decomp_dims [2] = {(hsize_t) N_ROWS, (hsize_t) N_COLS};
    uint64_t write_start_time = GetTimeMs64();
    write_data_int16(decomp_output, decomp_file_path, DATASET_NAME, decomp_dims);
    uint64_t write_end_time = GetTimeMs64();
    std::cout << "Write Time: " << (write_end_time - write_start_time) << "ms\n\n";
    
    return 0;
}

/*
FUNCTION: generate_data
DESCRIPTION: 
    Compresses data stored at file_path, then decompresses the data. Calculates min, max, mean
    of uncompressed and decompressed files, as well as the RMSE between them. Also calculates the 
    absolute mean of the uncompressed data to normalize the RMSE data. Prints out the values in 
    comma-delimited format to allow for easy write to csv by redirecting terminal output. Typically
    the main function that should be used to collect data for algorithm analysis. Otherwise, 
    performs the same process as test_compression.
INPUT: 
    string file_path: file path to input (uncompressed) data --> should be /root/.../data/input/file_name
    string compressed_file_path: file path to output (compressed) data --> should be 
                                 /root/.../data/output/compressed/file_name_compressed
    string decompressed_file_path: file path to output (decompressed) data --> should be 
                                 /root/.../data/output/decompressed/file_name_decompressed
OUTPUT:
    Compressed and uncompressed data files in /root/.../data/output/compressed and 
    /root/.../data/output/decompressed
    PRINTS: 
        - Min, max, mean of uncompressed and decompressed files, as well as the RMSE between them
          and abs mean of the uncompressed file
        - Read, write, compression, decompression time
    RETURNS: 0
*/
int generate_data(std::string file_path, std::string decomp_file_path, 
                  std::string DATASET_NAME, int N_ROWS, int N_COLS)
{
    uint64_t read_start_time = GetTimeMs64();

    // get input dimensions
    int nx_comp_input = get_input_dim(file_path, DATASET_NAME, 0);                                 // size of compressed data         
    int comp_input_dims [1] = {nx_comp_input};

    // create and fill uint8 array that contains the 1D compressed data
    std::vector<uint32_t> comp_input = read_data_uint8(file_path, DATASET_NAME, 
                                                       comp_input_dims, 1);

    uint64_t read_end_time = GetTimeMs64();

    // initialize output vector to store decompressed data
    size_t output_size = N_ROWS * N_COLS;
    std::vector<uint32_t> decomp_output_unsigned(output_size);

    // decompress the data
    uint64_t decompression_start_time = GetTimeMs64();
    IntegerCODEC &codec = *CODECFactory::getFromName("simdfastpfor256");
    codec.decodeArray(comp_input.data(), comp_input.size(), decomp_output_unsigned.data(), output_size);
    uint64_t decompression_end_time = GetTimeMs64(); 

    // read metadata from compressed data and initialize decompression parameters
    int32_t shift_val = decomp_output_unsigned[0];
    int32_t TRANSPOSE = decomp_output_unsigned[1];
    decomp_output_unsigned.erase(decomp_output_unsigned.begin(), decomp_output_unsigned.begin() + 2);

    // transform data back to original uncompressed data
    std::vector<int32_t> decomp_output;

    int64_t transpose_start_time = 0;
    int64_t transpose_end_time = 0;

    if (TRANSPOSE == 0)
    {
        decomp_output = shift_vector_signed(decomp_output_unsigned, shift_val);
    }
    else
    {
	std::vector<int32_t> decomp_output_unsigned_transposed = shift_vector_signed(decomp_output_unsigned, shift_val);
	decomp_output.resize(decomp_output_unsigned.size());
        transpose_start_time = GetTimeMs64();
        transpose_data(decomp_output_unsigned_transposed, decomp_output, N_COLS,
                       N_ROWS);
        transpose_end_time = GetTimeMs64();
    }                                                                                                  

    // write decompressed data to file   
    hsize_t decomp_dims [2] = {(hsize_t) N_ROWS, (hsize_t) N_COLS};
    uint64_t write_start_time = GetTimeMs64();
    write_data_int16(decomp_output, decomp_file_path, DATASET_NAME, decomp_dims);
    uint64_t write_end_time = GetTimeMs64();
	
    // output runtimes to store in log file
    std::cout << read_end_time - read_start_time << ",";
    std::cout << decompression_end_time - decompression_start_time << ",";
    std::cout << transpose_end_time - transpose_start_time << ",";
    std::cout << write_end_time - write_start_time << ",";

    return 0;
}

/*
DESCRIPTION:
- Compresses all files in data directory.
- Decompresses all files.
- Computes min, max, mean of uncompressed/decompressed data and calculates the RMSE between the
  uncompressed and decompressed data. Also calculates the absolute mean of the uncompressed data
  to normalize the RMSE.
ARGUMENTS:
- Directory path of uncompressed data: (should be /root/.../data/)
- Error bound mode: must be ABS (absolute error) or REL (relative error)
- Error bound: float/double that describes how tight the error bound for compression should be
- Compression mode: must be -c (best compression ratio) or -s (best speed)
- Data generation mode (OPTIONAL): -d if you wish to generate data for analysis
*/

int main(int argc, char *argv[])
{
    uint64_t start_time = GetTimeMs64();

    // initialize arguments
    std::string FILE_PATH;
    std::string OUTPUT_DIRECTORY;
    std::string DATASET_NAME;
    std::string ROWS;
    std::string COLS;
    bool DATA_GEN;

    int copt;
    while ((copt = getopt(argc, argv, "i:o:d:r:c:g")) != -1)
	switch (copt)
	{
    // input file path (compressed data)
	case 'i':
	    FILE_PATH.assign(optarg);
	    break;
    // output directory path (where to store the decompressed data)
	case 'o':
	    OUTPUT_DIRECTORY.assign(optarg);
	    break;
    // dataset name (/DataCT for older data, /Acoustic for newer data)
	case 'd':
	    DATASET_NAME.assign(optarg);
	    break;
    // number of rows in the dataset (size of dimension 0)
    case 'r':
	    ROWS.assign(optarg);
	    break;
    // number of columns in the dataset (size of dimension 1)
    case 'c':
	    COLS.assign(optarg);
	    break;
    // data generation flag used to gather data for testing/analysis --> output to terminal
    // to store in log file
	case 'g':
	    DATA_GEN = true;
	    break;
	}

    int N_ROWS = std::stoi(ROWS);
    int N_COLS = std::stoi(COLS);

    // assert that the required arguments have been called
    if (argc < 4)
    {
        std::cout << "Arguments must be listed as follows:\n[UNCOMPRESSED_FILE_PATH]"
                     "[OUTPUT_DIRECTORY_PATH][DATASET_NAME][-d OPTIONAL (data generation)]";
        return -1;
    }

    // use RegEx to extract raw file name (excluding path, _compressed tag, and extension)
    size_t path_idx = FILE_PATH.find_last_of("/");
    std::string RAW_FILE_NAME = FILE_PATH.substr(path_idx + 1); 
    size_t ext_idx = RAW_FILE_NAME.find("_compressed"); 
    RAW_FILE_NAME = RAW_FILE_NAME.substr(0, ext_idx); 
    std::string FILE_NAME = FILE_PATH.substr(path_idx);

    // file path of decompressed data
    std::string DECOMPRESSED_FILE_PATH = OUTPUT_DIRECTORY + RAW_FILE_NAME + "_decompressed.h5";
    std::cout << RAW_FILE_NAME;

    uint64_t end_time;

    // run decompression on file
    if (!DATA_GEN)
    {
        std::cout << "\nDecompressing file " << FILE_NAME << "\n\n";
        std::cout << "File Path: " << FILE_PATH << "\n";
        std::cout << "File Name: " << RAW_FILE_NAME << "\n";
        std::cout << "Decompressed File Path: " << DECOMPRESSED_FILE_PATH << "\n\n";
        decompress(FILE_PATH, DECOMPRESSED_FILE_PATH, DATASET_NAME, N_ROWS, N_COLS);
        end_time = GetTimeMs64();
        std::cout << "Total Time: " << end_time - start_time << "ms\n\n";
    }
    // run decompression on file and calculate statistics
    else
    {
        std::cout << FILE_NAME << ",";
        generate_data(FILE_PATH, DECOMPRESSED_FILE_PATH, DATASET_NAME, N_ROWS, N_COLS);
	    end_time = GetTimeMs64();
	    std::cout << end_time - start_time << ",\n";
    }

    return 0;
}
