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

std::vector<int32_t> read_data_int32(std::string file_name, std::string dataset_name, int* dims, int rank)
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
    std::vector<int32_t> data(N);

    // read in the data and store it to the array
    int status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
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

std::vector<uint8_t> read_data_uint8(std::string file_name, std::string dataset_name, int* dims, int rank)
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

    return data;
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
    hid_t datatype = H5Tcopy(H5T_NATIVE_UCHAR);                                                     // using uint8

    // create new dataset
    hid_t output_dataset = H5Dcreate(output_file, dataset_name.c_str(), datatype, dataspace, 
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // write data to file
    int status = H5Dwrite(output_dataset, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input.data());
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

std::vector<int32_t> shift_vector_signed(std::vector<uint32_t> &input)
{
    std::vector<int32_t> output;
    int32_t shift_val = input[0];
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
    
    for (int i = 0; i < arr.size(); i++)
    {
        sum += abs(arr[i]);
    }

    double mean = sum / arr.size();

    return mean;
}

double std_dev(std::vector<int32_t> arr)
{
    double std = 0;
    double avg = mean(arr);

    for (int i = 0; i < arr.size(); i++)
    {
	std += pow(arr[i] - avg, 2);
    }

    return sqrt(std / arr.size());
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
FUNCTION: compress
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

int gen_text(std::string file_path, std::string DATASET_NAME)
{
    // get input dimensions
    int X = get_input_dim(file_path, DATASET_NAME, 0);                               // ROW_CHANNEL
    int Y = get_input_dim(file_path, DATASET_NAME, 1);				     // COL_TIME
    int dims [2] = {X, Y};
	
    // read in uncompressed data and fill array
    std::vector<int16_t> hdf5_arr = read_data_int16(file_path, DATASET_NAME, 
                                                               dims, 2);
    ofstream output;
    output.open("sample_data.txt");

    for (int i = 0; i <= X; i++)
    {
	for (int j = 0; j <= Y; j++)
	{
	    output << hdf5_arr[i*X + j] << ",";
	}
	output << "\n";
    }

    output.close();

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
int generate_data(std::string file_path, std::string comp_file_path, 
                  std::string DATASET_NAME, bool TRANSPOSE)
{
    uint64_t read_start_time = GetTimeMs64();

    // get input dimensions
    int nx_uncomp_input = get_input_dim(file_path, DATASET_NAME, 0);                               // ROW_CHANNEL
    int ny_uncomp_input = get_input_dim(file_path, DATASET_NAME, 1);                               // COL_TIME
    int uncomp_input_dims [2] = {nx_uncomp_input, ny_uncomp_input};

    // read in uncompressed data and fill array
    std::vector<int32_t> uncomp_input_signed = read_data_int32(file_path, DATASET_NAME, 
                                                               uncomp_input_dims, 2);
    std::vector<uint32_t> uncomp_input;

    // retrieve magnitude of most negative value in order to shift array to uint32
    int32_t shift_val = abs(min(uncomp_input_signed));

    uint64_t transpose_start_time = 0;
    uint64_t transpose_end_time = 0;

    // transform input data depending on value of TRANSPOSE flag
    if (!TRANSPOSE)
    {
        // insert transpose flag (0) at the beginning of the array to 
        // indicate the data has not been tranposed
        uncomp_input_signed.insert(uncomp_input_signed.begin(), 0);
	    uncomp_input = shift_vector_unsigned(uncomp_input_signed, shift_val);
    }
    else
    {
        // initialize empty array and fill it with the tranposed version of the original
    	std::vector<int32_t> uncomp_input_signed_transposed(uncomp_input_signed.size());
    	transpose_start_time = GetTimeMs64();
    	transpose_data(uncomp_input_signed, uncomp_input_signed_transposed, nx_uncomp_input, 
                       ny_uncomp_input);
        transpose_end_time = GetTimeMs64();

        // insert transpose flag (1) at the beginning of the array to 
        // indicate the data has been tranposed
        uncomp_input_signed_transposed.insert(uncomp_input_signed_transposed.begin(), 1);
        uncomp_input = shift_vector_unsigned(uncomp_input_signed_transposed, shift_val);
    }

    // NOTE: first two elements of the data to be compressed: 
    // [0]: shift value to be used to bring the uint32 array back to its original values stored
    // in the int32 array
    // [1]: transpose flag (which has been shifted by the shift value) --> 0 means untransposed
    // and 1 means transposed (after shifting back to original values)
    size_t input_size = uncomp_input.size();

    uint64_t read_end_time = GetTimeMs64();
  
    uint64_t compression_start_time = GetTimeMs64();

    // initialize output byte array to store compressed data
    std::vector<uint32_t> comp_output(input_size + 1024);
    size_t output_size = comp_output.size();

    // compress the data stored in uncomp_input
    IntegerCODEC &codec = *CODECFactory::getFromName("simdfastpfor256");
    codec.encodeArray(uncomp_input.data(), input_size, comp_output.data(), output_size);

    // ensure that the data was able to be compressed
    if (output_size > comp_output.size())
    {
        std::cout << "Compression Failed: Output buffer exceeded" << "\n";
    }

    // shrink size of compressed output array to size of the data stored
    comp_output.resize(output_size);
    comp_output.shrink_to_fit();       
                                                   
    uint64_t compression_end_time = GetTimeMs64();                           
    uint64_t write_start_time = GetTimeMs64();

    // write compressed data to file
    hsize_t comp_dims [1] = {comp_output.size()};                                    
    write_data_uint8(comp_output, comp_file_path, DATASET_NAME, 
                     comp_dims);
    uint64_t write_end_time = GetTimeMs64();

    // calculate compression ratio
    int nx_comp_output = get_input_dim(comp_file_path, DATASET_NAME, 0);
    double comp_ratio = ((double) (uncomp_input.size() * sizeof(int16_t))) 
                        / ((double) comp_output.size() * sizeof(uint8_t));

    // calculate various statistics of uncompressed data
    int32_t min_val_uncomp = min(uncomp_input_signed);
    int32_t max_val_uncomp = max(uncomp_input_signed);
    double mean_uncomp = mean(uncomp_input_signed);
    double abs_mean_uncomp = absolute_mean(uncomp_input_signed);
    double standard_deviation = std_dev(uncomp_input_signed);
	
    // output statistics to store in log file
    std::cout << comp_ratio << ",";
    std::cout << abs_mean_uncomp << ",";
    std::cout << standard_deviation << ",";
    std::cout << min_val_uncomp << ",";
    std::cout << max_val_uncomp << ",";
    std::cout << mean_uncomp << ",";
    std::cout << compression_end_time - compression_start_time << ",";
    std::cout << read_end_time - read_start_time << ",";
    std::cout << write_end_time - write_start_time << ",";
    std::cout << transpose_end_time - transpose_end_time << ",";

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
    std::string DATASET_NAME;

    int copt;
    while ((copt = getopt(argc, argv, "i:d:")) != -1)
	switch (copt)
	{
    // input file path (uncompressed data)
	case 'i':
	    FILE_PATH.assign(optarg);
	    break;
    // dataset name (/DataCT for older data, /Acoustic for newer data)
	case 'd':
	    DATASET_NAME.assign(optarg);
	    break;
	}

    gen_text(FILE_PATH, DATASET_NAME);
    
    return 0;
}
