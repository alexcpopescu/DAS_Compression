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
#include "hdf5.h"
#include "sz.h"
#include "rw.h"

std::string DATASET_NAME("/DataCT"); // 2D dataset, where rows(X) correspond to channels and columns(Y) correspond to time

/*
FUNCTION: compress
DESCRIPTION: 
    Compresses the uncompressed data stored in input_arr (2D int16 data) and stores the compressed
    data in output_arr (1D uint8 data).
INPUT: 
    void* input_arr: uncompressed array containing hdf5 data
    int* input_dims: array of dimension sizes of the uncompressed data
    uint8_t* output_buff: output array to store compressed data
OUTPUT:
    compressed array
    RETURNS: byte size of compressed stream
*/

size_t compress(void* input_arr, int* input_dims, uint8_t* &output_arr)
{
    int datatype = SZ_INT16;                                                                        // using int16
    size_t outSize;
    output_arr = SZ_compress(datatype, input_arr, &outSize, 0, 0, 0, input_dims[1], input_dims[0]);
    return outSize;                                                                                 // size of compressed data (in bytes)
}

/*
FUNCTION: decompress
DESCRIPTION: 
    Deompresses the compressed data stored in input_arr (1D uint8 data) and stores the 
    decompressed data in output_arr (2D int16 data).
INPUT: 
    uint8_t* input_arr: array containing compressed data (1D uint8 data)
    int* output_dims: dimension sizes of the output (decompressed) data (same as uncompressed data)
    int16_t* output_arr: empty array to store decompressed data (2D int16 data)
    size_t bytelength: size of compressed data (in bytes)
OUTPUT:
    decompressed array
    RETURNS: void
*/

void decompress(uint8_t* input_arr, int* output_dims, int16_t* &output_arr, size_t bytelength)
{
    int datatype = SZ_INT16;                                                                        // using int16
	output_arr = (int16_t*)SZ_decompress(datatype, input_arr, bytelength, 0, 0, 0, output_dims[1], 
                                         output_dims[0]);
}


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

int16_t* read_data_int16(std::string file_name, std::string dataset_name, int* dims, int rank)
{
    // open the file and dataset using default params
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen(file, dataset_name.c_str(), H5P_DEFAULT);

    // calculate the product of the dimension sizes
    int malloc_size = 1;
    for (int i = 0; i < rank; i++)
    {
        malloc_size = malloc_size * dims[i];
    }

    // allocated memory to store the read data
    int16_t *data = (int16_t *)malloc(sizeof(int16_t) * malloc_size);

    // read in the data and store it to the array
    int status = H5Dread(dataset, H5T_NATIVE_SHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
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

uint8_t* read_data_uint8(std::string file_name, std::string dataset_name, int* dims, int rank)
{
    // open the file and dataset using default params
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen(file, dataset_name.c_str(), H5P_DEFAULT);

    // calculate the product of the dimension sizes
    int malloc_size = 1;
    for (int i = 0; i < rank; i++)
    {
        malloc_size = malloc_size * dims[i];
    }

    // allocated memory to store the read data
    uint8_t *data = (uint8_t *)malloc(sizeof(uint8_t) * malloc_size);

    // read in the data and store it to the array
    int status = H5Dread(dataset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
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

int write_data_int16(void* arr, std::string file_name, std::string dataset_name, 
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
    int status = H5Dwrite(output_dataset, H5T_NATIVE_SHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr);    // type of data in buffer (int16)
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

int write_data_uint8(void* arr, std::string file_name, std::string dataset_name, 
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
    int status = H5Dwrite(output_dataset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr);
    assert(status >= 0);                                                                            // ensure the write executed successfully

    // close the dataset and file
    status = H5Dclose(output_dataset);
    status = H5Fclose(output_file);

    return 0;
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

double rmse(int16_t* arr1, int16_t* arr2, int arr_size)
{
    double rmse_val = 0;

    for (int i = 0; i < arr_size; i++)
    {
        double sq_diff = pow((double) arr1[i] - (double) arr2[i], 2.0);                                 // calculate and square the difference (error) between elements at arr1[i], arr2[i]
        rmse_val += sq_diff;
    }

    rmse_val = rmse_val / arr_size;                                                                     // calculate the mean
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

int16_t min(int16_t* arr, int arr_size) 
{
    int16_t min_val = 0;

    for (int i=0; i<arr_size; i++)
    {
        int16_t curr_val = arr[i];
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

int16_t max(int16_t* arr, int arr_size) 
{
    int16_t max_val = 0;
    
    for (int i=0; i<arr_size; i++)
    {
        int16_t curr_val = arr[i];
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

double mean(int16_t* arr, int arr_size) 
{
    double sum = 0;
    
    for (int i=0; i<arr_size; i++)
    {
        sum += arr[i];
    }

    double mean = sum / arr_size;

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

double absolute_mean(int16_t* arr, int arr_size) 
{
    double sum = 0;
    
    for (int i=0; i<arr_size; i++)
    {
        sum += abs(arr[i]);
    }

    double mean = sum / arr_size;

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

int test_compression(std::string file_path, std::string compressed_file_path, 
                     std::string decompressed_file_path)
{
    uint64_t read_start_time = GetTimeMs64();

    // get input dimensions
    int nx_uncompressed_input = get_input_dim(file_path, DATASET_NAME, 0);                          // ROW_CHANNEL
    int ny_uncompressed_input = get_input_dim(file_path, DATASET_NAME, 1);                          // COL_TIME
    int uncompressed_input_dims [2] = {nx_uncompressed_input, ny_uncompressed_input};

    // read in uncompressed data and fill array
    int16_t *data_uncompressed_input = read_data_int16(file_path, DATASET_NAME, 
                                                       uncompressed_input_dims, 2);                 // using int16
    uint64_t read_end_time = GetTimeMs64();
    std::cout << "Read Time: " << (read_end_time - read_start_time) << "ms\n\n";

    std::cout << "Compressing Data...\n";
    uint64_t compression_start_time = GetTimeMs64();

    // output byte array to store compressed data
    uint8_t* data_compressed_output;

    // compressed the data stored in data_uncompressed_input
    size_t szsize_compressed = compress(data_uncompressed_input, uncompressed_input_dims, 
                                        data_compressed_output);       
                                                   
    free(data_uncompressed_input);
    uint64_t compression_end_time = GetTimeMs64();
    std::cout << "Compression Time: " << (compression_end_time - compression_start_time) << "ms\n";
    double compression_ratio = ((double) (nx_uncompressed_input * ny_uncompressed_input * 
                                 sizeof(int16_t))) / ((double) szsize_compressed);
    std::cout << "Compression Ratio: " << compression_ratio << "\n\n";                           
    uint64_t write_start_time = GetTimeMs64();

    // write compressed data to file
    hsize_t data_compressed_dims [1] = {szsize_compressed};                                         // buffer dimensions = size of compressed file                                       
    write_data_uint8(data_compressed_output, compressed_file_path, DATASET_NAME, 
               data_compressed_dims);
    uint64_t write_end_time = GetTimeMs64();
    std::cout << "Write Time: " << (write_end_time - write_start_time) << "ms\n\n";
    free(data_compressed_output);

    // get input dimensions
    int nx_compressed_input = get_input_dim(compressed_file_path, DATASET_NAME, 0);                 // size of compressed file in bytes         
    int compressed_input_dims [1] = {nx_compressed_input};

    // create and fill uint8 array that contains the 1D compressed data
    uint8_t *data_compressed_input = read_data_uint8(compressed_file_path, DATASET_NAME, 
                                                         compressed_input_dims, 1);

    // initialize output array to store decompressed data
    int16_t *data_decompressed_output;

    std::cout << "Decompressing Data...\n";
    uint64_t decompression_start_time = GetTimeMs64();

    int decompressed_output_dims [2] = {nx_uncompressed_input, ny_uncompressed_input};

    // decompress the data
    decompress(data_compressed_input, decompressed_output_dims, data_decompressed_output, 
               nx_compressed_input); 
                                                    
    free(data_compressed_input);
    uint64_t decompression_end_time = GetTimeMs64();
    std::cout << "Decompression Time: " << (decompression_end_time - decompression_start_time) 
              << "ms\n\n";

    // write decompressed data to file   
    hsize_t zfp_buff_decompressed_dims [2] = {(hsize_t) nx_uncompressed_input, 
                                              (hsize_t)  ny_uncompressed_input};
    write_data_int16(data_decompressed_output, decompressed_file_path, DATASET_NAME, 
                     zfp_buff_decompressed_dims);                 
    free(data_decompressed_output);

    // re-read in the uncompressed data
    int16_t *data_uncompressed = read_data_int16(file_path, DATASET_NAME, uncompressed_input_dims, 
                                                 2);                                                // using int16

    // re-read in decompressed data
    int16_t *data_decompressed = read_data_int16(decompressed_file_path, DATASET_NAME, 
                                                  uncompressed_input_dims, 2);                      // using int16

    // calculate the stats for the uncompressed and decompressed data
    int arr_size = nx_uncompressed_input * ny_uncompressed_input;
    double rmse_val = rmse(data_uncompressed, data_decompressed, arr_size);
    int16_t min_val_uncompressed = min(data_uncompressed, arr_size);
    int16_t min_val_decompressed = min(data_decompressed, arr_size);
    int16_t max_val_uncompressed = max(data_uncompressed, arr_size);
    int16_t max_val_decompressed = max(data_decompressed, arr_size);
    double mean_uncompressed = mean(data_uncompressed, arr_size);
    double mean_decompressed = mean(data_decompressed, arr_size);
    double absolute_mean_uncompressed = absolute_mean(data_uncompressed, arr_size);

    // summary for uncompressed data
    std::cout << "Summary for uncompressed data: \n";
    std::cout << "Min value: " << min_val_uncompressed << "\n";
    std::cout << "Max value: " << max_val_uncompressed << "\n";
    std::cout << "Mean value: " << mean_uncompressed << "\n";
    std::cout << "Mean value: " << absolute_mean_uncompressed << "\n\n";

    // summary for decompressed data
    std::cout << "Summary for decompressed data: \n";
    std::cout << "Min value: " << min_val_decompressed << "\n";
    std::cout << "Max value: " << max_val_decompressed << "\n";
    std::cout << "Mean value: " << mean_decompressed << "\n\n";

    // RMSE
    std::cout << "RMSE of uncompressed data with respect to the decompressed data: " << rmse_val 
              << "\n\n";

    // free allocated memory
    free(data_uncompressed);
    free(data_decompressed);

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

int generate_data(std::string file_path, std::string compressed_file_path, 
                  std::string decompressed_file_path)
{
    uint64_t read_start_time = GetTimeMs64();

    // get input dimensions
    int nx_uncompressed_input = get_input_dim(file_path, DATASET_NAME, 0);                          // ROW_CHANNEL
    int ny_uncompressed_input = get_input_dim(file_path, DATASET_NAME, 1);                          // COL_TIME
    int uncompressed_input_dims [2] = {nx_uncompressed_input, ny_uncompressed_input};

    // read in uncompressed data and fill array
    int16_t *data_uncompressed_input = read_data_int16(file_path, DATASET_NAME, 
                                                       uncompressed_input_dims, 2);                 // using int16
    uint64_t read_end_time = GetTimeMs64();
    uint64_t compression_start_time = GetTimeMs64();

    uint8_t* data_compressed_output;

    // compressed the data stored in data_uncompressed_input
    size_t szsize_compressed = compress(data_uncompressed_input, uncompressed_input_dims, 
                                        data_compressed_output);       
                                                   
    free(data_uncompressed_input);
    uint64_t compression_end_time = GetTimeMs64();
    double compression_ratio = ((double) (nx_uncompressed_input * ny_uncompressed_input * 
                                 sizeof(int16_t))) / ((double) szsize_compressed);                      
    uint64_t write_start_time = GetTimeMs64();

    // write compressed data to file
    hsize_t data_compressed_dims [1] = {szsize_compressed};                                         // buffer dimensions = size of compressed file                                       
    write_data_uint8(data_compressed_output, compressed_file_path, DATASET_NAME, 
               data_compressed_dims);
    uint64_t write_end_time = GetTimeMs64();
    free(data_compressed_output);

    // get input dimensions
    int nx_compressed_input = get_input_dim(compressed_file_path, DATASET_NAME, 0);                 // size of compressed file in bytes         
    int compressed_input_dims [1] = {nx_compressed_input};

    // create and fill uint8 array that contains the 1D compressed data
    uint8_t *data_compressed_input = read_data_uint8(compressed_file_path, DATASET_NAME, 
                                                         compressed_input_dims, 1);

    // initialize output array to store decompressed data
    int16_t *data_decompressed_output;

    uint64_t decompression_start_time = GetTimeMs64();

    int decompressed_output_dims [2] = {nx_uncompressed_input, ny_uncompressed_input};

    // decompress the data
    decompress(data_compressed_input, decompressed_output_dims, data_decompressed_output, 
               nx_compressed_input); 
                                                    
    free(data_compressed_input);
    uint64_t decompression_end_time = GetTimeMs64();

    // write decompressed data to file   
    hsize_t zfp_buff_decompressed_dims [2] = {(hsize_t) nx_uncompressed_input, 
                                              (hsize_t)  ny_uncompressed_input};
    write_data_int16(data_decompressed_output, decompressed_file_path, DATASET_NAME, 
                     zfp_buff_decompressed_dims);                 
    free(data_decompressed_output);

    // re-read in the uncompressed data
    int16_t *data_uncompressed = read_data_int16(file_path, DATASET_NAME, uncompressed_input_dims, 
                                                 2);                                                // using int16

    // re-read in decompressed data
    int16_t *data_decompressed = read_data_int16(decompressed_file_path, DATASET_NAME, 
                                                  uncompressed_input_dims, 2);                      // using int16

    // calculate the stats for the uncompressed and decompressed data
    int arr_size = nx_uncompressed_input * ny_uncompressed_input;
    double rmse_val = rmse(data_uncompressed, data_decompressed, arr_size);
    int16_t min_val_uncompressed = min(data_uncompressed, arr_size);
    int16_t min_val_decompressed = min(data_decompressed, arr_size);
    int16_t max_val_uncompressed = max(data_uncompressed, arr_size);
    int16_t max_val_decompressed = max(data_decompressed, arr_size);
    double mean_uncompressed = mean(data_uncompressed, arr_size);
    double mean_decompressed = mean(data_decompressed, arr_size);
    double absolute_mean_uncompressed = absolute_mean(data_uncompressed, arr_size);

    std::cout << compression_ratio << ",";
    std::cout << rmse_val << ",";
    std::cout << absolute_mean_uncompressed << ",";
    std::cout << min_val_uncompressed << ",";
    std::cout << max_val_uncompressed << ",";
    std::cout << mean_uncompressed << ",";
    std::cout << min_val_decompressed << ",";
    std::cout << max_val_decompressed << ",";
    std::cout << mean_decompressed << ",";
    std::cout << compression_end_time - compression_start_time << ",";
    std::cout << decompression_end_time - decompression_start_time << ",\n";

    // free allocated memory
    free(data_uncompressed);
    free(data_decompressed);

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
    // ensurearguments have been called correctly
    std::string DATA_QUERY;
    if (argc < 6)
    {
        std::cout << "Arguments must be listed as follows:\n[UNCOMPRESSED_FILE_PATH]"
                     "[OUTPUT_DIRECTORY_PATH] [ERROR_BOUND_MODE] [ERROR_BOUND]"
                     "[COMPRESSION_MODE] [-d OPTIONAL (data generation)]\n";
        return -1;
    }
    else if (argc == 7)
    {
        DATA_QUERY = argv[6];
    }
    else
    {
        DATA_QUERY = "NULL";
    }
    
    // initialize arguments
    std::string FILE_PATH = argv[1];
    sz_params sz;
    memset(&sz, 0, sizeof(sz_params));
    std::string ERROR_BOUND_MODE = argv[3];
    std::string ERROR_BOUND = argv[4];
    std::string COMPRESSION_MODE = argv[5];
    
    // using absolute error
    if (strcmp(ERROR_BOUND_MODE.c_str(), "ABS") == 0)
    {
        sz.errorBoundMode = ABS;                                                                    // initialize error mode
        sz.absErrBound = atof(argv[4]);                                                             // initialize error bound
    }
    // using relative error
    else if (strcmp(ERROR_BOUND_MODE.c_str(), "REL") == 0)
    {
        sz.errorBoundMode = REL;                                                                    // initialize error mode
        sz.relBoundRatio = atof(argv[4]);                                                           // initialize error bound
    }
    else
    {
        std::cout << "\nIncorrect error bound mode: Must be ABS or REL\n";
    }

    // using best compression ratio
    if (strcmp(COMPRESSION_MODE.c_str(), "-c") == 0)
    {
        sz.szMode = SZ_BEST_COMPRESSION;                                                            // initialize main compression method
        sz.gzipMode = SZ_BEST_COMPRESSION;                                                          // initialize auxiliary compression method
    }
    // using best compression speed
    else if (strcmp(COMPRESSION_MODE.c_str(), "-s") == 0)
    {
        sz.szMode = SZ_BEST_SPEED;                                                                  // initialize main compression method
        sz.gzipMode = SZ_BEST_SPEED;                                                                // initialize auxiliary compression method
    }
    else
    {
        std::cout << "\nIncorrect compression mode: Must be -c (BEST_COMPRESSION) or "
                     "-s (BEST_SPEED)\n";
    }
    
 // maximum number of quantization bins to use (quantization_intervals must be 0)
    sz.max_quant_intervals = 65536;
 /* This parameter refers to the number of quantization bins. When the quantization_intervals 
    is set to 0, the compressor will search the most appropriate number of quantization bins 
    with the maximum value (max_quant_intervals). */
    sz.quantization_intervals = 0;
    sz.sol_ID = SZ;
    sz.sampleDistance = 100;
    sz.predThreshold = 0.99;
    sz.psnr = 80.0;
    sz.segment_size = 32;
    sz.pwr_type = SZ_PWR_MIN_TYPE;
    int status = SZ_Init_Params(&sz);                                                               // initialize parameters

    // ensure initalization is successful
    if(status == SZ_NSCS)
    {
	    exit(0);
    }
    std::string OUTPUT_DIRECTORY = argv[2];
    std::string COMP_DIRECTORY = OUTPUT_DIRECTORY + "compressed/";
    std::string DECOMP_DIRECTORY = OUTPUT_DIRECTORY + "decompressed/";
    bool DATA_GENERATION;
    if (argc == 7 && DATA_QUERY == "-d")
    {
        DATA_GENERATION = true;
    }
    else
    {
        DATA_GENERATION = false;
    }

    // test compression/decompression on input file 
    size_t path_idx = FILE_PATH.find_last_of("/");
    std::string RAW_FILE_NAME = FILE_PATH.substr(path_idx+1);
    size_t ext_idx = RAW_FILE_NAME.find_first_of(".");
    RAW_FILE_NAME = RAW_FILE_NAME.substr(0, ext_idx); 
    std::string FILE_NAME = FILE_PATH.substr(path_idx+1);
    std::string ERROR_BOUND_STR = ERROR_BOUND;
    std::replace(ERROR_BOUND_STR.begin(), ERROR_BOUND_STR.end(), '.', '_');
    std::string COMPRESSED_FILE_PATH = COMP_DIRECTORY + RAW_FILE_NAME + 
	    			       ERROR_BOUND_MODE + ERROR_BOUND_STR + COMPRESSION_MODE +
				       "_compressed.h5";
    std::string DECOMPRESSED_FILE_PATH = DECOMP_DIRECTORY + RAW_FILE_NAME + 
	    			       ERROR_BOUND_MODE + ERROR_BOUND_STR + COMPRESSION_MODE +
				       "_decompressed.h5";
    if (!DATA_GENERATION)
    {
        std::cout << "\nTesting compression/decompression on file " << FILE_NAME << "\n\n";
        std::cout << "File Path: " << FILE_PATH << "\n";
        std::cout << "File Name: " << RAW_FILE_NAME << "\n";
        std::cout << "Compressed File Path: " << COMPRESSED_FILE_PATH << "\n";\
        std::cout << "Decompressed File Path: " << DECOMPRESSED_FILE_PATH << "\n\n";
        test_compression(FILE_PATH, COMPRESSED_FILE_PATH, DECOMPRESSED_FILE_PATH);
    }
    else
    {
        std::cout << FILE_NAME << ",";
        std::cout << ERROR_BOUND_MODE << ",";
        std::cout << ERROR_BOUND << ",";
        if (strcmp(COMPRESSION_MODE.c_str(), "-c") == 0)
        {
            std::cout << 1 << "," << 0 << ","; 
        }
        else
        {
            std::cout << 0 << "," << 1 << ","; 
        }
        generate_data(FILE_PATH, COMPRESSED_FILE_PATH, DECOMPRESSED_FILE_PATH);
    }
    
    SZ_Finalize();                                                                                  // close compressor/decompressor

    return 0;
}
