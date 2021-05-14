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
#include <sys/stat.h>
#include <stdio.h>
#include "hdf5.h"

std::string DATASET_NAME("Xcorr"); // 2D dataset, where rows(X) correspond to channels and columns(Y) correspond to time

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
    hid_t group = H5Gopen(file, "/Xcorr", H5P_DEFAULT);
    hid_t dataset = H5Dopen(group, "Xcorr", H5P_DEFAULT);
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

float* read_data_float(std::string file_name, int* dims, int rank)
{
    // open the file and dataset using default params
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t group = H5Gopen(file, "/Xcorr", H5P_DEFAULT);
    hid_t dataset = H5Dopen(group, "Xcorr", H5P_DEFAULT);

    // calculate the product of the dimension sizes
    int malloc_size = 1;
    for (int i = 0; i < rank; i++)
    {
        malloc_size = malloc_size * dims[i];
    }

    // allocated memory to store the read data
    float *data = (float *)malloc(sizeof(float) * malloc_size);

    // read in the data and store it to the array
    int status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    assert(status >= 0);                                                                            // ensure the read executed successfully

    // close the dataset and file
    status = H5Dclose(dataset);
    status = H5Fclose(file);

    return data;
}

/*
FUNCTION: rmse
DESCRIPTION:
    Calculates the root mean square error (RMSE) between two int32 arrays
INPUT: 
    int32_t* arr1, int32_t* arr2: arrays to calculate the rmse between
    int arr_size: size of the arrays
OUTPUT:
    RETURNS: RMSE
*/

double rmse(float* arr1, float* arr2, int arr_size)
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
    Calculate the minimum of the given int32 array
INPUT: 
    int32_t* arr: input array
    int arr_size: size of the array
OUTPUT:
    RETURNS: minumum value of arr
*/

float min(float* arr, int arr_size) 
{
    float min_val = 0;

    for (int i=0; i<arr_size; i++)
    {
        float curr_val = arr[i];
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
    Calculate the maximum of the given int32 array
INPUT: 
    int32_t* arr: input array
    int arr_size: size of the array
OUTPUT:
    RETURNS: maximum value of arr
*/

float max(float* arr, int arr_size) 
{
    float max_val = 0;
    
    for (int i=0; i<arr_size; i++)
    {
        float curr_val = arr[i];
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
    Calculate the mean of the given int32 array
INPUT: 
    int32_t* arr: input array
    int arr_size: size of the array
OUTPUT:
    RETURNS: mean value of arr
*/

double mean(float* arr, int arr_size) 
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
    Calculate the mean of the absolute values stored in given int32 array
INPUT: 
    int32_t* arr: input array
    int arr_size: size of the array
OUTPUT:
    RETURNS: mean value of arr
*/

double absolute_mean(float* arr, int arr_size) 
{
    double sum = 0;
    
    for (int i=0; i<arr_size; i++)
    {
        sum += abs(arr[i]);
    }

    double mean = sum / arr_size;

    return mean;
}

/*
FUNCTION: min_diff
DESCRIPTION:
    Calculate the minimum difference between two int16 arrays
INPUT: 
    int16_t* arr1: input array 1
    int16_t* arr1: input array 2
    int arr_size1: size of array 1
    int arr_size2: size of array 2
OUTPUT:
    RETURNS: minumum difference between the arrays
*/

double min_diff(double* arr1, double* arr2, int arr_size1, int arr_size2) 
{
    double min_diff_val = 1000;

    for (int i = 0; i < arr_size1; i++)
    {
        double curr_val1 = arr1[i];
        for (int j = 0; j < arr_size2; j++)
        {
            double curr_val2 = arr2[j];
            double diff_val = abs(curr_val1 - curr_val2);
            if (diff_val < min_diff_val)
            {
                min_diff_val = diff_val;
            }
        }
    }

    return min_diff_val;
}

/*
FUNCTION: max_diff
DESCRIPTION:
    Calculate the maximum difference between two int16 arrays
INPUT: 
    int16_t* arr1: input array 1
    int16_t* arr1: input array 2
    int arr_size1: size of array 1
    int arr_size2: size of array 2
OUTPUT:
    RETURNS: maxumum difference between the arrays
*/

double max_diff(double* arr1, double* arr2, int arr_size1, int arr_size2) 
{
    double max_diff_val = 0;

    for (int i = 0; i < arr_size1; i++)
    {
        double curr_val1 = arr1[i];
        for (int j = 0; j < arr_size2; j++)
        {
            double curr_val2 = arr2[j];
            double diff_val = abs(curr_val1 - curr_val2);
            if (diff_val > max_diff_val)
            {
                max_diff_val = diff_val;
            }
        }
    }

    return max_diff_val;
}

/* Returns a vector of file names stored in dir_str_p
 * 
 * 
 * CREDIT : Bin Dong, dbin@lbl.gov
 * 
 * */

std::vector<std::string> GetDirFileList(std::string dir_str_p)
{
    struct dirent **namelist;
    int namelist_length = scandir(dir_str_p.c_str(), &namelist, 0, alphasort);
    std::vector<std::string> file_list;
    int temp_index = 0;
    while (temp_index < namelist_length)
    {
        if (strcmp(namelist[temp_index]->d_name, "..") != 0 && strcmp(namelist[temp_index]->d_name, ".") != 0)
            file_list.push_back(namelist[temp_index]->d_name);
        temp_index++;
    }
    return file_list;
}

inline bool file_exists(std::string name)
{
    FILE *file;
    if (file = fopen(name.c_str(), "r")) 
    {
        fclose(file);
        return true;
    } 
    else 
    {
        return false;
    }
}

int compare_rmse(std::string file_path, std::string decompressed_file_path)
{
    // get input dimensions for uncompressed xcorr file
    int nx_uncompressed_input = get_input_dim(file_path, DATASET_NAME, 0);                          // ROW_CHANNEL
    int ny_uncompressed_input = get_input_dim(file_path, DATASET_NAME, 1);                          // COL_TIME
    int uncompressed_input_dims [2] = {nx_uncompressed_input, ny_uncompressed_input};

    // get input dimensions
    int nx_decompressed_input = get_input_dim(decompressed_file_path, DATASET_NAME, 0);                          // ROW_CHANNEL
    int ny_decompressed_input = get_input_dim(decompressed_file_path, DATASET_NAME, 1);                          // COL_TIME
    int decompressed_input_dims [2] = {nx_decompressed_input, ny_decompressed_input};

    // read in the uncompressed xcorr data
    float *data_uncompressed = read_data_float(file_path, uncompressed_input_dims, 
                                                 2);                                                

    // read in decompressed xcorr data
    float *data_decompressed = read_data_float(decompressed_file_path, uncompressed_input_dims, 2);                      

    int arr_size_uncomp = nx_uncompressed_input * ny_uncompressed_input;
    int arr_size_decomp = nx_decompressed_input * ny_decompressed_input;
    assert(arr_size_uncomp == arr_size_decomp);

    double rmse_val = rmse(data_uncompressed, data_decompressed, arr_size_uncomp);

    // RMSE
    std::cout << "RMSE of uncompressed xcorr data with respect to the decompressed xcorr data: " << rmse_val 
              << "\n\n";
    free(data_uncompressed);
    free(data_decompressed);

    return 0;
}

void generate_data(std::string file_path, std::string decompressed_file_path)
{
    // get input dimensions for uncompressed xcorr file
    int nx_uncompressed_input = get_input_dim(file_path, DATASET_NAME, 0);                          // ROW_CHANNEL
    int ny_uncompressed_input = get_input_dim(file_path, DATASET_NAME, 1);                          // COL_TIME
    int uncompressed_input_dims [2] = {nx_uncompressed_input, ny_uncompressed_input};

    // get input dimensions
    int nx_decompressed_input = get_input_dim(decompressed_file_path, DATASET_NAME, 0);                          // ROW_CHANNEL
    int ny_decompressed_input = get_input_dim(decompressed_file_path, DATASET_NAME, 1);                          // COL_TIME
    int decompressed_input_dims [2] = {nx_decompressed_input, ny_decompressed_input};

    // read in the uncompressed xcorr data
    float *data_uncompressed = read_data_float(file_path, uncompressed_input_dims, 2);                                                // using int16

    // read in decompressed xcorr data
    float *data_decompressed = read_data_float(decompressed_file_path, uncompressed_input_dims, 2);                      // using int16

    int arr_size_uncomp = nx_uncompressed_input * ny_uncompressed_input;
    int arr_size_decomp = nx_decompressed_input * ny_decompressed_input;
    assert(arr_size_uncomp == arr_size_decomp);

    double rmse_val = rmse(data_uncompressed, data_decompressed, arr_size_uncomp);
    float min_val_uncompressed = min(data_uncompressed, arr_size_uncomp);
    float min_val_decompressed = min(data_decompressed, arr_size_uncomp);
    float max_val_uncompressed = max(data_uncompressed, arr_size_uncomp);
    float max_val_decompressed = max(data_decompressed, arr_size_uncomp);
    double mean_uncompressed = mean(data_uncompressed, arr_size_uncomp);
    double mean_decompressed = mean(data_decompressed, arr_size_uncomp);
    double absolute_mean_uncompressed = absolute_mean(data_uncompressed, arr_size_uncomp);

    std::cout << rmse_val << ",";
    std::cout << absolute_mean_uncompressed << ",";
    std::cout << min_val_uncompressed << ",";
    std::cout << max_val_uncompressed << ",";
    std::cout << mean_uncompressed << ",";
    std::cout << min_val_decompressed << ",";
    std::cout << max_val_decompressed << ",";
    std::cout << mean_decompressed << ",\n";

    free(data_uncompressed);
    free(data_decompressed);
}

int main(int argc, char *argv[])
{
    std::string XCORR_DIRECTORY = argv[1];

    std::string DATA_QUERY;
    if (argc == 3)
    {
        DATA_QUERY = argv[2];
    }
    else
    {
        DATA_QUERY = "NULL";
    }

    bool DATA_GENERATION;
    if (argc == 3 && DATA_QUERY == "-d")
    {
        DATA_GENERATION = true;
    }
    else
    {
        DATA_GENERATION = false;
    }

    // retrieve iterator of file names
    std::vector<std::string> FILE_LIST = GetDirFileList(XCORR_DIRECTORY);

    for (std::string FILE_NAME : FILE_LIST)
    {
        std::string FILE_PATH = XCORR_DIRECTORY + FILE_NAME;
        size_t firstindex = FILE_NAME.find_first_of("."); 
        std::string RAW_FILE_NAME = FILE_NAME.substr(0, firstindex); 
        std::regex r("(.*)(decompressed)(.*)");

        if (!std::regex_match(FILE_NAME, r))
        {
            if (!DATA_GENERATION)
            {
                std::string DECOMPRESSED_FILE_PATH = XCORR_DIRECTORY + RAW_FILE_NAME + "REL";
                std::cout << "\nCalculating RMSE between uncompressed and decompressed xcorr files using best compression...\n";
                for (double err : {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1})
                {
                    std::string PARAM_TEXT = std::to_string(err);
                    std::regex s("0+$");
                    PARAM_TEXT = std::regex_replace(PARAM_TEXT, s, "");
                    std::replace(PARAM_TEXT.begin(), PARAM_TEXT.end(), '.', '_');
                    std::string DECOMPRESSED_FILE_PATH_CMP = DECOMPRESSED_FILE_PATH + PARAM_TEXT + "-c_decompressed.tdms_xcorr.h5";
                    
                    if (file_exists(DECOMPRESSED_FILE_PATH_CMP))
                    {
                        compare_rmse(FILE_PATH, DECOMPRESSED_FILE_PATH_CMP);
                    }
                }
                std::cout << "\nCalculating RMSE between uncompressed and decompressed xcorr files using best speed...\n";
                for (double err : {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1})
                {
                    std::string PARAM_TEXT = std::to_string(err);
                    std::regex s("0+$");
                    PARAM_TEXT = std::regex_replace(PARAM_TEXT, s, "");
                    std::replace(PARAM_TEXT.begin(), PARAM_TEXT.end(), '.', '_');
                    std::string DECOMPRESSED_FILE_PATH_SPD = DECOMPRESSED_FILE_PATH + PARAM_TEXT + "-s_decompressed.tdms_xcorr.h5";

                    if (file_exists(DECOMPRESSED_FILE_PATH_SPD))
                    {
                        compare_rmse(FILE_PATH, DECOMPRESSED_FILE_PATH_SPD);
                    }
                }
            }
            else
            {
                std::string DECOMPRESSED_FILE_PATH = XCORR_DIRECTORY + RAW_FILE_NAME + "REL";
		//std::cout << "\n" << DECOMPRESSED_FILE_PATH << "\n";
                for (double err : {0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009})
                {
                    std::string PARAM_TEXT = std::to_string(err);
                    std::regex s("0+$");
                    PARAM_TEXT = std::regex_replace(PARAM_TEXT, s, "");
                    std::replace(PARAM_TEXT.begin(), PARAM_TEXT.end(), '.', '_');
                    std::string DECOMPRESSED_FILE_PATH_CMP = DECOMPRESSED_FILE_PATH + PARAM_TEXT + "-c_decompressed.tdms_xcorr.h5";
                    if (file_exists(DECOMPRESSED_FILE_PATH_CMP))
                    {
                        std::cout << FILE_NAME << ",";
                        std::cout << "REL" << ",";
                        std::cout << err << ",";
                        std::cout << 1 << "," << 0 << ",";
                        generate_data(FILE_PATH, DECOMPRESSED_FILE_PATH_CMP);
                    }
                }
                for (double err : {0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009})
                {
                    std::string PARAM_TEXT = std::to_string(err);
                    std::regex s("0+$");
                    PARAM_TEXT = std::regex_replace(PARAM_TEXT, s, "");
                    std::replace(PARAM_TEXT.begin(), PARAM_TEXT.end(), '.', '_');
                    std::string DECOMPRESSED_FILE_PATH_SPD = DECOMPRESSED_FILE_PATH + PARAM_TEXT + "-s_decompressed.tdms_xcorr.h5";;

                    if (file_exists(DECOMPRESSED_FILE_PATH_SPD))
                    {
                        std::cout << FILE_NAME << ",";
                        std::cout << "REL" << ",";
                        std::cout << err << ",";
                        std::cout << 0 << "," << 1 << ",";
                        generate_data(FILE_PATH, DECOMPRESSED_FILE_PATH_SPD);
                    }
                }
            }
        }
    }
}
