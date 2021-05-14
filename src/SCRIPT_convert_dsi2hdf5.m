
%% Convert several Sac dsi files into hdf5 

addpath('/global/cscratch1/sd/alexpo7/media/sf_das_compression/src');

%files = dir('/bear0-data/vrtribal/SacDarkFiber/10chMedianStack_2kmOffset/*.mat');

files = dir('/global/project/projectdirs/m1248/DAS_ML_trainingDataset/30min_files_NoTrain/*.mat');

h5_dset_name='DataCT';
test_type_str = 'int16';
transpose_flag = 1;

%h5_path = '/bear0-data/vrtribal/SacDarkFiber/hdf5_conversions'; 
h5_path = '/global/project/projectdirs/m1248/alex/media/sf_das_compression/data/30min'; 

for k = 1:size(files,1);
	
	disp('working on')
	files(k).name

	filename = fullfile('/global/project/projectdirs/m1248/DAS_ML_trainingDataset/30min_files_NoTrain/',files(k).name);

	[filepath,name,ext] = fileparts(filename);
	
	dsi_file = fullfile('/global/project/projectdirs/m1248/DAS_ML_trainingDataset/30min_files_NoTrain/',files(k).name);
	
	h5_name = sprintf('%s.h5',name);
	h5_file = fullfile(h5_path,h5_name);

	dsi2h5(dsi_file, h5_file , h5_dset_name, test_type_str, transpose_flag);

		
end


