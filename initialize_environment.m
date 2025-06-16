
% Set plot values
set(groot, 'DefaultAxesFontSize', 14, 'DefaultLineLineWidth', 2, 'DefaultTextFontSize', 14);

% Add data and code paths
file_path = 'C:\Users\...';
results_path = 'C:\Users\...';
code_path = 'C:\Users\...';
addpath(code_path,file_path, results_path);

% Load initial data
load("MS_cls_info.mat"); % brain clusters and subjects indexes

% Set brain areas
clusters = [3, 4, 7, 9, 10];
clusters_name = {'RVA', 'LMS', 'V', 'RMS', 'LVA'};
num_cls = length(clusters);

% Set frequency bands
frq_bands = {7:12, 14:24, 28:48};
band_names = {'alpha', 'beta', 'gamma'};
num_bands = length(band_names);

% Set task phases
task_phases = {'baseline', 'biphase', 'post-onset'};
num_phase = length(task_phases);
