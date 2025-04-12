% Re-structure and save EEG data as a 3D matrix format (numver_of_trials, number_of_samples,number_of_bands)
% and assign binary labels to movement/non-movement intervals
%_________________________________
% Run the initialization script
run('initialize_environment.m')
rng(42); %set random seed for reproducibility
load('EEG_balanced_subtrl_cls_lat.mat')%Preprocessed EEG ICs activity and EMG latencies data for group analysis, with
% equal number of trials per subject
% Prepare factors
time_vec=times;%time vector
sampling_rate =512;
new_sampling_rate = 128;
factor = sampling_rate/new_sampling_rate; %downsampling factor
%% Initialize variables
subID_data = cell(1,num_cls);
data_labels = cell(1,num_cls);
dwn_smp_data_labels = cell(1,num_cls);
signal_pwr_data= cell(1,num_cls);% to save normalized power of bands of each cluster
nrm_pwr_data= cell(1,num_cls); % to save normalized power of bands of each cluster
dwn_smp_pwr_data = cell(1,num_cls); % to save downsampled power
nrm_dwn_smp_pwr_data = cell(1,num_cls); % to save downsampled normalized power
emg_onset_idx_data=cell(1,num_cls);
dwn_smp_emg_onset_idx_data=cell(1,num_cls);
dwn_smp_time_vec_window=cell(1,num_cls);
%% Loop over clusters to save the eeg power data in proper format for further process
%and to add labels. 
for cls=1:num_cls
    % Extract the subject ID and put them all inside an array
    subID_data{cls}= cls_subs_lat_ersp(cls).SubInd;

    % Extract the EMG onset latency of the subjects inside the cluster and put
    %them in an array
    emg_onsets_cls =cls_subs_lat_ersp(cls).SubsEMGonset;
    pwr_cls=cls_subs_lat_ersp(cls).Subs; % power bands data of this cluster
    power_data= permute(pwr_cls, [3, 2, 1]);  % convert to (total_trials, num_samples, num_bands)
    num_all_trls=length(emg_onsets_cls);% Extract number of total trials of all the ICs in this brain area

    % Find the range of data to consider from each trial
    mean_emgonset=mean(emg_onsets_cls);
    t_start=mean_emgonset-1; % start 1 s before mean emg onset of cluster
    t_end=mean_emgonset+1; % end 1 s after mean emg onset   
    %Extract indexes of the target data segment
    idx_start=find(time_vec>= t_start *  1000 ,1);
    idx_end=find(time_vec>= t_end * 1000 , 1);
    idxs = idx_start+1:idx_end;
    windowed_data=power_data( : ,  idxs,  : );
    time_vec_window=time_vec(idxs);% time vector of the selected window
    dwn_smp_time_vec_window{cls}=downsample(time_vec_window,factor);
    dwn_smp_time_vec=downsample(time_vec,factor);
    emg_onset_idx = zeros(num_all_trls,1);
    dwn_smp_cls_data=[];%to save balanced downsampled data
    labels=zeros(num_all_trls, length(idxs)); %initiate labels
    dwn_smp_labels=[];
    for trl=1:num_all_trls
        dwn_smp_cls_data(trl,:,:)=downsample( windowed_data(trl, : , :) , factor);%down sample selected segment
        emg_onset_idx(trl)=find(time_vec_window>=emg_onsets_cls(trl)*1000,1);% Extract latency of current trial
        labels(trl,  emg_onset_idx(trl) : end)=1; %set post EMG onset labels as 1
        dwn_smp_labels(trl,:)=downsample(labels(trl,:), factor); % down sample labels
        emg_onset_idx_ds=find(dwn_smp_time_vec>=emg_onsets_cls(trl)*1000,1)-find(dwn_smp_time_vec>=2000,1);
        % Plot to check it is working well!
        % figure
        % subplot(211);plot(normalize(cls_data_balanced(trl, : ,1),2));hold on,plot( labels(trl,:));
        % subplot(212);plot(normalize(cls_data_downsampled(trl, : , 1),2));hold on,plot( labels_ds(trl,:));
    end
    % Save the clusters data in cell format
    data_labels{cls}=labels;
    dwn_smp_data_labels{cls}=dwn_smp_labels;
    signal_pwr_data{cls}=windowed_data;
    nrm_pwr_data{cls}= normalize(windowed_data,2);
    dwn_smp_pwr_data{cls} =dwn_smp_cls_data;
    nrm_dwn_smp_pwr_data{cls}= normalize(dwn_smp_cls_data,2);
    emg_onset_idx_data{cls}=emg_onset_idx-idx_start;
    dwn_smp_emg_onset_idx_data{cls}=emg_onset_idx_ds;
end
%
cd(results_path) % switch to the destination path to save resulrs
save('EEG_semibalanced_binary_labeled_ds.mat',"time_vec_window",'dwn_smp_time_vec_window',...
    "subID_data", 'data_labels','dwn_smp_data_labels','signal_pwr_data','nrm_pwr_data','dwn_smp_pwr_data','nrm_dwn_smp_pwr_data',...
    'dwn_smp_emg_onset_idx_data',"emg_onset_idx_data",'-v7');
