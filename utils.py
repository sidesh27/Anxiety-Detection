import os
import math
import biosppy
import numpy as np
import pandas as pd
import datetime.datetime
from scipy.signal import butter, lfilter
from hrvanalysis import remove_ectopic_beats, interpolate_nan_values

def convert_disp_to_volt(data):
    """
    Converts the PZT (% of displacement of the piezo-electric sensor) to PZT (Voltage)
    Parameters: data (pandas dataframe)
    Returns: pandas series
    """
    return data['RSP']*(3/100)

def butter_bandpass(lowcut, highcut, fs, order):
    """
    Returns the Butterworth bandpass filter coefficients
    Parameters: lowcut (float), highcut (float), fs (float), order (int)
    Returns: b, a (numpy array)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order = 2):
    """
    Applies the butterworth bandpass filter
    Parameters: data (pandas series), lowcut (float), highcut (float), fs (float), order (int)
    Returns: filtered data (pandas series)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y

def process_dataframe(dataframe, processed_signal, peaks):
    """
    Stores the processed signal data in a new dataframe
    Parameters: dataframe (pandas dataframe), processed_signal (pandas series), peaks (numpy array)
    Returns: signal_df (pandas dataframe)
    """
    data = {'Time' : dataframe.Time, 'RSPinV' : processed_signal}
    processed_df = pd.DataFrame(data)
    signal_data = {'Time' : processed_df.Time[peaks[0]], 'RSPinV' : processed_df.RSPinV[peaks[0]]}
    signal_df = pd.DataFrame(signal_data)
    signal_df = signal_df.sort_values(['Time'])
    signal_df["bol"] = signal_df.duplicated(subset=['Time'])
    signal_df.to_csv('../data/resp_processed.csv', sep = '\t', index = False)
    return signal_df

def dateDiff(i_time,j_time):
    """
    Calculates the difference between two datetime objects
    Parameters: i_time (datetime object), j_time (datetime object)
    Returns: difference (float)
    """

    d1 = datetime.strptime(str(round(i_time,3)), "%H%M%S.%f")
    d2 = datetime.strptime(str(round(j_time,3)), "%H%M%S.%f")
    return((d1-d2).total_seconds())

def breathing_rate(start, end):
    """
    Calculates the breathing rate
    Parameters: start (datetime object), end (datetime object)
    Returns: breathing rate (float)
    """
    time_diff = dateDiff(end,start)
    return (1/time_diff)

def rsp_process(rsp, sampling_rate = 100):
    """
    Stores the processed respiration signal
    Parameters: rsp (pandas series), sampling_rate (int)
    Returns: None
    """
    processed_rsp = {'df' : pd.DataFrame({'RSP_Raw' : np.array(rsp)})}
    biosppy_rsp = dict(biosppy.signals.resp.resp(rsp, sampling_rate = sampling_rate, show = False))
    processed_rsp["df"]["RSP_Filtered"] = biosppy_rsp["filtered"]
    rsp_rate = biosppy_rsp['resp_rate']*60
    rsp_times = biosppy_rsp['resp_rate_ts']
    rsp_times = np.round(rsp_times*sampling_rate).astype(float)
    rsp_rate = interpolate(rsp_rate, rsp_times, sampling_rate)
    processed_rsp['df']['RSP_Rate'] = rsp_rate
    processed_rsp.to_csv('../data/resp_dataframe.csv', sep = '\t', index = False)

def extract_peaks():
    '''
    Extracts peaks from the ecg dataframe using Pan-Tompkins algorithm
    Parameters: None
    Returns: peaks (pandas dataframe)
    '''
    qrs_df = pd.read_csv(os.path.join("logs", os.listdir("logs")[-1]))
    peaks = qrs_df[(qrs_df["qrs_detected"] == 1)]
    peaks = peaks.sort_values(['timestamp'])
    return(peaks)

def split_time(i_time):
    '''
    Splits the time into hours, minutes, seconds and milliseconds   
    Parameters: i_time (string)
    Returns: hours, minutes, seconds, milliseconds (int)
    '''
    frac, _ = math.modf(i_time)
    hours = (int(i_time / 10000) + int(int(int(i_time % 10000) / 100) / 60))
    mins = (int(int((i_time % 100) / 60) + int(int(i_time % 10000) / 100)) % 60)
    secs = int((i_time % 100) % 60)
    ms = frac
    return(hours, mins, secs, ms)

def adjust_time(i_time):
    '''
    Adjusts the time to be in milliseconds
    Parameters: i_time (string)
    Returns: time in milliseconds (int)
    '''
    hours, mins, secs, ms = split_time(i_time)
    if(int(int((i_time % 100) / 60) + int(int(i_time % 10000) / 100)) == 60): 
        hours += 1
    return((hours * 10000 + mins * 100 + secs ) + ms)

def RR_interval(i_time,j_time):
    '''
    Calculates the RR interval between two consecutive peaks in seconds
    Parameters: i_time (string), j_time (string)
    Returns: RR interval in seconds (int)
    '''
    d1 = datetime.strptime(str(round(i_time, 3)), "%H%M%S.%f")
    d2 = datetime.strptime(str(round(j_time, 3)), "%H%M%S.%f")
    return((d1 - d2).total_seconds())

def extract_hr(peaks):
    '''
    Extracts heart rate from the peaks dataframe
    Parameters: peaks (pandas dataframe)
    Returns: peaks (pandas dataframe)
    '''
    bpm = []
    rri = []
    previous_timestamp = []
    for i in range(0, len(peaks.index) - 1):
        RR = RR_interval(peaks["timestamp"][peaks["timestamp"].index[i + 1]],peaks["timestamp"][peaks["timestamp"].index[i]])
        bpm.append(60 / RR)
        rri.append(RR)
        previous_timestamp.append(peaks["timestamp"][peaks["timestamp"].index[i]])
    bpm.insert(0, 0)
    rri.insert(0, 0)
    previous_timestamp.insert(0, 0)
    peaks["HR"] = bpm
    peaks["RR"] = rri
    peaks["p_timestamp"] = previous_timestamp
    peaks = peaks[["timestamp", "p_timestamp", "ecg_measurement", "RR", "HR"]]
    return(peaks)

def extract_NNI(peaks):
    '''
    Extracts Normal-to-Normal Intervals (NNI) from the peaks dataframe
    Parameters: peaks (pandas dataframe)
    Returns: peaks (pandas dataframe)
    '''
    nn_intervals_list = []
    nn_intervals_list = remove_ectopic_beats(rr_intervals = peaks["RR"][1:].tolist(), method = "malik")
    NNI_list = interpolate_nan_values(rr_intervals = nn_intervals_list)
    NNI_list.insert(0, 0)
    peaks["NNI"] = NNI_list
    return(peaks)

def cal_bio_mean(peaks, exposure_period_df):
    '''
    Calculates the mean heart rate for the biofeedback period
    Parameters: peaks (pandas dataframe), exposure_period_df (pandas dataframe)
    Returns: mean_biofeedback_hr (float)
    '''
    bio_df = peaks.loc[(peaks["timestamp"] >= exposure_period_df["s_time"].iloc[-1]) & (peaks["timestamp"] <= exposure_period_df["e_time"].iloc[-1])].copy()
    start_bio = exposure_period_df["s_time"].iloc[-1]
    end_bio = exposure_period_df["e_time"].iloc[-1]
    s_time  = e_time  = mean_hr = list()
    i_time = start_bio
    while(i_time < end_bio and adjust_time(i_time + 10) <= end_bio):
        i_time = adjust_time(i_time)
        j_time = adjust_time(i_time + 10)
        bio_slice = bio_df.loc[(bio_df["timestamp"] >= i_time) & (bio_df["timestamp"] <= j_time)].copy()
        s_time.append(i_time)
        e_time.append(j_time)
        mean_hr.append(bio_slice["HR"][1:].mean())
        i_time = i_time + 10
    mean_hr_dic = {"s_time" : s_time, "e_time" : e_time , "mean_hr" : mean_hr}
    bio_mean = pd.DataFrame(mean_hr_dic)
    mean_biofeedback_hr = bio_mean["mean_hr"].mean()
    return(mean_biofeedback_hr)

def basic_features(peaks, exposure_period_df):
    '''
    Calculates the basic HRV features of the patient
    Parameters: peaks (pandas dataframe), exposure_period_df (pandas dataframe)
    Returns: valid_peaks (pandas dataframe)
    '''
    valid_peaks = list()   #LIST OF DFs FOR EACH VIDEO CONTAINING PEAKS, NNI and HR

    #FINDING FIRST DIFFERENCE OF HRs and NNIs

    for j in range(1, 18):
        FD = []
        NNI_FD = []    
        valid = peaks.loc[(peaks["timestamp"] >= exposure_period_df["s_time"][j]) & (peaks["timestamp"] <= exposure_period_df["e_time"][j])].copy()
        for i in range(0, len(valid.index) - 1):
            f_diff = abs(valid["HR"][valid.index[i + 1]] - valid["HR"][valid.index[i]])
            f_diff_nn = abs(valid["NNI"][valid.index[i + 1]] - valid["NNI"][valid.index[i]])
            FD.append(f_diff)
            NNI_FD.append(f_diff_nn)
        FD.insert(0, 0)
        NNI_FD.insert(0, 0)
        valid.insert(0, "event", [exposure_period_df["event"][j]] * len(valid))
        valid["NNI_FD"] = NNI_FD
        valid["FD"] = FD
        valid_peaks.append(valid)

    #FINDING SECOND DIFFERENCE OF HRs   

    for j in range(17):
        SD = []
        valid = valid_peaks[j]
        for i in range(0, len(valid.index) - 2):
            s_diff = abs(valid["HR"][valid.index[i + 2]]-valid["HR"][valid.index[i]])
            SD.append(s_diff)
        SD.insert(0, 0)
        SD.insert(1, 0)
        valid["SD"] = SD
    return(valid_peaks)

def rmsValue(arr):
    '''
    Calculates the root mean square value of the successive differences between normal heartbeats
    Parameters: arr (list)
    Returns: rmssd (float)
    '''
    square = 0
    mean = 0.0
    rmssd = 0.0
    arr = arr.tolist()
    for i in range(0,len(arr)): 
        square += (arr[i] ** 2) 
    mean = (square / (float)(len(arr)-1)) 
    rmssd = math.sqrt(mean) 
    return rmssd

def adv_features(peaks, exposure_period_df):
    '''
    Calculates the advanced HRV features of the patient
    Parameters: peaks (pandas dataframe), exposure_period_df (pandas dataframe)
    Returns: final_df (pandas dataframe)
    '''
    video_mean_df = list()
    mean_biofeedback = cal_bio_mean(peaks, exposure_period_df)
    valid_peaks = basic_features(peaks, exposure_period_df)
    for i in range(17):
        window = valid_peaks[i]
        start_bio = exposure_period_df["s_time"][i + 1]
        end_bio = exposure_period_df["e_time"][i + 1]
        s_time = e_time = mean_hr = std_hr = NFD = NSD = avNN = sdNN = HRV = RMSSD = NN50 = pNN50 = pNN20 = event = window_list = list()
        i_time = adjust_time(start_bio)
        k = 1
        while(i_time < end_bio and adjust_time(i_time + 10) <= end_bio and len(window.loc[(window["timestamp"] >= i_time) & (window["timestamp"] <= adjust_time(i_time + 10))]) > 0):
            j_time = adjust_time(i_time + 10)
            window_slice = window.loc[(window["timestamp"] >= i_time) & (window["timestamp"] <= j_time)].copy()
            window_slice["HR"] = abs(window_slice["HR"] - mean_biofeedback)
            event.append(window_slice["event"][window_slice.index[0]])
            window_list.append(k)
            s_time.append(i_time)
            e_time.append(j_time)
            mean_hr.append(window_slice["HR"].mean())
            std_hr.append(window_slice["HR"].std(ddof = 1))
            NFD.append(window_slice["FD"][1:].mean())
            NSD.append(window_slice["SD"][2:].mean())
            avNN.append(window_slice["NNI"].mean())
            sdNN.append(window_slice["NNI"].std(ddof = 1))
            HRV.append(window_slice["NNI_FD"][1:].mean())
            RMSSD.append(rmsValue(window_slice["NNI_FD"][1:])) 
            NN50.append(len(window_slice[window_slice["NNI_FD"] > 0.05]))
            pNN50.append(((len(window_slice[window_slice["NNI_FD"] > 0.05]) + 1) / len(window_slice)))
            pNN20.append(((len(window_slice[window_slice["NNI_FD"] > 0.02]) + 1) / len(window_slice)))
            i_time = adjust_time(i_time + 10)
            k += 1
        mean_hr_dic = {"event" : event, "window" : window_list , "s_time" : s_time, "e_time" : e_time , "mean_hr" : mean_hr, "std" : std_hr, "NFD" : NFD, "NSD" : NSD, "HRV" : HRV, "avNN" : avNN, "sdNN" : sdNN, "RMSSD" : RMSSD, "NN50" : NN50, "pNN50" : pNN50, "pNN20" : pNN20}
        video_mean = pd.DataFrame(mean_hr_dic)
        video_mean_df.append(video_mean)
    final_df = pd.concat(video_mean_df)
    return(final_df)