import pandas as pd
import scipy.signal as ss
from utils import *

fs = 100             #Sampling Rate
lowcut = 0.1         #Lower bandwith
highcut = 24         #Upper bandwith
DATA_PATH = '../data/resp.txt' #Path to data

df = pd.read_csv(DATA_PATH, sep = '\t', header = None)
df.drop(labels = 2, axis = 1, inplace = True)
df.rename(columns = {0 : 'RSP', 1 : 'Time'}, inplace = True)
df = df.sort_values(['Time'])
df = df[df.Time<=120000.000]

signal = df['RSPinV']
processed_signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order = 1)
peaks = ss.find_peaks(processed_signal)

signal_df = process_dataframe(df, processed_signal, peaks)
bRate_list = list()
for i in range(signal_df.shape[0]-1):
    bRate_list.append(breathing_rate(signal_df.Time.iloc[i], signal_df.Time.iloc[i+1]))
print(f'Breathing rates: {bRate_list}')

rsp_process(signal_df['RSPinV'])