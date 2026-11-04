# ECG FILTERING - ENOCH OLATUNDE

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# 1. Download record 100 from MIT-BIH
record_name = '100'
# downloads to current folder if not exists
wfdb.dl_database('mitdb', './mitdb_data/', records=[record_name])
record = wfdb.rdrecord(f'./mitdb_data/{record_name}', sampto=3600) # 10 sec @ 360Hz
signal = record.p_signal[:,0]
fs = record.fs
print(f"Loaded: {record_name}, fs={fs} Hz, {len(signal)} samples")

# 2. Filters
def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return b, a

def apply_filters(data, fs):
    # 0.5-40 Hz bandpass
    b, a = butter_bandpass(0.5, 40, fs, order=4)
    filtered = filtfilt(b, a, data)
    # 50 Hz notch
    b_notch, a_notch = iirnotch(50, Q=30, fs=fs)
    filtered = filtfilt(b_notch, a_notch, filtered)
    return filtered

filtered = apply_filters(signal, fs)

# 3. Simple R-peak detection (threshold)
def detect_r_peaks(sig, fs):
    # simple: peaks above mean + 0.6*std, min distance 0.6*fs
    threshold = np.mean(sig) + 0.6*np.std(sig)
    peaks = []
    min_dist = int(0.6*fs)
    last = -min_dist
    for i in range(1, len(sig)-1):
        if sig[i] > sig[i-1] and sig[i] > sig[i+1] and sig[i] > threshold:
            if i - last > min_dist:
                peaks.append(i)
                last = i
    return np.array(peaks)

peaks = detect_r_peaks(filtered, fs)
hr = len(peaks) / (len(signal)/fs) * 60
print(f"Detected R-peaks: {len(peaks)}, Heart Rate: {hr:.1f} bpm")

# 4. Plot before/after
time = np.arange(len(signal))/fs
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(time, signal)
plt.title('Raw ECG - Record 100')
plt.ylabel('mV')
plt.subplot(2,1,2)
plt.plot(time, filtered)
plt.plot(time[peaks], filtered[peaks], 'ro', label='R-peaks')
plt.title(f'Filtered 0.5-40Hz + 50Hz Notch | HR: {hr:.1f} bpm')
plt.xlabel('Time [s]')
plt.ylabel('mV')
plt.legend()
plt.tight_layout()
plt.savefig('before_after.png', dpi=300)
plt.show()
