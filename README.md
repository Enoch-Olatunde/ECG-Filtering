# Biomedical Signal & Image Processing - 3 Project Portfolio

A hands-on portfolio covering ECG signal filtering, medical tabular classification, and medical image edge detection. Built with Python, Octave, SciPy, scikit-learn, and OpenCV.

> Dataset sources: PhysioNet MIT-BIH Arrhythmia Database, UCI Breast Cancer Wisconsin, NIH ChestX-ray

---

### 01 - ECG Filtering | `01-ecg-filtering/`
**Goal:** Clean noisy ECG and estimate heart rate.

**Pipeline:**
1. Load record `100` from MIT-BIH via `wfdb`
2. Plot raw ECG
3. Apply 0.5-40 Hz Butterworth bandpass (`scipy.signal.butter + filtfilt`)
4. Apply 50 Hz IIR notch filter for mains interference
5. Detect R-peaks via threshold + refractory period
6. Compute HR = (peaks / duration) * 60

**Result:**
- `before_after.png` - Raw vs Filtered
- Example output: `Detected R-peaks: 12, Heart Rate: 72.3 bpm`

**Run:**
```bash
pip install wfdb scipy matplotlib numpy
python ecg_filter.py
