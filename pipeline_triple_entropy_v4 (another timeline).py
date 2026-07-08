"""
=============================================================================
PIPELINE: EEG Seizure Prediction — v4 (FIX: leakage split + aliasing +
          labeling boundary + per-file clinical eval + event-level metrics)
=============================================================================

v3 đã sửa bug thr_off=0 (alarm kẹt ON vĩnh viễn). Rà soát sâu hơn phát hiện
5 vấn đề KHÁC, nghiêm trọng hơn về giá trị khoa học của kết quả báo cáo:

CÁC FIX TRONG v4:

  1. [LEAKAGE NGHIÊM TRỌNG] train_test_split() chia theo TỪNG WINDOW ngẫu
     nhiên, nhưng windows chồng lấp 50% (STEP_SIZE=5s/WINDOW_SIZE=10s) →
     windows liền kề (gần như trùng dữ liệu thô) rơi vào cả train và test.
     FIX: group_train_val_test_split() theo file_id — mỗi file/recording
     chỉ thuộc đúng 1 split duy nhất (GroupShuffleSplit).

  2. [SAI METRIC LÂM SÀNG] DualThresholdVoter có state tuần tự (hysteresis +
     suppression) nhưng bị chạy trên test set là 1 SUBSET ngẫu nhiên ghép
     nhiều file/seizure khác nhau → state rò qua biên file, Sensitivity/FAR/
     Latency không phản ánh đúng hành vi hệ thống trên 1 ca theo dõi liên tục.
     FIX: compute_clinical_metrics_grouped() — chạy Voter riêng từng file,
     gộp kết quả sau.

  3. [SAI ĐỊNH NGHĨA SENSITIVITY/LATENCY] code v3 tính Sensitivity theo
     WINDOW (TP/FN từng window) dù docstring nói "theo event" — sai lệch so
     với chuẩn lâm sàng (% cơn giật được phát hiện). Latency cũ cũng chỉ lấy
     ĐÚNG 1 giá trị cho toàn bộ chuỗi dù có nhiều cơn giật.
     FIX: tính theo EVENT (mỗi đoạn pre-ictal liên tục = 1 cơn giật), latency
     lấy riêng cho từng event rồi mới median/mean.

  4. [MISLABEL BIÊN ONSET] label_windows_regression(): cửa sổ chứa đúng thời
     điểm seizure bắt đầu (t <= sz_start < t+WINDOW_SIZE) không khớp điều
     kiện "current" (cần sz_start<=t) cũng không khớp "upcoming" (cần
     sz_start>=we) → rơi qua nhánh else, bị gán nhãn 0.0 (interictal) dù
     chứa dữ liệu ictal thật. FIX: thêm điều kiện onset-trong-cửa-sổ vào
     "current" để loại (-1) đúng.

  5. [ALIASING] transform() downsample bằng slicing thô x[::4] — không có
     anti-alias filter, khi dải Gamma lọc tới 70Hz nhưng 256Hz/4=64Hz
     (Nyquist 32Hz) → năng lượng cao tần gập ngược vào các băng thấp.
     FIX: scipy.signal.decimate (có low-pass trước downsample) + giảm
     DOWNSAMPLE_FACTOR 4→2.

  6. [DEAD CONFIG / CRASH ẨN] SFREQ_TARGET khai báo nhưng không dùng → file
     có sfreq khác nhau sẽ tạo windows shape khác nhau → crash khi ghi h5.
     FIX: resample về SFREQ_TARGET ngay trong load_edf().

  7. [TỐI ƯU] DualThresholdVoter._vote_ratio() O(N*n) lồng vòng lặp → cumsum
     vectorized O(N). make_generator() mở lại h5py.File mỗi batch → mở 1
     lần/epoch. Balancing dùng RandomState seed cố định (reproducible).

Các phần model/loss/entropy/kiến trúc giữ nguyên — không đổi logic train,
chỉ sửa data pipeline (split/label/resample) + evaluate + post-processing.
=============================================================================
"""

import os, re, gc, json, mne, h5py
import numpy as np
import tensorflow as tf
from scipy.signal import decimate
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from tensorflow.keras.layers import (
    Dense, Dropout, Bidirectional, LSTM, LayerNormalization,
    Conv1D, MaxPooling1D, Activation, Concatenate, Input,
    BatchNormalization, Add, GlobalAveragePooling1D, MultiHeadAttention,
    Reshape, Layer
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import antropy as ant
    _ANTROPY_OK = True
except ImportError:
    _ANTROPY_OK = False
    print("[WARNING] 'antropy' chưa cài. pip install antropy")

# =========================================================================
# CONFIG
# =========================================================================
TARGET_PATIENT    = "chb01"
DATASET_PATH      = r"C:\Users\Admin\Downloads\EEG_Project\EEG dataset"
HDF5_PATH         = f"chbmit_{TARGET_PATIENT}_v4.h5"
# FIX v4: đổi tên file cache (v2.h5 → v4.h5) — schema h5 đã đổi (thêm
# "file_id", dữ liệu đã resample về SFREQ_TARGET, nhãn đã sửa boundary
# onset). Nếu giữ tên cũ, entry point sẽ "bỏ qua preprocessing" và dùng
# nhầm file h5 cũ thiếu "file_id" → crash ở bước split.

WINDOW_SIZE       = 10      # giây
STEP_SIZE         = 5       # giây
PRE_ICTAL         = 1800    # 30 phút
POST_ICTAL        = 30      # giây

SFREQ_TARGET      = 256
BATCH_SIZE        = 64
# FIX v4 — BUG: factor=4 ở 256Hz cho ra 64Hz sau downsample (Nyquist=32Hz),
# nhưng băng Gamma lọc tới 70Hz → vi phạm Nyquist. transform() trước đây
# downsample bằng slicing thô (x[::4]) KHÔNG có anti-alias filter → năng
# lượng tần số cao bị "gập" (alias) ngược vào toàn bộ băng thấp (Delta/
# Theta/Alpha), làm nhiễu chính các đặc trưng quan trọng nhất. Hạ factor
# xuống 2 (Nyquist=64Hz, vẫn cắt một phần đỉnh Gamma nhưng nhẹ hơn nhiều)
# và chuyển sang scipy.signal.decimate (có anti-alias filter) ở transform().
DOWNSAMPLE_FACTOR = 2

PERM_ORDER        = 3
PERM_DELAY        = 1

EPOCHS            = 60
LABEL_SMOOTH      = 0.03
MIXUP_ALPHA       = 0.2
N_TTA             = 5

# --- Sensitivity target (lấy ngưỡng tối ưu dựa theo đây) ---
SENSITIVITY_TARGET = 0.85   # ràng buộc tối thiểu khi grid-search ngưỡng

# --- SlidingWindowVoter: K trong N windows liên tiếp (lớp lọc 1) ---
# QUAN TRỌNG: N/K=8/5 đã được xác nhận là quá chặt, khiến mọi threshold
# "hợp lý" (0.3-0.7) đều không đạt Sensitivity floor → grid-search bị đẩy
# về biên. Nới về 5/3 làm giá trị mặc định, có thể grid-search lại nếu cần.
VOTER_N           = 5
VOTER_K           = 3

# --- Dual-threshold / Schmitt trigger (lớp lọc 2 — chống flickering) ---
THR_ON            = 0.35    # giá trị mặc định — sẽ bị override bởi grid-search
THR_OFF           = 0.15    # giá trị mặc định — sẽ bị override bởi grid-search

# --- FIX BUG QUAN TRỌNG: thr_off=0.00 khiến alarm KHÔNG BAO GIỜ tắt
# (điều kiện "scores[i] < thr_off" không bao giờ true khi thr_off=0 và
# scores luôn >= 0 từ sigmoid) → 1 alarm event kéo dài tới hết file →
# Sensitivity giả tạo =1.0, FAR giả tạo thấp, Latency giả tạo ~0.
# Đặt sàn dương cho thr_off để loại trừ hoàn toàn case này khỏi grid.
THR_OFF_MIN       = 0.02

# --- Alarm suppression: không đếm thêm FA trong khoảng này sau 1 alarm ---
SUPPRESSION_S     = 1800    # giây (30 phút, = PRE_ICTAL)

# --- Asymmetric focal weights ---
ALPHA_FN          = 3.0     # penalty FN (bỏ sót cơn giật) — tăng Sensitivity
ALPHA_FP          = 1.0     # penalty FP (báo giả) — kiểm soát FAR

BANDS = [
    (0.5,  4),   # Delta
    (4,    8),   # Theta
    (8,   13),   # Alpha
    (13,  30),   # Beta
    (30,  70),   # Gamma
]

# =========================================================================
# 1. PARSE SUMMARY
# =========================================================================
def parse_summary(summary_path):
    seizure_map = defaultdict(list)
    if not os.path.exists(summary_path):
        return seizure_map
    with open(summary_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    for block in re.split(r'(?=File Name:)', content):
        fn_m = re.search(r'File Name:\s*(\S+\.edf)', block)
        if not fn_m: continue
        fname  = fn_m.group(1).strip()
        starts = re.findall(r'Seizure(?:\s+\d+)?\s+Start\s+Time:\s+(\d+)\s+second', block, re.IGNORECASE)
        ends   = re.findall(r'Seizure(?:\s+\d+)?\s+End\s+Time:\s+(\d+)\s+second',   block, re.IGNORECASE)
        for s, e in zip(starts, ends):
            seizure_map[fname].append((int(s), int(e)))
    return seizure_map

def load_all_seizure_maps(dataset_path):
    combined = {}
    for root, _, files in os.walk(dataset_path):
        if TARGET_PATIENT not in os.path.basename(root): continue
        for f in files:
            if re.match(r'chb\d+-summary\.txt', f, re.IGNORECASE):
                combined.update(parse_summary(os.path.join(root, f)))
    return combined

# =========================================================================
# 2. LOAD EDF
# =========================================================================
def load_edf(edf_path):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None
    seen = {}; to_drop = []
    for ch in raw.ch_names:
        base = re.sub(r'-\d+$', '', ch)
        if base in seen: to_drop.append(ch)
        else: seen[base] = ch
    if to_drop: raw.drop_channels(to_drop)
    raw.filter(0.5, 70.0, verbose=False, n_jobs=-1)
    raw.notch_filter(freqs=[50, 60], verbose=False, n_jobs=-1)
    # FIX v4: SFREQ_TARGET trước đây bị khai báo nhưng KHÔNG dùng → các file
    # EDF có sfreq gốc khác nhau (hiếm nhưng có thể xảy ra) sẽ tạo windows có
    # win_s khác nhau → crash khi ghi vào h5 dataset "X" (shape cố định lấy
    # từ file đầu tiên). Resample về SFREQ_TARGET ngay tại đây để đảm bảo
    # mọi file luôn đồng nhất sfreq trước khi windowing.
    if abs(raw.info['sfreq'] - SFREQ_TARGET) > 1e-6:
        raw.resample(SFREQ_TARGET, verbose=False, n_jobs=-1)
    data = raw.get_data() * 1e6; sfreq = raw.info['sfreq']
    del raw; gc.collect()
    return data.astype(np.float32), float(sfreq), list(seen.keys())

# =========================================================================
# 3. WINDOWING
# =========================================================================
def make_windows(data, sfreq):
    ws = int(WINDOW_SIZE * sfreq); ss = int(STEP_SIZE * sfreq)
    starts  = np.arange(0, data.shape[1] - ws + 1, ss)
    windows = np.stack([data[:, s : s + ws] for s in starts]).astype(np.float32)
    return windows, (starts / sfreq).astype(np.float32)

# =========================================================================
# 4. LABELING — lưu thêm start_time để tính Detection Latency
# =========================================================================
def label_windows_regression(start_times, seizure_times):
    """
    Trả về (labels, tts_arr):
      labels   : urgency score [0,1], -1 = ictal/post-ictal (loại)
      tts_arr  : time-to-seizure (giây), nan nếu không có seizure sắp tới
    """
    labels  = np.zeros(len(start_times), dtype=np.float32)
    tts_arr = np.full(len(start_times), np.nan, dtype=np.float32)

    for i, t in enumerate(start_times):
        we       = t + WINDOW_SIZE
        upcoming = [sz for sz in seizure_times if sz[0] >= we]
        # FIX v4 — BUG: cửa sổ "bao trùm" thời điểm seizure bắt đầu (t <= sz[0] < we)
        # trước đây KHÔNG khớp "current" (cần sz[0] <= t) và KHÔNG khớp "upcoming"
        # (cần sz[0] >= we) → rơi qua nhánh else, giữ label=0.0 mặc định (interictal)
        # dù cửa sổ chứa 1 phần dữ liệu ictal thật → nhiễm nhãn âm giả ngay tại biên
        # onset, đúng vùng quan trọng nhất để model học. Thêm điều kiện onset-trong-
        # cửa-sổ vào "current" để loại (-1) thay vì gán sai 0.0.
        current  = [sz for sz in seizure_times if sz[0] <= t < sz[1] or t <= sz[0] < we]
        post     = [sz for sz in seizure_times if sz[1] <= t < sz[1] + POST_ICTAL]
        if current or post:
            labels[i] = -1.0
        elif upcoming:
            tts = upcoming[0][0] - we
            tts_arr[i] = tts
            if tts <= PRE_ICTAL:
                raw_score = 1.0 - tts / PRE_ICTAL
                labels[i] = raw_score * (1 - LABEL_SMOOTH) + LABEL_SMOOTH * 0.5
            else:
                labels[i] = 0.0
    return labels, tts_arr

# =========================================================================
# 5. MULTI-BAND & NORMALIZE
# =========================================================================
def multiband_single_window(window, sfreq):
    return np.stack([
        mne.filter.filter_data(window.astype(np.float64), sfreq,
                               l_freq=lo, h_freq=hi, method='fir', verbose=False).astype(np.float32)
        for lo, hi in BANDS
    ], axis=0)

def multiband_batch(windows, sfreq):
    return np.array([multiband_single_window(w, sfreq) for w in windows], dtype=np.float32)

def normalize_batch(batch):
    m = batch.mean(axis=3, keepdims=True)
    s = batch.std( axis=3, keepdims=True)
    return ((batch - m) / (s + 1e-6)).astype(np.float32)

# =========================================================================
# 6. ENTROPY FUNCTIONS
# =========================================================================
def _safe_sample_entropy(sig, order=2):
    try:
        v = ant.sample_entropy(sig, order=order)
        return float(v) if np.isfinite(v) else 0.0
    except Exception: return 0.0

def _safe_spectral_entropy(sig, sfreq):
    try:
        v = ant.spectral_entropy(sig, sf=sfreq, method='fft', normalize=True)
        return float(v) if np.isfinite(v) else 0.0
    except Exception: return 0.0

def _safe_perm_entropy(sig, order=PERM_ORDER, delay=PERM_DELAY):
    try:
        v = ant.perm_entropy(sig, order=order, delay=delay, normalize=True)
        return float(v) if np.isfinite(v) else 0.0
    except Exception: return 0.0

def _compute_entropy_one_window(args):
    window, sfreq = args
    n_ch = window.shape[0]
    SE  = np.zeros(n_ch, dtype=np.float32)
    SPE = np.zeros(n_ch, dtype=np.float32)
    PE  = np.zeros(n_ch, dtype=np.float32)
    if _ANTROPY_OK:
        for c in range(n_ch):
            sig    = window[c]
            SE[c]  = _safe_sample_entropy(sig)
            SPE[c] = _safe_spectral_entropy(sig, sfreq)
            PE[c]  = _safe_perm_entropy(sig)
    return SE, SPE, PE

def entropy_batch_parallel(windows, sfreq, max_workers=4):
    n = len(windows); n_ch = windows[0].shape[0]
    SE_out  = np.zeros((n, n_ch), dtype=np.float32)
    SPE_out = np.zeros((n, n_ch), dtype=np.float32)
    PE_out  = np.zeros((n, n_ch), dtype=np.float32)
    args_list = [(windows[i], sfreq) for i in range(n)]
    if n < 8 or max_workers <= 1:
        for i, args in enumerate(args_list):
            SE_out[i], SPE_out[i], PE_out[i] = _compute_entropy_one_window(args)
    else:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = {exe.submit(_compute_entropy_one_window, a): i for i, a in enumerate(args_list)}
                for fut in as_completed(futures):
                    i = futures[fut]
                    SE_out[i], SPE_out[i], PE_out[i] = fut.result()
        except Exception:
            for i, args in enumerate(args_list):
                SE_out[i], SPE_out[i], PE_out[i] = _compute_entropy_one_window(args)
    return SE_out, SPE_out, PE_out

def normalize_entropy_arr(E):
    med = np.median(E, axis=0, keepdims=True)
    iqr = np.percentile(E, 75, axis=0, keepdims=True) - np.percentile(E, 25, axis=0, keepdims=True)
    return ((E - med) / (iqr + 1e-6)).astype(np.float32)

# =========================================================================
# 7. PREPROCESSING — lưu thêm "t" (start_time) và "tts" (time-to-seizure)
# =========================================================================
def run_preprocessing():
    print("=" * 70)
    print(f"PREPROCESSING v4 — {TARGET_PATIENT}")
    print("=" * 70)
    seizure_map = load_all_seizure_maps(DATASET_PATH)
    edf_files = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(DATASET_PATH)
        if TARGET_PATIENT in os.path.basename(root)
        for f in files if f.endswith('.edf')
    ])
    if not edf_files:
        raise FileNotFoundError(f"Không tìm thấy EDF cho {TARGET_PATIENT}")

    sd, ss, _ = load_edf(edf_files[0])
    n_ch  = sd.shape[0]
    win_s = int(WINDOW_SIZE * ss)
    del sd; gc.collect()

    with h5py.File(HDF5_PATH, "w") as h5f:
        shape0 = (0, len(BANDS), n_ch, win_s)
        h5f.create_dataset("X",   shape=shape0,   maxshape=(None, len(BANDS), n_ch, win_s), dtype="float32", chunks=True)
        h5f.create_dataset("SE",  shape=(0, n_ch), maxshape=(None, n_ch), dtype="float32", chunks=True)
        h5f.create_dataset("SPE", shape=(0, n_ch), maxshape=(None, n_ch), dtype="float32", chunks=True)
        h5f.create_dataset("PE",  shape=(0, n_ch), maxshape=(None, n_ch), dtype="float32", chunks=True)
        h5f.create_dataset("y",   shape=(0,), maxshape=(None,), dtype="float32", chunks=True)
        h5f.create_dataset("t",   shape=(0,), maxshape=(None,), dtype="float32", chunks=True)   # start_time
        h5f.create_dataset("tts", shape=(0,), maxshape=(None,), dtype="float32", chunks=True)   # time-to-seizure
        # FIX v4 — THÊM file_id: bắt buộc để (a) split train/val/test theo
        # GROUP (toàn bộ 1 file về cùng 1 split, không random theo window)
        # tránh leakage do windows chồng lấp 50% (STEP_SIZE=5s/WINDOW_SIZE=10s),
        # và (b) chạy DualThresholdVoter LIÊN TỤC trong từng file riêng biệt
        # khi evaluate, tránh trộn lẫn state/suppression giữa các file/seizure
        # khác nhau khi tính Sensitivity/FAR/Latency.
        h5f.create_dataset("file_id", shape=(0,), maxshape=(None,), dtype="int32", chunks=True)
        h5f.attrs.update({"n_channels": n_ch, "sfreq": ss, "perm_order": PERM_ORDER, "perm_delay": PERM_DELAY})

        written = 0
        file_names = []
        for edf_path in tqdm(edf_files, desc="EDF files"):
            res = load_edf(edf_path)
            if res is None: continue
            data, sfreq, _ = res
            windows, st = make_windows(data, sfreq); del data; gc.collect()
            if not len(windows): continue

            labels, tts_arr = label_windows_regression(st, seizure_map.get(os.path.basename(edf_path), []))
            mask    = labels >= 0.0
            windows = windows[mask]; labels = labels[mask]
            tts_arr = tts_arr[mask]; st_arr  = st[mask]
            if not len(windows): continue

            SE, SPE, PE = entropy_batch_parallel(windows, sfreq, max_workers=4)
            Xm = normalize_batch(multiband_batch(windows, sfreq))

            file_id = len(file_names)
            file_names.append(os.path.basename(edf_path))
            fid_arr = np.full(len(Xm), file_id, dtype=np.int32)

            nt = written + len(Xm)
            for key in ["X", "SE", "SPE", "PE", "y", "t", "tts", "file_id"]:
                h5f[key].resize(nt, axis=0)
            h5f["X"][written:nt]       = Xm
            h5f["SE"][written:nt]      = SE
            h5f["SPE"][written:nt]     = SPE
            h5f["PE"][written:nt]      = PE
            h5f["y"][written:nt]       = labels
            h5f["t"][written:nt]       = st_arr
            h5f["tts"][written:nt]     = tts_arr
            h5f["file_id"][written:nt] = fid_arr
            written = nt
            del Xm, SE, SPE, PE, labels; gc.collect()

        h5f.attrs["file_names"] = json.dumps(file_names)

    # Robust normalization entropy
    with h5py.File(HDF5_PATH, "a") as h5f:
        for key in ["SE", "SPE", "PE"]:
            arr = h5f[key][:]
            h5f[key][:] = normalize_entropy_arr(arr)
    print(f"[INFO] Preprocessing hoàn tất. Tổng windows: {written}")

# =========================================================================
# 8. AUGMENTATION
# =========================================================================
def augment_eeg_window(x):
    x = x.copy(); op = np.random.randint(0, 4)
    if op == 0:   x += np.random.normal(0, 0.01 * (np.std(x) + 1e-6), x.shape).astype(np.float32)
    elif op == 1: x *= np.random.uniform(0.85, 1.15)
    elif op == 2: x  = np.roll(x, np.random.randint(-8, 8), axis=2)
    elif op == 3:
        flip_mask = np.random.rand(x.shape[1]) < 0.25
        x[:, flip_mask, :] *= -1
    return x.astype(np.float32)

def mixup_entropy(E1, E2, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    return (lam * E1 + (1 - lam) * E2).astype(np.float32)

def transform(x):
    # FIX v4: dùng scipy.signal.decimate (có low-pass anti-alias filter trước
    # khi hạ sample rate) thay vì slicing thô x[::N] — slicing không lọc nên
    # năng lượng tần số > Nyquist mới bị "gập" ngược vào các băng thấp.
    x_ds = decimate(x, DOWNSAMPLE_FACTOR, axis=2, ftype='fir', zero_phase=True) if DOWNSAMPLE_FACTOR > 1 else x
    return x_ds.transpose(2, 0, 1).reshape(x_ds.shape[2], x_ds.shape[0] * x_ds.shape[1])

# =========================================================================
# 9. GENERATOR
# =========================================================================
def make_generator(idx_arr, h5_path, shuffle=True, augment=False):
    idx_arr = np.array(idx_arr)
    def gen():
        order = np.random.permutation(len(idx_arr)) if shuffle else np.arange(len(idx_arr))
        # FIX v4 (tối ưu): mở h5py.File 1 LẦN cho cả epoch, thay vì mở/đóng lại
        # mỗi batch (overhead I/O đáng kể khi số batch lớn, không đổi kết quả).
        with h5py.File(h5_path, "r") as f:
            for start in range(0, len(order), BATCH_SIZE):
                raw_idx = idx_arr[order[start : start + BATCH_SIZE]]
                uid, inv = np.unique(raw_idx, return_inverse=True)
                Xb   = f["X"][uid][inv]; SEb  = f["SE"][uid][inv]
                SPEb = f["SPE"][uid][inv]; PEb = f["PE"][uid][inv]; yb = f["y"][uid][inv]

                if augment:
                    pre_idx = np.where(yb > 0.0)[0]
                    for i in pre_idx:
                        Xb[i] = augment_eeg_window(Xb[i])
                        j = pre_idx[np.random.randint(len(pre_idx))]
                        if j != i:
                            SEb[i]  = mixup_entropy(SEb[i],  SEb[j])
                            SPEb[i] = mixup_entropy(SPEb[i], SPEb[j])
                            PEb[i]  = mixup_entropy(PEb[i],  PEb[j])
                        noise = np.random.normal(0, 0.02, SEb[i].shape).astype(np.float32)
                        SEb[i] += noise; SPEb[i] += noise; PEb[i] += noise

                Xout = np.stack([transform(x) for x in Xb]).astype(np.float32)
                yield {
                    "temporal_input": Xout,
                    "se_input":  SEb.astype(np.float32),
                    "spe_input": SPEb.astype(np.float32),
                    "pe_input":  PEb.astype(np.float32),
                }, yb.astype(np.float32)
    return gen

# =========================================================================
# 10. MODEL — v2: LearnableEntropyScale thay hardcode weights
# =========================================================================

class LearnableEntropyScale(Layer):
    """
    Học trọng số cho 3 entropy thay vì hard-code.
    Khởi tạo với prior 0.45/0.35/0.20, softmax để tổng = 1.
    """
    def __init__(self, init_weights=(0.45, 0.35, 0.20), **kwargs):
        super().__init__(**kwargs)
        self.init_logits = np.log(np.array(init_weights, dtype=np.float32) + 1e-8)

    def build(self, input_shape):
        self.logits = self.add_weight(
            name="entropy_logits", shape=(3,),
            initializer=tf.constant_initializer(self.init_logits),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        se, spe, pe = inputs
        w = tf.nn.softmax(self.logits)   # [w_se, w_spe, w_pe], sum=1
        return se * w[0], spe * w[1], pe * w[2]

    def get_weights_softmax(self):
        return tf.nn.softmax(self.logits).numpy()


def entropy_residual_block(x, units_hidden=128, units_out=32, name_prefix=""):
    shortcut = Dense(units_out, name=f"{name_prefix}_shortcut")(x)
    shortcut = BatchNormalization(name=f"{name_prefix}_bn_sc")(shortcut)

    h = Dense(units_hidden, kernel_initializer='he_normal', name=f"{name_prefix}_fc1")(x)
    h = BatchNormalization(name=f"{name_prefix}_bn1")(h)
    h = Activation('relu', name=f"{name_prefix}_relu1")(h)
    h = Dropout(0.25, name=f"{name_prefix}_drop1")(h)
    h = Dense(units_out, kernel_initializer='he_normal', name=f"{name_prefix}_fc2")(h)
    h = BatchNormalization(name=f"{name_prefix}_bn2")(h)

    out = Add(name=f"{name_prefix}_add")([h, shortcut])
    out = Activation('relu', name=f"{name_prefix}_relu2")(out)
    return out


def cross_entropy_attention_fusion(se, spe, pe, n_ch, d_model=32, n_heads=2):
    se_r  = Reshape((1, n_ch), name="se_reshape")(se)
    spe_r = Reshape((1, n_ch), name="spe_reshape")(spe)
    pe_r  = Reshape((1, n_ch), name="pe_reshape")(pe)
    stack = Concatenate(axis=1, name="ent_stack")([se_r, spe_r, pe_r])

    stack_proj = Dense(d_model, name="ent_proj")(stack)
    attn_out = MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model // n_heads, name="cross_ent_attn"
    )(stack_proj, stack_proj)

    attn_out = Add(name="attn_add")([stack_proj, attn_out])
    attn_out = LayerNormalization(name="attn_ln")(attn_out)

    fused = GlobalAveragePooling1D(name="ent_gap")(attn_out)
    fused = Dense(48, activation='relu', kernel_initializer='he_normal', name="ent_fusion_dense")(fused)
    fused = Dropout(0.2, name="ent_fusion_drop")(fused)
    return fused


def asymmetric_focal_huber_loss(delta=0.5, gamma=1.5, alpha_fn=ALPHA_FN, alpha_fp=ALPHA_FP):
    """
    Asymmetric Focal Huber Loss:
      - alpha_fn >> alpha_fp: phạt FN nặng hơn FP
        → model ưu tiên không bỏ sót (Sensitivity ↑, FAR có thể tăng nhẹ)
      - Sau đó dùng SlidingWindowVoter để kiểm soát FAR
    """
    def loss(y_true, y_pred):
        h = tf.keras.losses.huber(y_true, y_pred, delta=delta)
        focal_weight = tf.pow(tf.abs(y_true - y_pred), gamma)

        # Phân loại FN / FP
        fn_mask = tf.cast(y_true > y_pred, tf.float32)   # under-predict (bỏ sót)
        fp_mask = 1.0 - fn_mask                           # over-predict (báo giả)
        asym_weight = alpha_fn * fn_mask + alpha_fp * fp_mask

        return tf.reduce_mean(asym_weight * focal_weight * h)
    loss.__name__ = "asym_focal_huber"
    return loss


def build_model(time_ds, feat_size, n_ch):
    # Branch 1: Temporal CNN-BiLSTM
    temporal_input = Input(shape=(time_ds, feat_size), name="temporal_input")
    t = Conv1D(64, 7, strides=2, padding='same', kernel_initializer='he_normal', name="conv1")(temporal_input)
    t = BatchNormalization(name="bn_conv1")(t); t = Activation('relu')(t)
    t = MaxPooling1D(2, name="pool1")(t); t = Dropout(0.25)(t)
    t = Conv1D(128, 5, padding='same', kernel_initializer='he_normal', name="conv2")(t)
    t = BatchNormalization(name="bn_conv2")(t); t = Activation('relu')(t)
    t = Conv1D(128, 3, padding='same', kernel_initializer='he_normal', name="conv3")(t)
    t = BatchNormalization(name="bn_conv3")(t); t = Activation('relu')(t)
    t = MaxPooling1D(2, name="pool2")(t); t = Dropout(0.3)(t)
    t = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1), name="bilstm1")(t)
    t = LayerNormalization(name="ln_lstm1")(t)
    t = Bidirectional(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.1), name="bilstm2")(t)
    t = LayerNormalization(name="ln_lstm2")(t)
    temporal_out = Dense(64, activation='relu', kernel_initializer='he_normal', name="temporal_proj")(t)
    temporal_out = Dropout(0.3)(temporal_out)

    # Branch 2-4: Entropy với LearnableEntropyScale
    se_input  = Input(shape=(n_ch,), name="se_input")
    spe_input = Input(shape=(n_ch,), name="spe_input")
    pe_input  = Input(shape=(n_ch,), name="pe_input")

    se_out  = entropy_residual_block(se_input,  128, 32, name_prefix="se")
    spe_out = entropy_residual_block(spe_input, 128, 32, name_prefix="spe")
    pe_out  = entropy_residual_block(pe_input,  128, 32, name_prefix="pe")

    # Learnable scale thay hardcode
    scale_layer = LearnableEntropyScale(init_weights=(0.45, 0.35, 0.20), name="entropy_scale")
    se_scaled, spe_scaled, pe_scaled = scale_layer([se_out, spe_out, pe_out])

    # Cross-Entropy Attention
    entropy_fused = cross_entropy_attention_fusion(se_scaled, spe_scaled, pe_scaled, n_ch=32, d_model=32, n_heads=2)

    # Final Merge
    merged = Concatenate(name="final_merge")([temporal_out, entropy_fused])
    merged = Dense(96, activation='relu', kernel_initializer='he_normal', name="head_fc1")(merged)
    merged = BatchNormalization(name="head_bn1")(merged); merged = Dropout(0.3)(merged)
    merged = Dense(48, activation='relu', kernel_initializer='he_normal', name="head_fc2")(merged)
    merged = Dropout(0.2)(merged)
    outputs = Dense(1, activation='sigmoid', name="urgency_score")(merged)

    model = tf.keras.Model(
        inputs=[temporal_input, se_input, spe_input, pe_input],
        outputs=outputs,
        name="QuadInput_v2_AsymFocal"
    )
    return model

# =========================================================================
# 11. POST-PROCESSING: DualThresholdVoter (Schmitt Trigger + N/K Voter + Suppression)
# =========================================================================
class DualThresholdVoter:
    """
    Bộ lọc 3 lớp để giảm FAR, áp dụng theo thứ tự:

    LỚP 1 — N/K Voting (lọc nhiễu tức thời):
        Tại mỗi điểm thời gian, tính tỷ lệ windows vượt thr_on trong
        N windows gần nhất. Chỉ "đủ điều kiện" nếu ≥ K/N windows dương.
        → Loại bỏ các spike đơn lẻ do artifact/noise tức thời.

    LỚP 2 — Schmitt Trigger / Hysteresis (chống flickering):
        - Khi đang ở trạng thái OFF: cần điều kiện Lớp 1 đạt thr_on mới
          chuyển sang ON.
        - Khi đang ở trạng thái ON: chỉ chuyển lại OFF khi score tức thời
          tụt xuống dưới thr_off (thấp hơn thr_on).
        → Một khi đã alarm, không bị "rung" tắt/mở liên tục quanh ngưỡng
          đơn — đây là nguồn FAR lớn nhất trong hệ thống ngưỡng đơn.

    LỚP 3 — Suppression Window (chống đếm trùng cùng 1 episode):
        Sau khi 1 alarm event KẾT THÚC (chuyển ON→OFF), khóa không cho
        alarm mới trong SUPPRESSION_S giây tiếp theo.
        → Một episode báo động chỉ được tính 1 lần, không bị đếm thành
          nhiều FA liên tiếp trong cùng giai đoạn bất thường ngắn.

    Tham số:
        n, k        : N/K voting window (Lớp 1)
        thr_on      : ngưỡng kích hoạt (Lớp 2)
        thr_off     : ngưỡng tắt, phải < thr_on (Lớp 2)
        suppression_steps : số bước (windows) khóa sau mỗi alarm (Lớp 3),
                             tính từ STEP_SIZE: suppression_s / step_size
    """
    def __init__(self, n=VOTER_N, k=VOTER_K, thr_on=THR_ON, thr_off=THR_OFF,
                 suppression_s=SUPPRESSION_S, step_size=STEP_SIZE):
        assert thr_off < thr_on, "thr_off phải nhỏ hơn thr_on (hysteresis band)"
        assert thr_off >= THR_OFF_MIN, (
            f"thr_off={thr_off} < THR_OFF_MIN={THR_OFF_MIN}: với thr_off quá nhỏ "
            f"(đặc biệt =0), điều kiện tắt alarm 'scores[i] < thr_off' gần như "
            f"không bao giờ đúng vì scores ra từ sigmoid luôn >=0 → alarm bị kẹt "
            f"ON đến hết file, làm sai lệch hoàn toàn Sensitivity/FAR/Latency."
        )
        self.n = n; self.k = k
        self.thr_on = thr_on; self.thr_off = thr_off
        self.suppression_steps = max(1, int(round(suppression_s / step_size)))

    def _vote_ratio(self, scores):
        """Lớp 1: tỷ lệ windows >= thr_on trong N cửa sổ gần nhất.
        FIX v4 (tối ưu): thay vòng lặp O(N*n) bằng cumsum O(N) — không đổi
        kết quả, chỉ nhanh hơn đáng kể khi N (số windows) lớn."""
        pos = (np.asarray(scores) >= self.thr_on).astype(np.int32)
        csum = np.concatenate(([0], np.cumsum(pos)))
        idx = np.arange(len(scores))
        start = np.maximum(0, idx - self.n + 1)
        n_pos = csum[idx + 1] - csum[start]
        return n_pos >= self.k

    def predict(self, scores):
        """
        Trả về alarm_state (0/1) cho mỗi window, đã áp dụng đủ 3 lớp lọc.
        """
        vote_ok = self._vote_ratio(scores)   # Lớp 1
        alarm = np.zeros(len(scores), dtype=int)

        state = 0           # 0=OFF, 1=ON
        suppress_until = -1  # index, -1 = không bị khóa

        for i in range(len(scores)):
            if state == 0:
                # Đang OFF: có thể chuyển ON nếu vote đạt VÀ không bị suppression
                if vote_ok[i] and i > suppress_until:
                    state = 1
            else:
                # Đang ON: chỉ tắt khi score tức thời < thr_off (Lớp 2)
                if scores[i] < self.thr_off:
                    state = 0
                    suppress_until = i + self.suppression_steps   # Lớp 3

            alarm[i] = state
        return alarm

    def count_alarm_events(self, alarm):
        """Đếm số episode alarm riêng biệt (1 chuỗi liên tiếp ON = 1 event)."""
        events = 0; in_alarm = False
        for v in alarm:
            if v == 1 and not in_alarm:
                events += 1; in_alarm = True
            elif v == 0:
                in_alarm = False
        return events

# =========================================================================
# 12. CLINICAL METRICS (Sensitivity, FAR/h, Detection Latency)
# =========================================================================
def _find_runs(mask):
    """Trả về list (start, end) các đoạn chỉ số liên tục có mask=True (end exclusive).
    Dùng để xác định ranh giới từng 'episode pre-ictal' (1 cơn giật sắp tới)
    trong 1 chuỗi windows liên tục theo thời gian."""
    runs, start = [], None
    mask = np.asarray(mask)
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def compute_clinical_metrics(y_true, y_pred_scores, tts_arr, thr_on, thr_off,
                              window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                              pre_ictal=PRE_ICTAL,
                              n=VOTER_N, k=VOTER_K, suppression_s=SUPPRESSION_S):
    """
    Tính 3 chỉ số lâm sàng trên 1 CHUỖI LIÊN TỤC DUY NHẤT (đúng 1 file/
    recording, theo đúng thứ tự thời gian gốc).

    ⚠️ FIX v4: hàm này giờ chỉ nên gọi cho dữ liệu của ĐÚNG 1 file liên tục.
    Gọi trực tiếp trên dữ liệu ghép nhiều file/seizure (ví dụ 1 test-set
    subsample ngẫu nhiên) sẽ làm rò state hysteresis/suppression qua biên
    file → dùng compute_clinical_metrics_grouped() cho trường hợp đó.

    1. Sensitivity = tỷ lệ EVENT (cơn giật) có ≥1 alarm trong vùng pre-ictal
       dẫn tới nó — KHÔNG còn tính theo window như bản v3 (bug: docstring
       v3 nói "TP: cơn giật có ≥1 alarm" nhưng code lại đếm theo window,
       2 định nghĩa khác nhau → Sensitivity v3 không phải số liệu lâm sàng
       chuẩn). Mỗi event = 1 đoạn liên tục windows có label>0.

    2. FAR (False Alarm Rate) [alarm-event/giờ]:
       = số EPISODE báo giả / tổng giờ interictal.

    3. Detection Latency [giây]: tính RIÊNG cho TỪNG event (bản v3 chỉ lấy
       1 giá trị latency duy nhất cho toàn bộ chuỗi dù có nhiều cơn giật).
    """
    voter = DualThresholdVoter(n=n, k=k, thr_on=thr_on, thr_off=thr_off,
                                suppression_s=suppression_s, step_size=step_size)
    alarm = voter.predict(y_pred_scores)
    y_bin = (y_true > 0).astype(int)

    # --- Sensitivity & Latency theo EVENT ---
    runs = _find_runs(y_bin == 1)
    tp_events = fn_events = 0
    latencies = []
    for s, e in runs:
        hit_idx = np.where(alarm[s:e] == 1)[0]
        if len(hit_idx) > 0:
            tp_events += 1
            tts_at_alarm = tts_arr[s + hit_idx[0]]
            if not np.isnan(tts_at_alarm):
                latencies.append(pre_ictal - tts_at_alarm)
        else:
            fn_events += 1
    n_events = tp_events + fn_events
    sensitivity = (tp_events / n_events) if n_events > 0 else np.nan

    # Giữ thêm số liệu window-level để tham khảo/debug (KHÔNG dùng làm Sensitivity chính)
    tp_win = int(np.sum((alarm == 1) & (y_bin == 1)))
    fn_win = int(np.sum((alarm == 0) & (y_bin == 1)))

    # --- FAR/h (đếm theo EPISODE, không theo window) ---
    interictal_mask = (y_true == 0)
    n_inter_windows = np.sum(interictal_mask)
    total_interictal_hours = (n_inter_windows * step_size) / 3600.0
    inter_alarm = alarm[interictal_mask]
    fa_events = voter.count_alarm_events(inter_alarm)
    far_per_hour = fa_events / (total_interictal_hours + 1e-8)

    median_latency = float(np.median(latencies)) if latencies else np.nan
    mean_latency   = float(np.mean(latencies))   if latencies else np.nan

    return {
        "sensitivity":      sensitivity,
        "far_per_hour":     far_per_hour,
        "median_latency_s": median_latency,
        "mean_latency_s":   mean_latency,
        "tp_events": tp_events, "fn_events": fn_events,
        "tp_win": tp_win, "fn_win": fn_win,
        "fa_events":        fa_events,
        "total_inter_h":    total_interictal_hours,
        "n_latency_events": len(latencies),
        "alarm":            alarm,
        "latencies":        latencies,
    }


def compute_clinical_metrics_grouped(y_true, y_pred_scores, tts_arr, file_ids,
                                      thr_on, thr_off,
                                      window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                      pre_ictal=PRE_ICTAL,
                                      n=VOTER_N, k=VOTER_K, suppression_s=SUPPRESSION_S):
    """
    FIX v4 — BUG GỐC trong v3: val/test set là một SUBSET CHỌN NGẪU NHIÊN
    theo WINDOW (không theo file), rồi được sort lại theo index toàn cục
    trước khi đưa thẳng vào compute_clinical_metrics() như 1 chuỗi liên tục
    duy nhất. Hệ quả: DualThresholdVoter (có state tuần tự: hysteresis ON/
    OFF + suppression theo thời gian) chạy XUYÊN QUA ranh giới giữa các file/
    seizure khác nhau → 1 alarm "ON" cuối file A có thể bị tính suppression
    chặn alarm thật ở đầu file B, ngược lại "OFF" cuối file A lại không được
    suppression bảo vệ cho phần đầu file B. Toàn bộ Sensitivity/FAR/Latency
    tính ra từ v3 vì vậy không đáng tin cậy về mặt lâm sàng.

    Hàm này chạy Voter RIÊNG cho từng file_id (đúng thứ tự thời gian gốc
    trong file đó — yêu cầu y_true/y_pred_scores/tts_arr/file_ids đã được
    sort theo index gốc, xem load_split()), rồi GỘP kết quả: TP/FN event
    cộng dồn, FAR tính trên tổng giờ interictal của TOÀN BỘ các file, và
    danh sách latency nối từ tất cả file/event.
    """
    file_ids = np.asarray(file_ids)
    uniq = np.unique(file_ids)

    tp_events = fn_events = tp_win = fn_win = fa_events_total = 0
    total_inter_h = 0.0
    all_latencies = []
    alarm_full = np.zeros(len(y_true), dtype=int)

    for fid in uniq:
        g = np.where(file_ids == fid)[0]
        m = compute_clinical_metrics(y_true[g], y_pred_scores[g], tts_arr[g],
                                      thr_on, thr_off, window_size, step_size,
                                      pre_ictal, n, k, suppression_s)
        tp_events += m["tp_events"]; fn_events += m["fn_events"]
        tp_win    += m["tp_win"];    fn_win    += m["fn_win"]
        fa_events_total += m["fa_events"]
        total_inter_h   += m["total_inter_h"]
        all_latencies   += m["latencies"]
        alarm_full[g] = m["alarm"]

    n_events = tp_events + fn_events
    sensitivity = (tp_events / n_events) if n_events > 0 else np.nan
    far_per_hour = fa_events_total / (total_inter_h + 1e-8)
    median_latency = float(np.median(all_latencies)) if all_latencies else np.nan
    mean_latency   = float(np.mean(all_latencies))   if all_latencies else np.nan

    return {
        "sensitivity":      sensitivity,
        "far_per_hour":     far_per_hour,
        "median_latency_s": median_latency,
        "mean_latency_s":   mean_latency,
        "tp_events": tp_events, "fn_events": fn_events,
        "tp_win": tp_win, "fn_win": fn_win,
        "fa_events":        fa_events_total,
        "total_inter_h":    total_inter_h,
        "n_latency_events": len(all_latencies),
        "alarm":            alarm_full,
        "n_files":          len(uniq),
    }


def grid_search_dual_threshold(y_val_true, y_val_pred, tts_val, file_ids_val,
                                sensitivity_floor=SENSITIVITY_TARGET,
                                thr_on_grid=None, gap_grid=None, verbose=True):
    """
    Grid-search (thr_on, thr_off) trên VALIDATION set.

    Mục tiêu: trong số các cặp (thr_on, thr_off) đạt Sensitivity ≥ floor,
    chọn cặp có FAR thấp nhất.

    FIX v4: dùng compute_clinical_metrics_grouped (chạy Voter riêng từng
    file rồi gộp) thay vì gọi trực tiếp trên mảng val đã ghép nhiều file —
    val set giờ là tập các FILE riêng biệt (group split), nên buộc phải
    đánh giá theo từng file để Sensitivity/FAR không bị lệch do state Voter
    tràn qua biên file (xem docstring compute_clinical_metrics_grouped).

    FIX so với v3:
    - thr_off bị chặn >= THR_OFF_MIN (không cho =0, tránh bug "ON vĩnh viễn")
    - Grid mở rộng xuống thấp hơn (0.05) để xác nhận đường cong không bị
      cắt cụt giả tạo tại biên cũ (0.10)
    - In toàn bộ bảng kết quả (verbose) để kiểm chứng đường cong
      sensitivity-vs-threshold thực tế, không chỉ in kết quả cuối
    - Fallback dùng Youden's J (sensitivity - far_normalized) thay vì chỉ
      chọn sensitivity cao nhất tuyệt đối — tránh chọn điểm "FAR vô cực"
    """
    if thr_on_grid is None:
        thr_on_grid = np.arange(0.05, 0.95, 0.05)
    if gap_grid is None:
        gap_grid = np.arange(0.05, 0.35, 0.05)

    results = []  # lưu toàn bộ để debug + fallback
    best = None   # (far, sens, thr_on, thr_off)

    for thr_on in thr_on_grid:
        for gap in gap_grid:
            thr_off = round(thr_on - gap, 4)
            if thr_off < THR_OFF_MIN:
                continue
            m = compute_clinical_metrics_grouped(y_val_true, y_val_pred, tts_val, file_ids_val,
                                                  thr_on=thr_on, thr_off=thr_off)
            results.append((thr_on, thr_off, m["sensitivity"], m["far_per_hour"]))

            if verbose:
                print(f"    thr_on={thr_on:.2f} thr_off={thr_off:.2f} "
                      f"→ sens={m['sensitivity']:.3f} far={m['far_per_hour']:.3f}/h")

            if not np.isnan(m["sensitivity"]) and m["sensitivity"] >= sensitivity_floor:
                if best is None or m["far_per_hour"] < best[0]:
                    best = (m["far_per_hour"], m["sensitivity"], thr_on, thr_off)

    if best is None:
        print(f"[WARNING] Không tìm được (thr_on, thr_off) đạt Sensitivity ≥ {sensitivity_floor}.")
        print(f"          Fallback: chọn cặp tối ưu Youden's J = sensitivity - far/far_max "
              f"(cân bằng, không chỉ vét sensitivity tuyệt đối).")
        if not results:
            raise RuntimeError("Grid rỗng — kiểm tra lại thr_on_grid/gap_grid/THR_OFF_MIN.")

        valid_results = [r for r in results if not np.isnan(r[2])]
        if not valid_results:
            raise RuntimeError("Không có event pre-ictal nào trong validation set để tính Sensitivity.")
        fars = np.array([r[3] for r in valid_results])
        far_max = fars.max() if fars.max() > 0 else 1.0
        best_j = None
        for thr_on, thr_off, sens, far in valid_results:
            j = sens - (far / far_max)   # chuẩn hoá FAR về [0,1] để so sánh công bằng với sens
            if best_j is None or j > best_j[0]:
                best_j = (j, far, sens, thr_on, thr_off)
        _, far, sens, thr_on, thr_off = best_j
        best = (far, sens, thr_on, thr_off)

    far, sens, thr_on, thr_off = best
    print(f"[INFO] Grid-search kết quả: thr_on={thr_on:.2f}  thr_off={thr_off:.2f}  "
          f"→ Sensitivity(val)={sens:.3f}  FAR(val)={far:.3f}/h")
    return thr_on, thr_off

# =========================================================================
# 13. TRAINING
# =========================================================================
# =========================================================================
# 12B. GROUP SPLIT (FIX v4 — chống leakage do windows chồng lấp)
# =========================================================================
def group_train_val_test_split(idx_all, file_ids, is_pre, test_size=0.2, val_size=0.125, random_state=42):
    """
    Split theo GROUP=file_id (mỗi file/recording chỉ thuộc đúng 1 split),
    thay cho split ngẫu nhiên theo từng window (bug v3 — xem comment trong
    run_training). test_size/val_size là tỉ lệ WINDOW mục tiêu (GroupShuffleSplit
    cân theo group size để xấp xỉ tỉ lệ này, không đảm bảo tuyệt đối vì các
    file có số windows khác nhau).
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    pos_tr, pos_te = next(gss1.split(idx_all, groups=file_ids))
    idx_tr, idx_te = idx_all[pos_tr], idx_all[pos_te]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    pos_tr2, pos_va2 = next(gss2.split(idx_tr, groups=file_ids[idx_tr]))
    idx_va, idx_tr = idx_tr[pos_va2], idx_tr[pos_tr2]

    for name, idx in [("train", idx_tr), ("val", idx_va), ("test", idx_te)]:
        if not np.any(is_pre[idx]):
            print(f"[WARNING] Split '{name}' không chứa window pre-ictal nào sau group-split "
                  f"(random_state={random_state}). Thử random_state khác nếu cần Sensitivity "
                  f"hợp lệ trên split này.")
    return idx_tr, idx_va, idx_te


def run_training():
    print("\n" + "=" * 70)
    print(f"TRAINING v4 — ASYM FOCAL + VOTER — {TARGET_PATIENT}")
    print("=" * 70)

    with h5py.File(HDF5_PATH, "r") as h5f:
        y_all      = h5f["y"][:]
        tts_all    = h5f["tts"][:]
        file_id_all = h5f["file_id"][:]
        X_shape    = h5f["X"].shape
        n_ch       = int(h5f.attrs["n_channels"])

    print(f"[INFO] Total windows  : {len(y_all)}")
    print(f"[INFO] n_channels     : {n_ch}")
    print(f"[INFO] Số file (group): {len(np.unique(file_id_all))}")
    print(f"[INFO] Pre-ictal ratio: {(y_all > 0).mean():.3f}")

    idx_all = np.arange(len(y_all)); is_pre = (y_all > 0).astype(int)
    # FIX v4 — BUG LEAKAGE NGHIÊM TRỌNG trong v3: train_test_split() cũ chia
    # theo TỪNG WINDOW một cách ngẫu nhiên. Vì STEP_SIZE=5s < WINDOW_SIZE=10s,
    # các windows liên tiếp CHỒNG LẤP 50% dữ liệu thô → 2 windows kề nhau gần
    # như chắc chắn rơi vào 2 split khác nhau (train/test) một cách ngẫu nhiên,
    # khiến model gần như "nhìn thấy" một phần test ngay trong lúc train →
    # AUC/Sensitivity báo cáo được THỔI PHỒNG, không phản ánh khả năng tổng
    # quát hoá thật. Ngoài ra Voter (Sensitivity/FAR/Latency) cần 1 chuỗi
    # LIÊN TỤC theo thời gian để hoạt động đúng — windows rời rạc ngẫu nhiên
    # không đáp ứng được điều này.
    # → Chuyển sang split theo GROUP = file_id: toàn bộ windows của 1 file
    #   luôn nằm trọn trong 1 split duy nhất (train HOẶC val HOẶC test).
    idx_tr, idx_va, idx_te = group_train_val_test_split(
        idx_all, file_id_all, is_pre, test_size=0.2, val_size=0.125, random_state=42)
    print(f"[INFO] Split (windows): train={len(idx_tr)}  val={len(idx_va)}  test={len(idx_te)}")
    print(f"[INFO] Split (files)  : train={len(np.unique(file_id_all[idx_tr]))}  "
          f"val={len(np.unique(file_id_all[idx_va]))}  test={len(np.unique(file_id_all[idx_te]))}")

    y_tr = y_all[idx_tr]
    rng = np.random.RandomState(42)   # FIX v4: seed cố định cho reproducibility (trước đây dùng global np.random không seed)
    idx_i = idx_tr[y_tr == 0.0]; idx_p = idx_tr[y_tr > 0.0]
    n0    = min(len(idx_i), len(idx_p) * 2)
    idx_tr_bal = np.concatenate([rng.choice(idx_i, n0, replace=False), idx_p])
    rng.shuffle(idx_tr_bal)

    feat_size = X_shape[1] * X_shape[2]
    time_ds   = int(np.ceil(X_shape[3] / DOWNSAMPLE_FACTOR))

    out_sig = (
        {
            "temporal_input": tf.TensorSpec(shape=(None, time_ds, feat_size), dtype=tf.float32),
            "se_input":       tf.TensorSpec(shape=(None, n_ch),               dtype=tf.float32),
            "spe_input":      tf.TensorSpec(shape=(None, n_ch),               dtype=tf.float32),
            "pe_input":       tf.TensorSpec(shape=(None, n_ch),               dtype=tf.float32),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )

    train_ds = tf.data.Dataset.from_generator(
        make_generator(idx_tr_bal, HDF5_PATH, shuffle=True, augment=True),
        output_signature=out_sig
    ).prefetch(tf.data.AUTOTUNE)

    def load_split(idx):
        idx_s = np.sort(idx)   # thứ tự tăng dần theo index gốc = đúng thứ tự thời gian trong từng file
        with h5py.File(HDF5_PATH, "r") as h5f:
            Xr   = h5f["X"][idx_s];  SEr  = h5f["SE"][idx_s]
            SPEr = h5f["SPE"][idx_s]; PEr  = h5f["PE"][idx_s]
            yr   = h5f["y"][idx_s];   ttsr = h5f["tts"][idx_s]
            fidr = h5f["file_id"][idx_s]
        Xout = np.stack([transform(x) for x in Xr]).astype(np.float32)
        inp = {
            "temporal_input": Xout,
            "se_input":  SEr.astype(np.float32),
            "spe_input": SPEr.astype(np.float32),
            "pe_input":  PEr.astype(np.float32),
        }
        # FIX v4: trả thêm file_id (đã sort cùng thứ tự) — cần để evaluate
        # đúng theo từng file liên tục (compute_clinical_metrics_grouped)
        return inp, yr.astype(np.float32), ttsr.astype(np.float32), fidr.astype(np.int64)

    val_inputs,  y_val,  tts_val,  fid_val  = load_split(idx_va)
    test_inputs, y_test, tts_test, fid_test = load_split(idx_te)

    # Build & compile
    model = build_model(time_ds, feat_size, n_ch)
    model.summary()

    steps_per_epoch = len(idx_tr_bal) // BATCH_SIZE
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = steps_per_epoch * 3

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-4,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-5,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=asymmetric_focal_huber_loss(delta=0.5, gamma=1.5),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        ]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12,
                                          restore_best_weights=True, mode='min', verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            f"best_{TARGET_PATIENT}_v4.keras",
            monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=f"logs/{TARGET_PATIENT}_v4", histogram_freq=0),
    ]

    history = model.fit(
        train_ds,
        validation_data=(val_inputs, y_val),
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    # Print learned entropy weights
    for layer in model.layers:
        if isinstance(layer, LearnableEntropyScale):
            w = layer.get_weights_softmax()
            print(f"\n[INFO] Learned entropy weights — SE:{w[0]:.3f} SPE:{w[1]:.3f} PE:{w[2]:.3f}")

    # =========================================================================
    # 14. EVALUATE — Sensitivity / FAR / Latency với TTA + Voter
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATE — Clinical Metrics (Sensitivity / FAR / Latency)")
    print("=" * 70)

    def tta_predict(inputs, n_tta=N_TTA):
        preds = [model.predict(inputs, batch_size=128, verbose=0).flatten()]
        for _ in range(n_tta - 1):
            aug = {
                "temporal_input": inputs["temporal_input"],
                "se_input":  inputs["se_input"]  + np.random.normal(0, 0.01, inputs["se_input"].shape).astype(np.float32),
                "spe_input": inputs["spe_input"] + np.random.normal(0, 0.01, inputs["spe_input"].shape).astype(np.float32),
                "pe_input":  inputs["pe_input"]  + np.random.normal(0, 0.01, inputs["pe_input"].shape).astype(np.float32),
            }
            preds.append(model.predict(aug, batch_size=128, verbose=0).flatten())
        return np.mean(preds, axis=0)

    y_pred = tta_predict(test_inputs)
    y_pred_val = tta_predict(val_inputs)   # cần dự đoán trên VAL để grid-search ngưỡng

    y_bin     = (y_test > 0).astype(int)
    y_bin_val = (y_val  > 0).astype(int)

    auc_roc = auc_pr = None
    if len(np.unique(y_bin)) > 1:
        auc_roc = roc_auc_score(y_bin, y_pred)
        auc_pr  = average_precision_score(y_bin, y_pred)
        prec, rec, thr_arr = precision_recall_curve(y_bin, y_pred)
        print(f"\n  AUC-ROC: {auc_roc:.4f}  |  AUC-PR: {auc_pr:.4f}")

    # --- Grid-search (thr_on, thr_off) TRÊN VALIDATION, không phải test ---
    # Đây là điểm sửa quan trọng so với v2: chọn ngưỡng trên test set là data
    # leakage (ngưỡng "nhìn thấy" test trước khi báo cáo kết quả trên chính nó).
    print(f"\n[INFO] Grid-search dual-threshold trên VALIDATION set "
          f"(mục tiêu: FAR thấp nhất với Sensitivity ≥ {SENSITIVITY_TARGET})...")
    thr_on, thr_off = grid_search_dual_threshold(
        y_val, y_pred_val, tts_val, fid_val, sensitivity_floor=SENSITIVITY_TARGET
    )

    # Cảnh báo nếu kết quả vẫn trùng đúng biên thấp nhất của grid — đây là
    # dấu hiệu Voter N/K (hoặc model) vẫn đang ép ngưỡng về biên, cần xem lại
    # thay vì coi kết quả là "tối ưu" thật.
    _thr_on_floor = 0.05
    if abs(thr_on - _thr_on_floor) < 1e-6:
        print(f"[WARNING] thr_on chọn được = biên thấp nhất của grid ({_thr_on_floor}). "
              f"Khả năng Voter N/K={VOTER_N}/{VOTER_K} vẫn đang quá chặt, ép ngưỡng "
              f"về biên thay vì hội tụ tự nhiên. Khuyến nghị: thử VOTER_N/K nhỏ hơn "
              f"(ví dụ 3/2) hoặc kiểm tra lại model nếu vẫn không cải thiện.")

    # --- Áp dụng (thr_on, thr_off) đã chọn lên TEST set (không động lại) ---
    # FIX v4: dùng bản grouped — test set gồm nhiều FILE riêng biệt (group
    # split), Voter phải chạy liên tục TRONG TỪNG FILE rồi gộp kết quả.
    metrics = compute_clinical_metrics_grouped(y_test, y_pred, tts_test, fid_test,
                                                thr_on=thr_on, thr_off=thr_off)
    alarm = metrics["alarm"]

    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  TEST SET — DualThresholdVoter (thr_on={thr_on:.2f}, thr_off={thr_off:.2f})")
    print(f"  │  Voter N/K       : {VOTER_N}/{VOTER_K}   Suppression: {SUPPRESSION_S}s   Files: {metrics['n_files']}")
    print(f"  │  Sensitivity     : {metrics['sensitivity']:.4f} "
          f"({metrics['tp_events']} TP / {metrics['tp_events']+metrics['fn_events']} seizure events)")
    print(f"  │  FAR             : {metrics['far_per_hour']:.3f} alarm-event/h "
          f"({metrics['fa_events']} FA / {metrics['total_inter_h']:.1f} h)")
    print(f"  │  Median Latency  : {metrics['median_latency_s']:.1f} s trước seizure")
    print(f"  │  Mean Latency    : {metrics['mean_latency_s']:.1f} s trước seizure")
    print(f"  └──────────────────────────────────────────────────────────┘")

    print(f"\n  Classification Report (sau DualThresholdVoter):")
    print(classification_report(y_bin, alarm, target_names=["Interictal", "Pre-ictal"], zero_division=0))

    # =========================================================================
    # 15. PLOTS
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    axes[0,0].plot(history.history['loss'],     label='Train')
    axes[0,0].plot(history.history['val_loss'], label='Val')
    axes[0,0].set_title('Asym Focal-Huber Loss'); axes[0,0].legend()

    axes[0,1].plot(history.history['mae'],     label='Train MAE')
    axes[0,1].plot(history.history['val_mae'], label='Val MAE')
    axes[0,1].set_title('MAE'); axes[0,1].legend()

    axes[0,2].plot(history.history['rmse'],     label='Train RMSE')
    axes[0,2].plot(history.history['val_rmse'], label='Val RMSE')
    axes[0,2].set_title('RMSE'); axes[0,2].legend()

    axes[0,3].hist(y_pred[y_test == 0], bins=60, alpha=0.6, label='Interictal', color='steelblue')
    axes[0,3].hist(y_pred[y_test > 0],  bins=60, alpha=0.6, label='Pre-ictal',  color='tomato')
    axes[0,3].axvline(thr_on,  color='black', linestyle='--', label=f'thr_on={thr_on:.2f}')
    axes[0,3].axvline(thr_off, color='gray',  linestyle=':',  label=f'thr_off={thr_off:.2f}')
    axes[0,3].set_title('Score Distribution'); axes[0,3].legend()

    if len(np.unique(y_bin)) > 1:
        from sklearn.metrics import roc_curve
        fpr, tpr, thr_roc = roc_curve(y_bin, y_pred)
        axes[1,0].plot(fpr, tpr, color='darkorange', label=f'AUC={auc_roc:.3f}')
        axes[1,0].plot([0,1],[0,1], 'k--')
        axes[1,0].set_title('ROC'); axes[1,0].set_xlabel('FPR'); axes[1,0].set_ylabel('TPR'); axes[1,0].legend()

        axes[1,1].plot(rec, prec, color='green', label=f'AP={auc_pr:.3f}')
        axes[1,1].axvline(SENSITIVITY_TARGET, color='red', linestyle='--', label=f'Sens target={SENSITIVITY_TARGET}')
        axes[1,1].set_title('Precision-Recall'); axes[1,1].set_xlabel('Recall'); axes[1,1].legend()

        # FAR vs Sensitivity tại các thr_on khác nhau, GIỮ gap cố định bằng
        # với gap đã chọn bởi grid-search (thr_on - thr_off), để biểu đồ phản
        # ánh đúng cùng 1 cấu hình voter đang dùng — KHÔNG gọi hàm với 1 ngưỡng
        # đơn như bản cũ (đó là nguồn lỗi TypeError vì signature đã đổi).
        fixed_gap = thr_on - thr_off
        thrs_on = np.linspace(0.05, 0.90, 25)
        sens_list = []; far_list = []
        for thr_on_i in thrs_on:
            thr_off_i = max(THR_OFF_MIN, thr_on_i - fixed_gap)
            if thr_off_i >= thr_on_i:
                continue
            m_i = compute_clinical_metrics_grouped(y_test, y_pred, tts_test, fid_test,
                                                    thr_on=thr_on_i, thr_off=thr_off_i)
            sens_list.append(m_i['sensitivity'])
            far_list.append(m_i['far_per_hour'])
        axes[1,2].plot(far_list, sens_list, 'bo-', markersize=4)
        axes[1,2].axvline(metrics['far_per_hour'], color='red', linestyle='--', label=f'FAR đã chọn={metrics["far_per_hour"]:.2f}/h')
        axes[1,2].axhline(SENSITIVITY_TARGET, color='green', linestyle='--', label=f'Sens target={SENSITIVITY_TARGET}')
        axes[1,2].set_xlabel('FAR (alarm-event/h)'); axes[1,2].set_ylabel('Sensitivity')
        axes[1,2].set_title(f'FAR vs Sensitivity (gap cố định={fixed_gap:.2f})'); axes[1,2].legend()

    # Latency histogram — dùng `alarm` (kết quả của thr_on/thr_off đã chọn),
    # không dùng biến `voted` (không tồn tại trong v3, đó là lỗi của bản cũ)
    pre_mask_test = y_bin == 1
    if np.sum(pre_mask_test) > 0:
        valid_tts = tts_test[pre_mask_test]
        pre_alarm_test = alarm[pre_mask_test]
        valid_mask_nonan = ~np.isnan(valid_tts)
        valid_tts = valid_tts[valid_mask_nonan]
        pre_alarm_test = pre_alarm_test[valid_mask_nonan]
        detected_tts = valid_tts[pre_alarm_test == 1]
        if len(detected_tts) > 0:
            latency_minutes = (PRE_ICTAL - detected_tts) / 60.0
            axes[1,3].hist(latency_minutes, bins=20, color='purple', alpha=0.7)
            axes[1,3].axvline(0, color='red', linestyle='--', label='Onset (t=0)')
            axes[1,3].set_xlabel('Phút trước seizure (latency âm = phát hiện sớm)')
            axes[1,3].set_title('Detection Latency Distribution'); axes[1,3].legend()
        else:
            axes[1,3].text(0.5, 0.5, 'Không có alarm trong pre-ictal',
                            ha='center', va='center', transform=axes[1,3].transAxes)

    plt.suptitle(f'v4 Triple Entropy AsymFocal + DualThreshold Fixed — {TARGET_PATIENT}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_png = f"results_{TARGET_PATIENT}_v4.png"
    plt.savefig(out_png, dpi=130)
    print(f"\n[INFO] Biểu đồ: {out_png}")

    model.save(f"chbmit_{TARGET_PATIENT}_v4.keras")
    print(f"[INFO] Model đã lưu.")

# =========================================================================
# ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    if not os.path.exists(HDF5_PATH):
        run_preprocessing()
    else:
        print(f"[INFO] HDF5 '{HDF5_PATH}' tồn tại, bỏ qua preprocessing.")
    run_training()
