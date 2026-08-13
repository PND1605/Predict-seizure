# EEG Seizure Prediction — Pipeline 3E v4

`pipeline_3e_v4 (Platt scaling + adaptive percentile).py` là pipeline nghiên cứu dự đoán cơn động kinh từ EEG theo hướng **patient-specific**, sử dụng dữ liệu CHB-MIT.

> **Scope:** mỗi bệnh nhân được huấn luyện một model riêng. Model không được dùng trực tiếp cho bệnh nhân khác nếu chưa retrain. Pipeline đánh giá bằng **LOSO-CV (Leave-One-Seizure-File-Out Cross-Validation)**. fileciteturn0file0L6-L22

---

## 1. Mục tiêu

Pipeline thực hiện toàn bộ chuỗi:

```text
CHB-MIT EDF
   │
   ├── Load EEG
   ├── Band-pass + Notch filtering
   ├── Resampling
   ├── Sliding-window segmentation
   │
   ├── Adaptive SOP labeling
   │
   ├── Temporal features
   │     └── Multiband EEG
   │
   ├── Entropy features
   │     ├── Sample Entropy (SE)
   │     ├── Spectral Entropy (SPE)
   │     └── Permutation Entropy (PE)
   │
   ├── HDF5 dataset
   │
   └── LOSO-CV
         ├── Train
         ├── Validation
         ├── Platt calibration
         ├── Adaptive percentile threshold
         ├── Firing Power + hysteresis
         └── Clinical/event-level evaluation
```

Pipeline sử dụng kiến trúc hai nhánh: temporal branch với Conv1D + BiLSTM và entropy branch với SE/SPE/PE, sau đó fusion trước lớp sigmoid output. Kiến trúc được thiết kế khoảng ~200K parameters để hạn chế overfitting trên dataset nhỏ. fileciteturn1file7L361-L390

---

## 2. Các thành phần chính

### 2.1. Preprocessing

Mặc định:

| Tham số | Giá trị |
|---|---:|
| Sampling frequency | 256 Hz |
| Window | 10 s |
| Step | 5 s |
| Overlap | 50% |
| High-pass | 0.5 Hz |
| Low-pass | 70 Hz |
| Notch | 50, 60 Hz |
| Bands | 0.5–4, 4–8, 8–13, 13–30, 30–70 Hz |

EEG được đọc bằng MNE, lọc, resample về 256 Hz và chuyển đơn vị từ V sang µV. fileciteturn0file0L327-L365

### 2.2. Labeling

Pipeline sử dụng:

- Default `pre_ictal_s = 1800 s`
- `SPH = 60 s`
- `post_ictal_s = 120 s`
- Adaptive SOP có thể chọn trong:

```text
600, 900, 1200, 1800, 2400, 3000 seconds
```

Adaptive SOP lựa chọn khoảng pre-ictal dựa trên mức độ tách biệt giữa phân phối log-energy của vùng pre-ictal candidate và baseline interictal. Nếu dữ liệu nền không đủ hoặc phân phối không tách biệt, pipeline fallback về SOP mặc định. fileciteturn0file0L155-L187 fileciteturn0file0L440-L516

### 2.3. Entropy features

Mỗi EEG window được tính:

- Sample Entropy — SE
- Spectral Entropy — SPE
- Permutation Entropy — PE

Entropy được lưu raw trong HDF5. Median/IQR normalization chỉ được tính từ **training set của từng LOSO fold**, sau đó áp dụng cho train/validation/test của fold đó để tránh data leakage. fileciteturn0file0L577-L624

### 2.4. Model

```text
Temporal branch:
Conv1D(32)
    ↓
Conv1D(64)
    ↓
MaxPooling1D
    ↓
BiLSTM(32)
    ↓
Dense(32)

Entropy branch:
SE + SPE + PE
    ↓
Dense(64)
    ↓
Dense(32)

Fusion:
Temporal + Entropy
    ↓
Dense(48)
    ↓
Sigmoid
```

Loss sử dụng `Asymmetric Focal Huber`, trong đó false negative được phạt mạnh hơn false positive (`alpha_fn=3.0`, `alpha_fp=1.0`). fileciteturn1file7L393-L405

---

## 3. Calibration và threshold

### Platt scaling

Khi `use_calibration=True`, raw model scores trên validation được hiệu chỉnh bằng logistic regression:

```text
P(preictal | score) = sigmoid(A × score + B)
```

Calibration được fit trên validation của chính fold và sau đó áp dụng cho validation/test; test không tham gia quá trình fit. fileciteturn2file0L60-L97

### Adaptive percentile threshold

Mặc định:

```python
threshold_mode = "relative"
```

Threshold được suy ra từ tỷ lệ pre-ictal của validation:

```text
preictal_frac = mean(y_val > 0)

pct_on  = 100 × (1 - preictal_frac)
pct_off = pct_on - relative_pct_margin
```

Sau đó percentile được chuyển thành `thr_on` và `thr_off`. Cách này thay thế grid-search threshold trực tiếp trên Sensitivity/FAR của validation, vốn có nguy cơ overfit khi mỗi validation fold chỉ có một seizure event. fileciteturn2file0L100-L160

---

## 4. Post-processing

Pipeline sử dụng `DualThresholdVoter` với:

1. **Firing Power** — tỷ lệ window vượt `thr_on` trong cửa sổ trượt.
2. **Hysteresis** — `thr_on` để bật alarm và `thr_off` để tắt.
3. **Suppression window** — tránh đếm nhiều alarm trong cùng một episode.

Mặc định:

```text
Firing Power window = 60 s
Minimum thr_off     = 0.02
Suppression         = 900 s
```

fileciteturn1file4L227-L247

---

## 5. Evaluation

Pipeline đánh giá theo **LOSO-CV**:

```text
Fold i:
    TEST  = seizure file i
    VAL   = seizure file i+1
    TRAIN = tất cả file còn lại
```

Mỗi fold có model riêng. Pipeline yêu cầu tối thiểu 2 seizure files để thực hiện LOSO-CV. fileciteturn1file5L258-L295

### Clinical/event-level metrics

Pipeline báo cáo:

- AUC-ROC
- AUC-PR
- Event-level Sensitivity
- FAR (alarm-event/hour)
- TP
- FN-late
- FN-miss
- Warning time

`SPH = 60 s`: alarm chỉ được tính là TP nếu tại thời điểm alarm còn ít nhất 60 giây trước onset. Warning time được định nghĩa là số giây còn lại trước onset tại alarm hợp lệ. fileciteturn2file0L15-L26

### Target metrics tham khảo

```text
AUC-ROC       ≥ 0.75
AUC-PR        ≥ 0.50
Sensitivity   ≥ 0.80
FAR           ≤ 0.50 / hour
Warning time  ≥ 120 s
```

Đây là các target được khai báo trong source, không phải kết quả thực nghiệm của pipeline hiện tại. fileciteturn0file0L41-L46

---

## 6. Yêu cầu dữ liệu

Pipeline cần dữ liệu EEG dạng **EDF** và file summary của CHB-MIT để xác định seizure onset/offset. Nên tải toàn bộ file của bệnh nhân từ nguồn uy tín (ít nhất cần có đủ file của CHB-01)

Dataset directory được cấu hình bằng biến môi trường:

```text
EEG_DATASET_PATH
```

Nếu không đặt biến môi trường, source hiện có một default Windows path:

```text
C:\Users\Admin\Downloads\EEG_Project\EEG dataset
```

Nên ưu tiên dùng `EEG_DATASET_PATH` để pipeline portable hơn. fileciteturn0file0L131-L150

Ví dụ Windows PowerShell:

```powershell
$env:EEG_DATASET_PATH="D:\EEG\CHB-MIT"
python "pipeline_3e_v4 (Platt scaling + adaptive percentile).py"
```

---

## 7. Cài đặt

### 7.1. Tạo môi trường ảo

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 7.2. Cài dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 7.3. Kiểm tra import

```bash
python -c "import numpy, scipy, sklearn, mne, h5py, tensorflow, antropy, matplotlib, tqdm; print('Dependencies OK')"
```

---

## 8. Chạy pipeline

File có entry point trực tiếp:

```bash
python "pipeline_3e_v4 (Platt scaling + adaptive percentile).py"
```

Khi chạy:

1. Kiểm tra HDF5 đã preprocessing hoàn tất chưa.
2. Nếu chưa, preprocessing toàn bộ EDF.
3. Tạo HDF5.
4. Sinh LOSO folds.
5. Train model cho từng fold.
6. Calibration bằng Platt scaling.
7. Chọn threshold.
8. Đánh giá test.
9. Tổng hợp toàn bộ folds.
10. Lưu plot và JSON summary.

Entry point tự động chạy preprocessing nếu HDF5 chưa có flag `preprocessing_complete=True`; nếu HDF5 đã hoàn tất thì preprocessing được bỏ qua. fileciteturn2file0L553-L590

---

## 9. Output

Mặc định source ghi kết quả vào:

```text
C:\Users\Admin\Downloads\EEG_Project\output_dir
```

Các output chính:

```text
output_dir/
├── fold0_best.keras
├── fold1_best.keras
├── ...
├── loso_cv_chb01.png
└── summary_chb01.json
```

HDF5 preprocessing mặc định nằm trong:

```text
C:\Users\Admin\Downloads\EEG_Project\hdf5_path\
└── chbmit_chb01_rebuild.h5
```

Tên patient thay đổi theo `Config.patient`. Source cũng kiểm tra flag `preprocessing_complete` thay vì chỉ kiểm tra file HDF5 có tồn tại hay không. fileciteturn0file0L263-L280 fileciteturn3file0L11-L19

JSON summary chứa:

```text
patient
research_mode
evaluation
sph_s
n_folds
auc_roc_mean
auc_roc_std
auc_pr_mean
auc_pr_std
overall_sensitivity
overall_far_per_h
warn_time_median_s
warn_time_mean_s
n_detected
n_total_seizures
total_fa
total_inter_h
fold_details
```

fileciteturn2file0L476-L502

---

## 10. Cấu hình quan trọng

Các tham số thường cần thay đổi nằm trong `Config`:

```python
patient = "chb01"

window_s = 10
step_s = 5

sfreq = 256

pre_ictal_s = 1800
post_ictal_s = 120
sph_s = 60

adaptive_sop = True

epochs = 30
patience = 8
batch_size = 64

alpha_fn = 3.0
alpha_fp = 1.0

use_calibration = True

threshold_mode = "relative"
fixed_thr_on = 0.50
fixed_thr_off = 0.40
relative_pct_margin = 5.0
```

Các giá trị trên là cấu hình mặc định được khai báo trực tiếp trong source. fileciteturn0file0L131-L260

---

## 11. Lưu ý về nghiên cứu

### Không thay đổi test set trong quá trình tuning

Test seizure của mỗi fold phải được giữ độc lập. Calibration và threshold selection được thực hiện trước khi đánh giá test.

### Không normalize entropy trên toàn dataset

Entropy normalization phải được fit từ training portion của từng fold. Đây là một phần quan trọng để tránh leakage. fileciteturn1file2L138-L167

### Không đánh giá chỉ bằng window-level accuracy

Seizure prediction cần xem xét event-level Sensitivity, FAR, SPH và Warning Time. Pipeline đã triển khai các metric này theo từng file để tránh state của voter bị rò từ file này sang file khác. fileciteturn2file0L29-L57

### Patient-specific

Kết quả của `chb01` chỉ nên được diễn giải trong phạm vi model được train cho `chb01`; cross-patient generalization không thuộc scope của pipeline này. fileciteturn0file0L6-L12

---

## 12. Cấu trúc project đề xuất

```text
EEG_Project/
│
├── pipeline_3e_v4 (Platt scaling + adaptive percentile).py
├── requirements.txt
├── README.md
│
├── EEG dataset/
│   └── CHB-MIT EDF + summary files
│
├── hdf5_path/
│   └── chbmit_<patient>_rebuild.h5
│
└── output_dir/
    ├── fold*_best.keras
    ├── loso_cv_<patient>.png
    └── summary_<patient>.json
```

---

## 13. Tóm tắt pipeline

```text
EDF
 ↓
MNE load
 ↓
0.5–70 Hz + notch 50/60 Hz
 ↓
Resample 256 Hz
 ↓
10 s windows / 5 s step
 ↓
Adaptive SOP labeling
 ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Multiband EEG         SE/SPE/PE
 │                       │
 ▼                       ▼
Conv1D → BiLSTM       Dense → Dense
 │                       │
 └──────────┬────────────┘
            ▼
       Feature Fusion
            ▼
        Dense(48)
            ▼
       Sigmoid score
            ▼
       Platt scaling
            ▼
 Adaptive percentile threshold
            ▼
 Firing Power + Hysteresis
            ▼
       Clinical metrics
            ▼
         LOSO-CV
            ▼
   JSON + plots + checkpoints
```

---

## 14. Phiên bản và phạm vi

**Pipeline:** `3E v4`  
**Biến thể:** Platt scaling + adaptive percentile  
**Research mode:** Patient-specific  
**Evaluation:** LOSO-CV  
**Dataset target:** CHB-MIT  
**Task:** EEG seizure prediction / pre-ictal detection

Pipeline hiện tại là code nghiên cứu; các target metric trong README là benchmark mục tiêu được khai báo trong source, không phải cam kết hiệu năng lâm sàng.
