# Thực Tập Công Nghiệp — Class-Incremental Multi-Organ Segmentation (BTCV)

Đây là đồ án **Class-Incremental Learning (CIL) cho bài toán phân đoạn đa cơ quan (multi-organ segmentation)** trên bộ dữ liệu CT **BTCV** (Multi-Atlas Labeling Beyond the Cranial Vault), gồm 4 task tuần tự và 13 cơ quan (+ background = 14 lớp). Kiến trúc mô hình: **U-Net / U-Net++ với encoder ResNet-34** (qua thư viện `segmentation_models_pytorch`).

Repo so sánh 4 chiến lược:

| Notebook | Chiến lược | Vai trò |
|---|---|---|
| `joint_training.ipynb` | Train chung tất cả 13 organ cùng lúc | **Upper bound** (không CIL, để so sánh) |
| `FineTuning.ipynb` | Fine-tune tuần tự, không có cơ chế chống quên | **Lower bound / Task 1 trainer** |
| `LwF_baseline.ipynb` | Learning without Forgetting (Li & Hoiem, 2016) | Baseline CIL #1 |
| `MiB_Baseline.ipynb` | Modeling the Background (Cermelli et al., CVPR 2020) | Baseline CIL #2 |
| `Pixel‑Aware EWC.ipynb` | Online EWC-style regularization (Kirkpatrick et al., 2017, biến thể) | Baseline CIL #3 |

Toàn bộ code được viết bằng **tiếng Việt** (comment, print, docstring) và ban đầu được xây dựng để chạy trên **Google Colab (GPU T4, Python 3.10)** với dữ liệu lưu trên Google Drive. README này hướng dẫn cách chạy lại **hoàn toàn trên máy local**.

---

## 1. Yêu cầu hệ thống

- Python 3.10+ (khớp với môi trường gốc trên Colab)
- GPU NVIDIA khuyến nghị ≥ 8 GB VRAM (bản gốc dùng T4 16 GB, `batch_size=16`, ảnh 512×512). Có thể chạy CPU nhưng sẽ rất chậm.
- ~10–15 GB dung lượng trống (RawData.zip của BTCV + dữ liệu `.npy` đã tiền xử lý + checkpoint).
- Tài khoản [Synapse](https://www.synapse.org/) để tải bộ dữ liệu BTCV (miễn phí, cần đăng ký).

## 2. Cài đặt môi trường

```bash
git clone https://github.com/TruomghocAI/Thuc_Tap_Cong_Nghiep.git
cd Thuc_Tap_Cong_Nghiep

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Nếu chưa có `requirements.txt` trong repo, tạo file với nội dung sau (đã tổng hợp từ toàn bộ `!pip install` + import trong 5 notebook):

```
torch>=2.0
torchvision>=0.15
segmentation-models-pytorch>=0.3.3
nibabel>=5.0
numpy>=1.24
scipy>=1.10
jupyterlab
tqdm
matplotlib
pandas
seaborn
```

> Nếu máy có GPU, cài `torch`/`torchvision` theo đúng bản CUDA của máy theo hướng dẫn tại [pytorch.org](https://pytorch.org/get-started/locally/) thay vì bản CPU mặc định.

## 3. Tải & chuẩn bị dữ liệu BTCV

1. Đăng ký tài khoản tại [synapse.org](https://www.synapse.org/), tìm project **"Multi-Atlas Labeling Beyond the Cranial Vault"** (BTCV, `syn3193805`) và tải file **`RawData.zip`** (chỉ cần phần **Training** — 30 CT volume có label).
2. Đặt file zip vào một thư mục local, ví dụ:
   ```
   Thuc_Tap_Cong_Nghiep/
   └── data_raw/RawData.zip
   ```

## 4. ⚠️ Việc cần sửa trước khi chạy (notebook viết cho Colab)

Cả 5 notebook đều có các cell dành riêng cho Colab, cần sửa để chạy local:

**a) Bỏ cell mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```
Xoá hoặc comment cell này (không tồn tại trong Jupyter/local Python).

**b) Đổi đường dẫn gốc dữ liệu**

Notebook tiền xử lý dùng:
```python
PROJECT_NAME = "MultiOrganSeg"
DRIVE_ROOT   = f"/content/drive/MyDrive/{PROJECT_NAME}"
```
còn 4 notebook train (LwF, FineTuning, joint_training, EWC) lại dùng:
```python
DRIVE_ROOT = "/content/drive/MyDrive/Multi-Atlas_Labeling_Beyond_the_Cranial_Vault"
```
**Đây là hai tên thư mục khác nhau** — nếu giữ nguyên, các notebook train sẽ không tìm thấy dữ liệu do notebook tiền xử lý tạo ra. Bạn cần **chọn một đường dẫn local duy nhất và sửa đồng bộ ở tất cả các notebook**, ví dụ:
```python
DRIVE_ROOT = "./data/Multi-Atlas_Labeling_Beyond_the_Cranial_Vault"
```
(`MiB_Baseline.ipynb` là ngoại lệ — notebook này đã dùng sẵn đường dẫn local `./data/Multi-Atlas_Labeling_Beyond_the_Cranial_Vault`, không cần sửa phần Drive.)

**c) Bỏ cell `!pip install ...`** (không bắt buộc, nhưng nếu đã `pip install -r requirements.txt` thì có thể skip để đỡ chạy lại mỗi lần mở notebook).

## 5. Thứ tự chạy pipeline

### Bước 1 — Tiền xử lý dữ liệu
Chạy `processing data/data_exploration_and_preprocessing.ipynb`:
- Giải nén `RawData.zip`
- Ghép cặp ảnh CT ↔ label theo ID (chỉ giữ 30 volume Training có label)
- Trực quan hoá dữ liệu (overview, class imbalance, 3 views)
- Tiền xử lý: clip HU về `[-1000, 1000]`, chuẩn hoá `float32 [0,1]`, resize `512×512`, bỏ slice có < 0.5% pixel foreground
- Xuất ra: `data/processed/images/*.npy`, `data/processed/masks/*.npy`, `data/processed/metadata.json`

Ảnh CT lưu ra là **2D single-channel** `(512, 512) float32`, mask là `(512, 512) uint8` giá trị 0–13.

### Bước 2 — Tạo checkpoint Task 1 dùng chung
Cả `LwF_baseline.ipynb`, `MiB_Baseline.ipynb` và `Pixel‑Aware EWC.ipynb` đều **giả định đã có sẵn** file:
```
<CHECKPOINT_DIR>/task1_best.pth
```
và sẽ báo lỗi `FileNotFoundError` nếu chưa có. File này **không được tạo ra bởi bất kỳ notebook nào trong repo một cách hiển nhiên** — cách để tạo nó là:

Mở `FineTuning.ipynb`, đặt `CURRENT_TASK = 1`, chạy toàn bộ notebook. Vì Task 1 không có cơ chế chống quên nào để phân biệt giữa các phương pháp (task đầu tiên luôn là train thường), checkpoint `task1_best.pth` sinh ra từ đây dùng chung cho tất cả các baseline CIL phía sau. Đảm bảo `CHECKPOINT_DIR` của `FineTuning.ipynb` trùng với `CHECKPOINT_DIR` mà `LwF_baseline.ipynb` / `MiB_Baseline.ipynb` / `Pixel‑Aware EWC.ipynb` sẽ đọc (mặc định cùng là `checkpoints_baseline/`).

### Bước 3 — Chạy các baseline

- **`joint_training.ipynb`** — độc lập, không phụ thuộc `task1_best.pth`. Train một lần trên toàn bộ 13 organ. Chạy toàn bộ notebook.
- **`FineTuning.ipynb`** — sau khi đã tạo `task1_best.pth` ở Bước 2, đổi `CURRENT_TASK = 2`, chạy lại toàn bộ notebook; lặp lại với `CURRENT_TASK = 3`, rồi `4`. Mỗi lần chạy tự load checkpoint của task liền trước.
- **`LwF_baseline.ipynb`** — chạy toàn bộ notebook, vòng lặp `STARTING_TASK → 4` tự động train Task 2, 3, 4 với knowledge distillation từ teacher là checkpoint task liền trước.
- **`MiB_Baseline.ipynb`** — tương tự LwF nhưng dùng kỹ thuật Modeling-the-Background; `ALPHA_PER_TASK` đã tune sẵn `{1:0.0, 2:0.3, 3:0.5, 4:0.6}`.
- **`Pixel‑Aware EWC.ipynb`** — tính Fisher Information từ checkpoint Task 1, sau đó phạt thay đổi trọng số quan trọng khi train Task 2→4.

Tất cả (trừ `FineTuning.ipynb`) có thể **Run All** một lần từ đầu đến cuối; `FineTuning.ipynb` cần chạy lại thủ công 4 lần (đổi `CURRENT_TASK`).

## 6. Kết quả đầu ra

Sau khi chạy, mỗi notebook ghi ra thư mục con của `DRIVE_ROOT`/`DATA_ROOT`:

- `checkpoints_baseline/` hoặc `checkpoints_ewc_v2/`: các file `.pth` (model, optimizer, scheduler, epoch, seed, config...)
- `logs_baseline/` hoặc `logs_ewc_v2/`: log JSON theo từng task, per-organ DSC/mIoU
- Ma trận hiệu năng CIL `R[i][j]` (Task i đánh giá trên organ của Task j) cùng các chỉ số dẫn xuất: **Average Accuracy, Forgetting, Backward Transfer**, tính trên **test set niêm phong (sealed)** — chỉ chạy 1 lần sau khi đã train xong toàn bộ, không dùng để chọn hyperparameter.

## 7. Một số lưu ý quan trọng (phát hiện khi đọc code)

- **`LwF_baseline.ipynb` có lỗi khiến notebook không chạy được ngay**: hàm `ensure_ct_25d_array(...)` được gọi trong `BTCVDataset.__getitem__` nhưng **không được định nghĩa ở đâu trong notebook**; đồng thời code còn tham chiếu `TRAIN_CFG["in_channels"]` và `TRAIN_CFG["num_slices"]`, hai key này **không tồn tại** trong `TRAIN_CFG`. Notebook cũng dùng `smp.UnetPlusPlus(in_channels=3, ...)` — tức kỳ vọng ảnh đầu vào 2.5D 3-kênh — trong khi notebook tiền xử lý ở Bước 1 chỉ xuất ảnh **2D 1-kênh**. Bạn cần tự bổ sung hàm `ensure_ct_25d_array` (ví dụ: lặp lại slice 2D thành 3 kênh, hoặc đơn giản hơn là đổi sang `smp.Unet(in_channels=1, ...)` giống các notebook khác) và thêm 2 key còn thiếu vào `TRAIN_CFG` trước khi chạy.
- **Cách định nghĩa Task khác nhau giữa các notebook**: `LwF_baseline.ipynb` dùng `TASK_ORGANS = {1: [6,7,1], 2: [2,3,11], 3: [8,9,10], 4: [4,5,12,13]}`, trong khi `MiB_Baseline.ipynb`, `Pixel‑Aware EWC.ipynb`, `FineTuning.ipynb`, `joint_training.ipynb` lại dùng cách chia theo kích thước cơ quan (lớn → nhỏ): `{1: [6,7], 2: [1,2,3,8], 3: [4,9,10,11], 4: [5,12,13]}`. Nếu muốn so sánh công bằng giữa các phương pháp, cần thống nhất lại `TASK_ORGANS` giữa các notebook.
- `MiB_Baseline.ipynb` và `Pixel‑Aware EWC.ipynb` có cơ chế kiểm tra `assert_checkpoint_compatible()` dựa trên `split_checksum`/`config_hash`/`seed` ghi trong checkpoint; checkpoint Task 1 tạo từ `FineTuning.ipynb` (Bước 2) không có các trường này nhưng được code cho phép qua với cảnh báo "LEGACY WARNING" — đây là hành vi được thiết kế có chủ đích, không phải lỗi.
- Notebook `Pixel‑Aware EWC.ipynb` tự nhận là **"Online EWC-style regularization baseline"**, không phải cài đặt EWC gốc/offline theo Kirkpatrick et al. (2017) — nên khi báo cáo kết quả, gọi đúng tên này để tránh gây hiểu nhầm.

## 8. Giấy phép

Phát hành theo **MIT License** — xem file [`LICENSE`](./LICENSE).

14. Citation / Acknowledgement

This codebase is for research and educational purposes on class-incremental multi-organ CT segmentation using the BTCV abdominal CT dataset.
