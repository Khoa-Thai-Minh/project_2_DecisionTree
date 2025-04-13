import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import math




#phần code 2.1 của dataset thứ 3
# -------------------------------------------------------------
# Bước 0: Fetch dữ liệu bằng ucimlrepo
# -------------------------------------------------------------
print("--- Đang tải dữ liệu Chronic Kidney Disease (ID: 336) từ ucimlrepo ---")
try:
    chronic_kidney_disease = fetch_ucirepo(id=336)
    X_orig = chronic_kidney_disease.data.features
    y_orig = chronic_kidney_disease.data.targets
    print("Đã tải dữ liệu thành công.")
except Exception as e:
    print(f"Lỗi khi tải dữ liệu: {e}")
    print("Vui lòng kiểm tra kết nối mạng hoặc ID dataset.")
    exit()

X = X_orig.copy()
y = y_orig.copy()

if len(y.columns) == 1:
    target_col_name = y.columns[0]
else:
    target_col_name = 'class'
    if target_col_name not in y.columns: exit(f"Lỗi: Cột target '{target_col_name}' không tồn tại.")

# -------------------------------------------------------------
# Bước 1: Tiền xử lý dữ liệu (Code giữ nguyên như trước)
# -------------------------------------------------------------
print("\n--- Bắt đầu tiền xử lý dữ liệu ---")
# 1.1 Xử lý cột Target (y)
replace_map_target = {'ckd': 1, 'notckd': 0, 'ckd\t': 1}
y[target_col_name] = y[target_col_name].replace(replace_map_target)
y[target_col_name] = pd.to_numeric(y[target_col_name], errors='coerce')
if y[target_col_name].isnull().any():
    nan_target_indices = y[y[target_col_name].isnull()].index
    y = y.drop(nan_target_indices)
    X = X.drop(nan_target_indices)
y[target_col_name] = y[target_col_name].astype(int)
y_final = y[target_col_name]

# 1.2 Xử lý giá trị thiếu trong Features (X)
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()
if 'id' in numeric_cols: numeric_cols.remove('id')
if 'id' in X.columns: X = X.drop('id', axis=1)
for col in numeric_cols:
    if X[col].isnull().any(): X[col].fillna(X[col].median(), inplace=True)
for col in categorical_cols:
    if X[col].isnull().any(): X[col].fillna(X[col].mode()[0], inplace=True)

# 1.3 Mã hóa One-Hot cho biến hạng mục (Nominal)
nominal_features = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
cols_to_encode = [col for col in nominal_features if col in X.columns]
X_encoded = pd.get_dummies(X, columns=cols_to_encode, drop_first=False, prefix_sep='_')
X_final = X_encoded
print("--- Tiền xử lý dữ liệu hoàn tất ---")

# -------------------------------------------------------------
# Bước 2: Chia dữ liệu và Visualize trên CÙNG MỘT FIGURE
# -------------------------------------------------------------
print("\n--- Bắt đầu chia dữ liệu Train/Test và Visualize ---")

datasets_ckd = {}
split_ratios = {'40/60': 0.6, '60/40': 0.4, '80/20': 0.2, '90/10': 0.1}
random_seed = 42

# --- Tính toán layout cho subplot ---
n_ratios = len(split_ratios)
n_plots_per_ratio = 2 # 1 cho train, 1 cho test
total_plots = 1 + n_ratios * n_plots_per_ratio # 1 cho gốc + plots cho các tỷ lệ
# Tính số hàng và cột cần thiết, ưu tiên cột nhiều hơn một chút
n_cols = 3 # Đặt số cột mong muốn (ví dụ: 3)
n_rows = math.ceil(total_plots / n_cols)

# --- Tạo Figure và Axes tổng ---
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4)) # Điều chỉnh figsize nếu cần
# Làm phẳng mảng axes để dễ truy cập bằng chỉ số duy nhất
axes = axes.flatten()
plot_index = 0 # Chỉ số của subplot hiện tại

# --- Vẽ biểu đồ phân phối lớp gốc ---
ax = axes[plot_index]
sns.countplot(x=y_final, ax=ax, palette='viridis')
ax.set_title('Phân phối lớp gốc (Đã xử lý)')
ax.set_xticks(ticks=[0, 1])
ax.set_xticklabels([f'Not CKD (0)\n({(y_final==0).sum()})', f'CKD (1)\n({(y_final==1).sum()})'])
ax.set_ylabel("Số lượng mẫu")
ax.set_xlabel("Lớp")
plot_index += 1

# --- Vòng lặp chia dữ liệu và vẽ subplot ---
for name, test_prop in split_ratios.items():
    print(f"\n--- Chia dữ liệu CKD tỷ lệ {name} (Train/Test) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final,
        test_size=test_prop,
        random_state=random_seed,
        shuffle=True,
        stratify=y_final
    )
    datasets_ckd[name] = {
        'feature_train': X_train, 'label_train': y_train,
        'feature_test': X_test, 'label_test': y_test
    }
    print(f"Kích thước: Train={len(X_train)}, Test={len(X_test)}")

    # --- Vẽ subplot cho tập Train ---
    if plot_index < len(axes): # Kiểm tra xem còn chỗ vẽ không
        ax = axes[plot_index]
        sns.countplot(x=y_train, ax=ax, palette='Blues') # Màu khác cho train
        ax.set_title(f'Train Set ({name}) - {len(y_train)} mẫu')
        ax.set_xticks(ticks=[0, 1])
        ax.set_xticklabels([f'Not CKD\n({(y_train==0).sum()})', f'CKD\n({(y_train==1).sum()})'])
        ax.set_ylabel("Số lượng mẫu")
        ax.set_xlabel(None) # Bỏ label trục x để đỡ rối
        plot_index += 1
    else:
        print("Warning: Không đủ subplot để vẽ Train set.")


    # --- Vẽ subplot cho tập Test ---
    if plot_index < len(axes): # Kiểm tra xem còn chỗ vẽ không
        ax = axes[plot_index]
        sns.countplot(x=y_test, ax=ax, palette='Oranges') # Màu khác cho test
        ax.set_title(f'Test Set ({name}) - {len(y_test)} mẫu')
        ax.set_xticks(ticks=[0, 1])
        ax.set_xticklabels([f'Not CKD\n({(y_test==0).sum()})', f'CKD\n({(y_test==1).sum()})'])
        ax.set_ylabel("Số lượng mẫu")
        ax.set_xlabel(None) # Bỏ label trục x
        plot_index += 1
    else:
         print("Warning: Không đủ subplot để vẽ Test set.")


# --- Ẩn các subplot không sử dụng ---
for i in range(plot_index, len(axes)):
    fig.delaxes(axes[i])

# --- Điều chỉnh layout và hiển thị ---
fig.suptitle('Phân phối lớp CKD gốc và sau khi chia Train/Test (Stratified)', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 1]) # Điều chỉnh layout tổng thể
plt.show() # Hiển thị figure chứa tất cả các subplot

# -------------------------------------------------------------
# Hoàn thành Bước 2.1
# -------------------------------------------------------------
print("\n\n--- Hoàn thành Section 2.1: Preparing the datasets cho CKD ---")
# ... (Phần thông báo cuối cùng giữ nguyên) ...
print(f"Đã tạo {len(datasets_ckd)} bộ dữ liệu train/test (tương ứng 4 tỷ lệ).")
print(f"Tổng cộng có {len(datasets_ckd) * 4} subsets dữ liệu.")
print("Các bộ dữ liệu được lưu trong dictionary 'datasets_ckd'.")
print("Ví dụ truy cập tập feature train của tỷ lệ 80/20:")
print("X_train_8020 = datasets_ckd['80/20']['feature_train']")
print("Ví dụ truy cập tập label test của tỷ lệ 60/40:")
print("y_test_6040 = datasets_ckd['60/40']['label_test']")