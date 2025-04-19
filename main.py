import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from tabulate import tabulate

# Đọc dữ liệu
data = pd.read_csv("data.csv", keep_default_na=False)

# Hiển thị tên các cột
columns_list = [[index + 1, column] for index, column in enumerate(data.columns.tolist())]
print("Danh sách các cột trong dữ liệu:")
print(tabulate(columns_list, headers=["STT", "Tên cột"], tablefmt="pretty"))
print()


print("Thông tin thống kê dữ liệu:")
print(data.info())
print("\nThống kê mô tả:")
print(data.describe())
print()

# Xử lý dữ liệu lỗi
print("Kiểm tra giá trị thiếu:")
print(data.isnull().sum())
data = data.dropna()

# Encode các cột phân loại
categorical_cols = [
    "gender",
    "region",
    "income_level",
    "smoking_status",
    "alcohol_consumption",
    "dietary_habits",
    "physical_activity",
    "air_pollution_exposure",
    "stress_level",
    "EKG_results",
]

data_encoded = pd.get_dummies(data, columns=categorical_cols)

# Tách đặc trưng và nhãn
X = data_encoded.drop("heart_attack", axis=1)
y = data_encoded["heart_attack"]

# ================================
# CHỌN GIẢM CHIỀU DỮ LIỆU (PCA / LDA / None)
# ================================
reduction_method = input("Chọn phương pháp giảm chiều (PCA/LDA/None): ").strip().lower()

if reduction_method == "pca":
    pca = PCA(n_components=0.8)  # Tự động chọn số chiều giữ 95% phương sai
    X = pca.fit_transform(X)
    print(f"PCA giữ lại {X.shape[1]} chiều (để giữ lại 95% phương sai).")

    # Vẽ biểu đồ phương sai tích lũy
    plt.figure(figsize=(8, 5))
    explained_variance = PCA().fit(X).explained_variance_ratio_.cumsum()
    plt.plot(explained_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.title("Cumulative Explained Variance - PCA")
    plt.xlabel("Số chiều giữ lại")
    plt.ylabel("Tỷ lệ phương sai tích lũy")
    plt.grid(True)
    plt.show()

elif reduction_method == "lda":
    lda = LDA(n_components=1)  # Vì có 2 lớp => tối đa là 1
    X = lda.fit_transform(X, y)
    print(f"LDA giảm chiều còn: {X.shape[1]} chiều.")

else:
    print("Không giảm chiều dữ liệu.")

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Huấn luyện mô hình Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = clf.predict(X_test)

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
headers = ["", "Dự đoán: Không bị", "Dự đoán: Bị"]
table = [
    ["Thực tế: Không bị", cm[0][0], cm[0][1]],
    ["Thực tế: Bị",        cm[1][0], cm[1][1]]
]
print("\nMa trận nhầm lẫn:")
print(tabulate(table, headers=headers, tablefmt="pretty"))

# Classification report
report_dict = classification_report(y_test, y_pred, output_dict=True)
table = []
total_support = len(y_test)
for label, metrics in report_dict.items():
    if isinstance(metrics, dict):
        row = [label]
        row += [f"{metrics[col]:.2f}" for col in ["precision", "recall", "f1-score", "support"]]
        table.append(row)
    elif label == "accuracy":
        table.append(["accuracy", "", "", f"{metrics:.2f}", f"{total_support}"])

headers = ["Label", "Precision", "Recall", "F1-Score", "Support"]
print("\nKết quả:")
print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

# Vẽ biểu đồ nếu muốn
user_input = input("\nBạn có muốn vẽ biểu đồ không? (Y/Else): ").strip().lower()
if user_input == 'y':
    class_counts = pd.Series(y_pred).value_counts()
    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar', color=['blue', 'orange'])
    plt.title('Số lượng dự đoán của các lớp (Heart Attack vs No Attack)')
    plt.xlabel('Lớp')
    plt.ylabel('Số lượng dự đoán')
    plt.xticks(rotation=0)
    plt.show()
else:
    print("Không vẽ biểu đồ.")
