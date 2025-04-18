import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tabulate import tabulate

# Đọc dữ liệu
data = pd.read_csv("data.csv")

# In cột để xác minh
columns_list = [[index + 1, column] for index, column in enumerate(data.columns.tolist())]
print("Danh sách các cột trong dữ liệu:")
print(tabulate(columns_list, headers=["STT", "Tên cột"], tablefmt="pretty"))
print()

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

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Huấn luyện mô hình Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = clf.predict(X_test)
# In Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
headers = ["", "Dự đoán: Không bị", "Dự đoán: Bị"]
table = [
    ["Thực tế: Không bị", cm[0][0], cm[0][1]],
    ["Thực tế: Bị",        cm[1][0], cm[1][1]]
]
print("Ma trận nhầm lẫn:")
print(tabulate(table, headers=headers, tablefmt="pretty"))
print()
# Lấy classification_report dưới dạng dictionary
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
print("Kết quả:")
print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
print()

# Yêu cầu người dùng nhập lựa chọn
user_input = input("Bạn có muốn vẽ biểu đồ không? (Y/Else): ").strip().lower()

if user_input == 'y':
    # Tạo đồ thị biểu diễn số lượng dự đoán cho mỗi lớp (No Attack, Heart Attack)
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

