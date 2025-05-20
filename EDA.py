from matplotlib import pyplot as plt
import numpy as np

# Dữ liệu từ model_acc
model_acc = { 
    "add" : {"overall_accuracy" : 85.39, "color_accuracy" : 83.12, "brand_accuracy" : 94.30, "car_name_accuracy": 78.75},
    "co-attention" : {"overall_accuracy": 84.46, "color_accuracy" : 83.64, "brand_accuracy" : 93.95, "car_name_accuracy": 75.79},
    "concat" : {"overall_accuracy": 84.36, "color_accuracy" : 82.95, "brand_accuracy" : 94.46, "car_name_accuracy": 75.49},
    "gated-fusion" : {"overall_accuracy": 85.76, "color_accuracy" : 84.24, "brand_accuracy" : 94.76, "car_name_accuracy": 78.29},
    "multiplication" : {"overall_accuracy": 84.63, "color_accuracy" : 83.70, "brand_accuracy" : 94.99, "car_name_accuracy": 75.20} 
}

# Chuẩn bị dữ liệu
models = list(model_acc.keys())
metrics = ["overall_accuracy", "color_accuracy", "brand_accuracy", "car_name_accuracy"]
metric_names = ["Overall", "Color", "Brand", "Car Name"]
num_metrics = len(metrics)

# Tạo dữ liệu cho radar chart
data = np.array([[model_acc[model][metric] for metric in metrics] for model in models])

# Tạo góc cho radar chart
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]  # Đóng vòng radar

# Khởi tạo biểu đồ
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Vẽ từng model
for i, model in enumerate(models):
    values = data[i].tolist()
    values += values[:1]  # Đóng vòng
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
    ax.fill(angles, values, alpha=0.25)

# Tùy chỉnh biểu đồ
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_names)
ax.set_title('So sánh độ chính xác của các mô hình Fusion (Radar Chart)', size=15, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Thêm lưới và giới hạn trục y
ax.yaxis.grid(True)
ax.set_ylim(70, 100)  # Giới hạn trục y từ 70-100% cho phù hợp dữ liệu

plt.tight_layout()
plt.show()