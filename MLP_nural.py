# ==================== MLP - Multi-Layer Perceptron ====================
# การจดจำตัวเลข (Digit Recognition) ด้วย Neural Network บน MNIST dataset
#
# MLP คืออะไร?
# - MLP = โครงข่ายประสาทเทียมแบบหลายชั้น (Neural Network)
# - มี 3 ส่วน:
#     1. Input Layer  : รับข้อมูลเข้า (เช่น 784 pixels จากรูป 28x28)
#     2. Hidden Layer : ประมวลผล/เรียนรู้ patterns (มีได้หลายชั้น, แต่ละชั้นมีหลาย neurons)
#     3. Output Layer : ผลลัพธ์สุดท้าย (เช่น 10 neurons สำหรับตัวเลข 0-9)
# - แต่ละ neuron เชื่อมกับ neurons ในชั้นถัดไปด้วย "weight" (น้ำหนัก)
# - เรียนรู้โดยปรับ weight ให้ทำนายได้แม่นยำขึ้น (Backpropagation)
#
# Flow: รูปตัวเลข (28x28) → Flatten (784) → MLP → ทำนายตัวเลข 0-9

# ==================== 1. Import Libraries ====================
from scipy.io import loadmat                    # อ่านไฟล์ .mat (MATLAB)
import matplotlib.pyplot as plt                  # สร้างกราฟและแสดงรูป
import numpy as np                               # จัดการ array
from sklearn.neural_network import MLPClassifier # โมเดล MLP
from sklearn.metrics import accuracy_score       # วัดความแม่นยำ

# ==================== 2. Load Dataset ====================
# MNIST = ชุดข้อมูลรูปตัวเลข 0-9 ขนาด 28x28 (70,000 รูป)
mnist_data = loadmat('mnist-original.mat')

# จัดรูปแบบข้อมูล
mnist = {
    'data': mnist_data['data'].T,    # .T = transpose ให้แต่ละแถวคือ 1 รูป (784 pixels)
    'target': mnist_data['label'][0]  # ตัวเลขที่ถูกต้อง 0-9
}

# ==================== 3. เตรียมข้อมูล ====================
x, y = mnist['data'], mnist['target']

# Shuffle ข้อมูล: สุ่มลำดับเพื่อไม่ให้โมเดลเรียนรู้ตามลำดับ
shuffle_idx = np.random.permutation(70000)
x, y = x[shuffle_idx], y[shuffle_idx]

# แบ่ง Train/Test: 60,000 train, 10,000 test
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

# ==================== 4. สร้างและ Train โมเดล ====================
# MLPClassifier default: 1 hidden layer, 100 neurons
model = MLPClassifier()
model.fit(x_train, y_train)  # Train โมเดล (อาจใช้เวลาหลายนาที)

# ==================== 5. Predict & Evaluate ====================
y_pred = model.predict(x_test)  # ทำนายจาก test set

# วัดความแม่นยำ
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# ==================== 6. แสดงผลลัพธ์ ====================
# สร้าง grid 10x10 = 100 รูป
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={"xticks": [], "yticks": []},
                         gridspec_kw={"wspace": 0.1, "hspace": 0.1})

# วนลูปแสดงรูป + ผลทำนาย
for i, ax in enumerate(axes.flat):
    # แสดงรูปตัวเลข (reshape 784 → 28x28)
    ax.imshow(x_test[i].reshape(28, 28), cmap='binary')
    
    # แสดงค่าจริง (มุมซ้ายล่าง)
    ax.text(0.05, 0.05, str(int(y_test[i])), transform=ax.transAxes, color='black')
    
    # แสดงค่าทำนาย (มุมขวาล่าง) - เขียว=ถูก, แดง=ผิด
    color = 'green' if y_pred[i] == y_test[i] else 'red'
    ax.text(0.75, 0.05, str(int(y_pred[i])), transform=ax.transAxes, color=color)

plt.show()
# สังเกต: ตัวเลขสีเขียว = โมเดลทายถูก, สีแดง = ทายผิด