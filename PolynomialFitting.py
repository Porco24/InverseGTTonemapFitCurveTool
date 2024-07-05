import numpy as np
import matplotlib.pyplot as plt

# 定义参数
P = 1.0
a = 1.0
m = 0.22
l = 0.4
c = 1.33
b = 0.0

# 定义原函数
def original_function(x):
    return (m * np.power(x/m, c) + b) * (1 - np.power(x/m, 2) * (3 - 2*x/m)) + a * (m + x - m) * (np.power(x/m, 2) * (3 - 2*x/m))

# 生成数据点
x_data = np.linspace(0, m, 500)  # 根据实际情况调整数据范围
x_data = x_data[x_data != 0]
y_data = original_function(x_data)

# 为提高拟合精度，增加关键点附近的数据密度
x_data_high_density = np.concatenate((np.linspace(0.0, 0.02, 500), np.linspace(0.02, 0.3, 500)))
y_data_high_density = original_function(x_data_high_density)

# 使用 numpy.polyfit 进行多项式拟合
degree = 5  # 选择多项式的阶数
coeffs = np.polyfit(y_data, x_data, degree)


# 定义拟合的反函数
def inverse_function(y):
    return np.polyval(coeffs, y)

# 输出拟合后的反函数解析表达式
def polynomial_expression(coeffs):
    terms = [f"{coeff:.6f}*x^{i}" if i > 0 else f"{coeff:.6f}" for i, coeff in enumerate(coeffs[::-1])]
    return " + ".join(terms)

print(f"拟合的反函数解析表达式: y = {polynomial_expression(coeffs)}")

# 测试拟合的反函数
y_test = np.linspace(min(y_data), max(y_data), 1000)
x_test = inverse_function(y_test)

# 绘制图像
plt.figure(figsize=(10, 6))

# 原函数图像
plt.plot(x_data, y_data, label='Original Function $f(x)$', color='blue')

# 拟合的反函数图像
plt.plot(y_test, x_test, label='Fitted Inverse Function $f^{-1}(y)$', linestyle='--', color='red')

# y = x 图像
plt.plot(y_test, y_test, label='$y = x$', linestyle=':', color='green')

plt.xlabel('x / y')
plt.ylabel('y / x')
plt.legend()
plt.title('Original Function and Fitted Inverse Function')
plt.grid(True)
plt.show()
