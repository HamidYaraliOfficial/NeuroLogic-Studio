# 🧠 NeuroLogic Studio - Neural Networks & Deep Learning Project

---

## 🇬🇧 English

# 🧠 NeuroLogic Studio

**A Professional Desktop Application for Neural Networks & Deep Learning Research**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.5+-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Neural Networks](https://img.shields.io/badge/Neural_Networks-Delta_Perceptron-red.svg)](https://en.wikipedia.org/wiki/Perceptron)
[![Hopfield](https://img.shields.io/badge/Hopfield-Network-orange.svg)](https://en.wikipedia.org/wiki/Hopfield_network)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Part 1: Delta Perceptron & Logic Gates](#part-1-delta-perceptron--logic-gates)
- [Part 2: Hopfield Network & SR Latch](#part-2-hopfield-network--sr-latch)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**NeuroLogic Studio** is a comprehensive desktop application developed for neural networks and deep learning research. This project implements two fundamental neural network architectures:

1. **Delta Perceptron** for solving logic gate problems
2. **Hopfield Network** for simulating SR Latch behavior

The application features a modern, professional PyQt6 interface with bilingual support (English/Persian) and customizable themes, making it an excellent tool for students, researchers, and AI enthusiasts.

---

## ✨ Features

### 🧩 **Core Features**

| Feature | Description |
|---------|-------------|
| **Delta Perceptron** | Single-layer perceptron with delta learning rule |
| **Hopfield Network** | Recurrent neural network for associative memory |
| **Logic Systems** | 3-input (Odd Parity) and 4-input (Majority) |
| **Activation Functions** | Sigmoid (Binary & Bipolar) |
| **Bias Integration** | SR Latch bias implementation |

### 🎨 **User Interface**

- **Bilingual Support**: Full English and Persian (فارسی) interface
- **Dual Themes**: Light and Dark mode with professional styling
- **Real-time Visualization**: Interactive loss curves and energy landscapes
- **Responsive Design**: Optimized for various screen resolutions
- **Tab-based Navigation**: Organized workflow for different experiments

### 📊 **Data Visualization**

- **Loss Curves**: Real-time training loss visualization
- **Energy Landscapes**: Hopfield network energy analysis
- **Truth Tables**: Automatic generation and display
- **State History**: Track network evolution step-by-step

### ⚙️ **Technical Capabilities**

- **Multi-threaded Training**: Non-blocking neural network training
- **Matplotlib Integration**: High-quality scientific plotting
- **JSON Serialization**: Save and load network configurations
- **Progress Tracking**: Real-time training progress indicators

---

## 🏗️ Technical Architecture

### **Neural Network Implementations**

#### **Delta Perceptron**
```python
class DeltaPerceptron:
    - Sigmoid activation (binary/bipolar)
    - Delta learning rule with momentum
    - Configurable learning rate and tolerance
    - Loss history tracking
Hopfield Network
python
class HopfieldNetwork:
    - Hebbian learning rule
    - Asynchronous update mechanism
    - Energy function calculation
    - Bias integration for SR Latch
Technology Stack
text
┌─────────────────────────────────────────────────────────┐
│                    NeuroLogic Studio                    │
├─────────────────────────────────────────────────────────┤
│  Frontend: PyQt6 (GUI Framework)                       │
│  Backend: NumPy (Numerical Computing)                   │
│  Visualization: Matplotlib (Scientific Plotting)        │
│  Threading: QThread (Async Processing)                  │
│  Data: JSON (Configuration Storage)                     │
└─────────────────────────────────────────────────────────┘
💻 Installation
Prerequisites
bash
Python 3.8 or higher
pip package manager
Installation Steps
bash
# Install required packages
pip install PyQt6 numpy matplotlib

# Verify installation
python -c "import PyQt6, numpy, matplotlib; print('All packages installed successfully!')"
Running the Application
bash
# Navigate to project directory
cd NeuroLogic-Studio

# Run the application
python main.py
📚 Usage Guide
Quick Start
Launch Application: Run python main.py

Select Language: Choose English or فارسی from top bar

Choose Theme: Toggle between Light/Dark mode

Select Part: Navigate between Part 1 and Part 2 tabs

Part 1: Delta Perceptron Training
Configuration Options
System: Choose between 3-Input (Odd Parity) or 4-Input (Majority)

Mode: Select Binary or Bipolar activation

Training Process
Select system type and activation mode

Click "Train Network" button

Observe real-time training progress

View truth table with computed outputs

Analyze loss curve for convergence

Expected Results
3-Input System: Not linearly separable → ~50% accuracy expected

4-Input System: Linearly separable → 100% accuracy achievable

Part 2: Hopfield Network & SR Latch
SR Latch Simulation
S (Set): Set the latch (S=1, R=0)

R (Reset): Reset the latch (S=0, R=1)

Hold: Maintain current state (S=0, R=0)

Invalid: Undefined state (S=1, R=1)

Hopfield Configuration
Initialize Hopfield network with stored patterns

Configure initial Q and Q̄ values

Apply SR Latch bias

Run asynchronous update

Observe state evolution and energy landscape

🧬 Mathematical Foundation
Delta Perceptron
Activation Function (Binary):

text
σ(z) = 1 / (1 + e^(-βz))
Activation Function (Bipolar):

text
σ(z) = 2 / (1 + e^(-βz)) - 1
Delta Learning Rule:

text
Δw = η · δ · x
Δb = η · δ
Hopfield Network
Hebbian Learning:

text
W_ij = (1/N) * Σ(p_i^(μ) · p_j^(μ))
Energy Function:

text
E = -½ Σ_ij W_ij s_i s_j - Σ_i b_i s_i
Update Rule:

text
s_i(t+1) = sign(Σ_j W_ij s_j(t) + b_i)
📁 Project Structure
text
NeuroLogic-Studio/
│
├── main.py                      # Main application file
├── README.md                    # This documentation
│
├── outputs/                     # Generated outputs
│   ├── part1_loss_curves.png    # Loss curve export
│   └── part2_energy_landscape.png # Energy landscape export
│
└── requirements.txt             # Python dependencies
🖼️ Screenshots
Part 1: Logic Gates Training
Truth table display for 3/4-input systems

Real-time loss curve plotting

Training progress with accuracy metrics

Part 2: SR Latch Simulation
Hopfield network initialization

SR Latch state control panel

Energy landscape visualization

State history tracking

🔬 Theoretical Background
Linear Separability
3-Input Odd Parity: XOR generalization - NOT linearly separable

4-Input Majority Function: Linearly separable - can be learned by perceptron

Hopfield Networks
Associative memory model

Energy minimization property

Stable states correspond to stored patterns

SR Latch implemented via bias vectors

🤝 Contributing
Contributions are welcome! Please follow these guidelines:

Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

Development Guidelines
Follow PEP 8 style guide

Add comments for complex logic

Update documentation for new features

Test changes thoroughly

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👏 Acknowledgments
PyQt6 team for excellent GUI framework

NumPy developers for numerical computing

Matplotlib contributors for visualization tools

All researchers in neural networks field

📧 Support
For issues, feature requests, or questions, please open an issue on GitHub.

Made with ❤️ for Neural Networks & Deep Learning Research

🇮🇷 فارسی
🧠 NeuroLogic Studio - پروژه شبکه‌های عصبی و یادگیری عمیق
https://img.shields.io/badge/Python-3.8+-blue.svg
https://img.shields.io/badge/PyQt6-6.5+-green.svg
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/%D8%B4%D8%A8%DA%A9%D9%87_%D8%B9%D8%B5%D8%A8%DB%8C-Perceptron-red.svg
https://img.shields.io/badge/Hopfield-%D8%B4%D8%A8%DA%A9%D9%87-orange.svg

📖 فهرست مطالب
معرفی

ویژگی‌ها

معماری فنی

نصب و راه‌اندازی

راهنمای استفاده

بخش ۱: پرسپترون دلتا و گیت‌های منطقی

بخش ۲: شبکه هوپفیلد و SR Latch

ساختار پروژه

تصاویر

مشارکت

مجوز

🎯 معرفی
NeuroLogic Studio یک برنامه دسکتاپ حرفه‌ای برای تحقیقات شبکه‌های عصبی و یادگیری عمیق است. این پروژه دو معماری اساسی شبکه عصبی را پیاده‌سازی می‌کند:

پرسپترون دلتا برای حل مسائل گیت‌های منطقی

شبکه هوپفیلد برای شبیه‌سازی SR Latch

این برنامه دارای رابط کاربری مدرن PyQt6 با پشتیبانی دو زبانه (انگلیسی/فارسی) و تم‌های قابل تنظیم است.

✨ ویژگی‌ها
🧩 ویژگی‌های اصلی
ویژگی	توضیحات
پرسپترون دلتا	پرسپترون تک‌لایه با قانون یادگیری دلتا
شبکه هوپفیلد	شبکه عصبی بازگشتی برای حافظه انجمنی
سیستم‌های منطقی	۳-ورودی (Odd Parity) و ۴-ورودی (Majority)
توابع فعال‌سازی	سیگموئید (باینری و بایپولار)
یکپارچه‌سازی بایاس	پیاده‌سازی بایاس SR Latch
🎨 رابط کاربری
پشتیبانی دو زبانه: رابط انگلیسی و فارسی کامل

تم‌های دوگانه: حالت روشن و تاریک با طراحی حرفه‌ای

تجسم بلادرنگ: منحنی‌های خطا و چشم‌انداز انرژی تعاملی

طراحی پاسخگو: بهینه‌سازی برای رزولوشن‌های مختلف صفحه

ناوبری مبتنی بر تب: گردش کار سازمان‌یافته برای آزمایش‌های مختلف

📊 تجسم داده‌ها
منحنی خطا: تجسم خطای آموزش در زمان واقعی

چشم‌انداز انرژی: تحلیل انرژی شبکه هوپفیلد

جدول حقیقت: تولید و نمایش خودکار

تاریخچه حالت: ردیابی گام به گام تکامل شبکه

⚙️ قابلیت‌های فنی
آموزش چند نخی: آموزش شبکه عصبی بدون مسدود کردن رابط

یکپارچه‌سازی متپلاتلیب: رسم نمودارهای علمی با کیفیت بالا

سریالیزیشن JSON: ذخیره و بارگذاری تنظیمات شبکه

ردیابی پیشرفت: نشانگرهای پیشرفت آموزش بلادرنگ

🏗️ معماری فنی
پیاده‌سازی شبکه‌های عصبی
پرسپترون دلتا
python
class DeltaPerceptron:
    - فعال‌سازی سیگموئید (باینری/بایپولار)
    - قانون یادگیری دلتا با ممنتوم
    - نرخ یادگیری و تلرانس قابل تنظیم
    - ردیابی تاریخچه خطا
شبکه هوپفیلد
python
class HopfieldNetwork:
    - قانون یادگیری هبی
    - مکانیزم بروزرسانی ناهمگام
    - محاسبه تابع انرژی
    - یکپارچه‌سازی بایاس برای SR Latch
💻 نصب و راه‌اندازی
پیش‌نیازها
bash
Python 3.8 یا بالاتر
مدیر بسته pip
مراحل نصب
bash
# نصب پکیج‌های مورد نیاز
pip install PyQt6 numpy matplotlib

# تأیید نصب
python -c "import PyQt6, numpy, matplotlib; print('تمامی پکیج‌ها با موفقیت نصب شدند!')"
اجرای برنامه
bash
# رفتن به دایرکتوری پروژه
cd NeuroLogic-Studio

# اجرای برنامه
python main.py
📚 راهنمای استفاده
شروع سریع
اجرای برنامه: python main.py را اجرا کنید

انتخاب زبان: انگلیسی یا فارسی را از نوار بالا انتخاب کنید

انتخاب تم: بین حالت روشن و تاریک جابجا شوید

انتخاب بخش: بین تب بخش ۱ و بخش ۲ جابجا شوید

بخش ۱: آموزش پرسپترون دلتا
گزینه‌های پیکربندی
سیستم: انتخاب بین ۳-ورودی (Odd Parity) یا ۴-ورودی (Majority)

حالت: انتخاب فعال‌سازی باینری یا بایپولار

فرآیند آموزش
نوع سیستم و حالت فعال‌سازی را انتخاب کنید

روی دکمه "آموزش شبکه" کلیک کنید

پیشرفت آموزش بلادرنگ را مشاهده کنید

جدول حقیقت با خروجی‌های محاسبه شده را ببینید

منحنی خطا را برای همگرایی تحلیل کنید

بخش ۲: شبکه هوپفیلد و SR Latch
شبیه‌سازی SR Latch
S (تنظیم): تنظیم latch (S=1, R=0)

R (بازنشانی): بازنشانی latch (S=0, R=1)

نگهداری: حفظ حالت فعلی (S=0, R=0)

نامعتبر: حالت تعریف نشده (S=1, R=1)

پیکربندی هوپفیلد
شبکه هوپفیلد را با الگوهای ذخیره شده مقداردهی کنید

مقادیر اولیه Q و Q̄ را تنظیم کنید

بایاس SR Latch را اعمال کنید

بروزرسانی ناهمگام را اجرا کنید

تکامل حالت و چشم‌انداز انرژی را مشاهده کنید

🧬 پایه ریاضی
پرسپترون دلتا
تابع فعال‌سازی (باینری):

text
σ(z) = 1 / (1 + e^(-βz))
تابع فعال‌سازی (بایپولار):

text
σ(z) = 2 / (1 + e^(-βz)) - 1
قانون یادگیری دلتا:

text
Δw = η · δ · x
Δb = η · δ
شبکه هوپفیلد
یادگیری هبی:

text
W_ij = (1/N) * Σ(p_i^(μ) · p_j^(μ))
تابع انرژی:

text
E = -½ Σ_ij W_ij s_i s_j - Σ_i b_i s_i
قانون بروزرسانی:

text
s_i(t+1) = sign(Σ_j W_ij s_j(t) + b_i)
📁 ساختار پروژه
text
NeuroLogic-Studio/
│
├── main.py                      # فایل اصلی برنامه
├── README.md                    # این مستندات
│
├── outputs/                     # خروجی‌های تولید شده
│   ├── part1_loss_curves.png    # خروجی منحنی خطا
│   └── part2_energy_landscape.png # خروجی چشم‌انداز انرژی
│
└── requirements.txt             # وابستگی‌های پایتون
🖼️ تصاویر
بخش ۱: آموزش گیت‌های منطقی
نمایش جدول حقیقت برای سیستم‌های ۳/۴-ورودی

رسم منحنی خطا بلادرنگ

پیشرفت آموزش با معیارهای دقت

بخش ۲: شبیه‌سازی SR Latch
مقداردهی شبکه هوپفیلد

پنل کنترل حالت SR Latch

تجسم چشم‌انداز انرژی

ردیابی تاریخچه حالت

🔬 پیشینه نظری
جدایی‌پذیری خطی
سیستم ۳-ورودی Odd Parity: تعمیم XOR - خطی‌جداپذیر نیست

تابع اکثریت ۴-ورودی: خطی‌جداپذیر - قابل یادگیری توسط پرسپترون

شبکه‌های هوپفیلد
مدل حافظه انجمنی

خاصیت کمینه‌سازی انرژی

حالت‌های پایدار مطابق با الگوهای ذخیره شده

پیاده‌سازی SR Latch از طریق بردارهای بایاس

🤝 مشارکت
مشارکت‌ها پذیرفته می‌شوند! لطفاً این راهنماها را دنبال کنید:

مخزن را Fork کنید

یک شاخه ویژگی ایجاد کنید

تغییرات خود را Commit کنید

به شاخه Push کنید

یک Pull Request باز کنید

راهنماهای توسعه
از راهنمای سبک PEP 8 پیروی کنید

برای منطق پیچیده توضیحات اضافه کنید

مستندات را برای ویژگی‌های جدید به‌روز کنید

تغییرات را به طور کامل تست کنید

📄 مجوز
این پروژه تحت مجوز MIT منتشر شده است - برای جزئیات فایل LICENSE را ببینید.

👏 قدردانی
تیم PyQt6 برای چارچوب GUI عالی

توسعه‌دهندگان NumPy برای محاسبات عددی

مشارکت‌کنندگان Matplotlib برای ابزارهای تجسم

همه محققان حوزه شبکه‌های عصبی

ساخته شده با ❤️ برای تحقیقات شبکه‌های عصبی و یادگیری عمیق

🇨🇳 中文
🧠 NeuroLogic Studio - 神经网络与深度学习项目
https://img.shields.io/badge/Python-3.8+-blue.svg
https://img.shields.io/badge/PyQt6-6.5+-green.svg
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-Delta_Perceptron-red.svg
https://img.shields.io/badge/Hopfield-%E7%BD%91%E7%BB%9C-orange.svg

📖 目录
概述

功能

技术架构

安装

使用指南

第一部分：Delta感知器与逻辑门

第二部分：Hopfield网络与SR锁存器

项目结构

截图

贡献

许可证

🎯 概述
NeuroLogic Studio 是一个为神经网络与深度学习研究开发的专业桌面应用程序。该项目实现了两种基本的神经网络架构：

Delta感知器 用于解决逻辑门问题

Hopfield网络 用于模拟SR锁存器行为

该应用程序具有现代化的PyQt6界面，支持双语（英语/中文）和可自定义主题，是学生、研究人员和AI爱好者的优秀工具。

✨ 功能
🧩 核心功能
功能	描述
Delta感知器	带Delta学习规则的单层感知器
Hopfield网络	用于联想记忆的循环神经网络
逻辑系统	3输入（奇偶校验）和4输入（多数表决）
激活函数	Sigmoid（二进制和双极性）
偏置集成	SR锁存器偏置实现
🎨 用户界面
双语支持: 完整的中文和English界面

双主题: 浅色和深色模式，专业样式

实时可视化: 交互式损失曲线和能量景观

响应式设计: 针对不同屏幕分辨率优化

标签导航: 为不同实验组织工作流程

📊 数据可视化
损失曲线: 实时训练损失可视化

能量景观: Hopfield网络能量分析

真值表: 自动生成和显示

状态历史: 逐步跟踪网络演化

⚙️ 技术能力
多线程训练: 非阻塞神经网络训练

Matplotlib集成: 高质量科学绘图

JSON序列化: 保存和加载网络配置

进度跟踪: 实时训练进度指示器

🏗️ 技术架构
神经网络实现
Delta感知器
python
class DeltaPerceptron:
    - Sigmoid激活（二进制/双极性）
    - 带动量的Delta学习规则
    - 可配置学习率和容差
    - 损失历史跟踪
Hopfield网络
python
class HopfieldNetwork:
    - Hebbian学习规则
    - 异步更新机制
    - 能量函数计算
    - SR锁存器偏置集成
💻 安装
前提条件
bash
Python 3.8 或更高版本
pip 包管理器
安装步骤
bash
# 安装所需包
pip install PyQt6 numpy matplotlib

# 验证安装
python -c "import PyQt6, numpy, matplotlib; print('所有包安装成功！')"
运行应用程序
bash
# 进入项目目录
cd NeuroLogic-Studio

# 运行应用程序
python main.py
📚 使用指南
快速开始
启动应用程序: 运行 python main.py

选择语言: 从顶部栏选择中文或English

选择主题: 在浅色/深色模式之间切换

选择部分: 在第一部分和第二部分标签之间切换

第一部分：Delta感知器训练
配置选项
系统: 选择3输入（奇偶校验）或4输入（多数表决）

模式: 选择二进制或双极性激活

训练过程
选择系统类型和激活模式

点击"训练网络"按钮

观察实时训练进度

查看带计算输出的真值表

分析损失曲线的收敛情况

第二部分：Hopfield网络与SR锁存器
SR锁存器仿真
S（置位）: 置位锁存器（S=1, R=0）

R（复位）: 复位锁存器（S=0, R=1）

保持: 保持当前状态（S=0, R=0）

无效: 未定义状态（S=1, R=1）

Hopfield配置
使用存储模式初始化Hopfield网络

配置初始Q和Q̄值

应用SR锁存器偏置

运行异步更新

观察状态演化和能量景观

🧬 数学基础
Delta感知器
激活函数（二进制）:

text
σ(z) = 1 / (1 + e^(-βz))
激活函数（双极性）:

text
σ(z) = 2 / (1 + e^(-βz)) - 1
Delta学习规则:

text
Δw = η · δ · x
Δb = η · δ
Hopfield网络
Hebbian学习:

text
W_ij = (1/N) * Σ(p_i^(μ) · p_j^(μ))
能量函数:

text
E = -½ Σ_ij W_ij s_i s_j - Σ_i b_i s_i
更新规则:

text
s_i(t+1) = sign(Σ_j W_ij s_j(t) + b_i)
📁 项目结构
text
NeuroLogic-Studio/
│
├── main.py                      # 主应用程序文件
├── README.md                    # 本文档
│
├── outputs/                     # 生成的输出
│   ├── part1_loss_curves.png    # 损失曲线导出
│   └── part2_energy_landscape.png # 能量景观导出
│
└── requirements.txt             # Python依赖项
🖼️ 截图
第一部分：逻辑门训练
3/4输入系统的真值表显示

实时损失曲线绘制

带精度指标的训练进度

第二部分：SR锁存器仿真
Hopfield网络初始化

SR锁存器状态控制面板

能量景观可视化

状态历史跟踪

🔬 理论基础
线性可分性
3输入奇偶校验: XOR泛化 - 不是线性可分的

4输入多数表决函数: 线性可分 - 可以被感知器学习

Hopfield网络
联想记忆模型

能量最小化性质

稳定状态对应存储模式

通过偏置向量实现SR锁存器

🤝 贡献
欢迎贡献！请遵循以下指南：

Fork仓库

创建功能分支

提交更改

推送到分支

打开Pull Request

开发指南
遵循PEP 8风格指南

为复杂逻辑添加注释

更新新功能的文档

彻底测试更改

📄 许可证
本项目根据MIT许可证发布 - 详情请参阅LICENSE文件。

👏 致谢
PyQt6团队提供的优秀GUI框架

NumPy开发者提供的数值计算

Matplotlib贡献者提供的可视化工具

神经网络领域的所有研究人员

用 ❤️ 为神经网络与深度学习研究制作