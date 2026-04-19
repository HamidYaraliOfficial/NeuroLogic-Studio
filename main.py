# main.py

import sys
import numpy as np
from itertools import product
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QTableWidget, QTableWidgetItem, QTextEdit, QTabWidget,
                             QSpinBox, QGroupBox, QGridLayout, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ═══════════════════════════════════════════════════════════════
# LOGIC & NEURAL NETWORK CLASSES
# ═══════════════════════════════════════════════════════════════

def generate_inputs(n_inputs, mode='binary'):
    combos = list(product([0, 1], repeat=n_inputs))
    X = np.array(combos, dtype=float)
    if mode == 'bipolar':
        X = 2 * X - 1
    return X

def logic_3input(A, B, C):
    return int((A and not B and not C) or (not A and B and not C) or 
               (not A and not B and C) or (A and B and C))

def logic_4input(A, B, C, D):
    return int((A and B and C) or (A and B and D) or 
               (A and C and D) or (B and C and D))

def get_targets(X_binary, n_inputs):
    targets = []
    for row in X_binary:
        row = row.astype(int)
        if n_inputs == 3:
            targets.append(logic_3input(*row))
        else:
            targets.append(logic_4input(*row))
    return np.array(targets, dtype=float)

def sigmoid_binary(x, beta=1.0):
    return 1.0 / (1.0 + np.exp(-beta * x))

def sigmoid_bipolar(x, beta=1.0):
    return 2.0 / (1.0 + np.exp(-beta * x)) - 1.0

def sigmoid_binary_deriv(out):
    return out * (1.0 - out)

def sigmoid_bipolar_deriv(out):
    return 0.5 * (1.0 - out ** 2)

class DeltaPerceptron:
    def __init__(self, n_inputs, mode='binary', lr=0.1, max_epochs=5000, tol=1e-4):
        self.mode = mode
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        np.random.seed(42)
        self.w = np.random.randn(n_inputs) * 0.1
        self.b = np.random.randn() * 0.1
        self.loss_history = []

    def _activate(self, net):
        if self.mode == 'binary':
            return sigmoid_binary(net)
        else:
            return sigmoid_bipolar(net)

    def _deriv(self, out):
        if self.mode == 'binary':
            return sigmoid_binary_deriv(out)
        else:
            return sigmoid_bipolar_deriv(out)

    def _target_transform(self, Y):
        if self.mode == 'bipolar':
            return 2.0 * Y - 1.0
        return Y

    def fit(self, X, Y, callback=None):
        T = self._target_transform(Y)
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            for xi, ti in zip(X, T):
                net = np.dot(self.w, xi) + self.b
                out = self._activate(net)
                err = ti - out
                delta = err * self._deriv(out)
                self.w += self.lr * delta * xi
                self.b += self.lr * delta
                total_loss += 0.5 * err ** 2
            self.loss_history.append(total_loss)
            if callback and epoch % 10 == 0:
                callback(epoch, total_loss)
            if total_loss < self.tol:
                return epoch + 1
        return self.max_epochs

    def predict(self, X):
        preds = []
        for xi in X:
            net = np.dot(self.w, xi) + self.b
            out = self._activate(net)
            if self.mode == 'binary':
                preds.append(1 if out >= 0.5 else 0)
            else:
                preds.append(1 if out >= 0.0 else 0)
        return np.array(preds)

    def accuracy(self, X, Y):
        preds = self.predict(X)
        return np.mean(preds == Y.astype(int)) * 100

class HopfieldNetwork:
    def __init__(self, n=2):
        self.n = n
        self.W = np.zeros((n, n))

    def train(self, patterns):
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            p = np.array(p, dtype=float)
            self.W += np.outer(p, p)
        self.W /= self.n
        np.fill_diagonal(self.W, 0)

    def energy(self, state, bias=None):
        s = np.array(state, dtype=float)
        e = -0.5 * s @ self.W @ s
        if bias is not None:
            e -= np.dot(bias, s)
        return e

    def update_async(self, state, bias=None, max_iter=20):
        s = np.array(state, dtype=float).copy()
        history = [s.copy()]
        bias = np.zeros(self.n) if bias is None else np.array(bias, dtype=float)
        for iteration in range(max_iter):
            s_prev = s.copy()
            for i in range(self.n):
                net_i = np.dot(self.W[i], s) + bias[i]
                s[i] = 1.0 if net_i >= 0 else -1.0
            history.append(s.copy())
            if np.array_equal(s, s_prev):
                return s, history, True
        return s, history, False

def sr_bias(S, R, strength=2.0):
    s_bip = 2 * S - 1
    r_bip = 2 * R - 1
    b_Q = strength * (s_bip - r_bip) / 2.0
    b_Qbar = strength * (r_bip - s_bip) / 2.0
    return np.array([b_Q, b_Qbar])

# ═══════════════════════════════════════════════════════════════
# TRAINING THREAD
# ═══════════════════════════════════════════════════════════════

class TrainingThread(QThread):
    progress = pyqtSignal(int, float)
    finished = pyqtSignal(object, int, float)

    def __init__(self, n_inputs, mode, X, Y):
        super().__init__()
        self.n_inputs = n_inputs
        self.mode = mode
        self.X = X
        self.Y = Y

    def run(self):
        model = DeltaPerceptron(self.n_inputs, mode=self.mode, lr=0.1, max_epochs=10000, tol=1e-5)
        epochs = model.fit(self.X, self.Y, callback=self.progress.emit)
        acc = model.accuracy(self.X, self.Y)
        self.finished.emit(model, epochs, acc)

# ═══════════════════════════════════════════════════════════════
# MATPLOTLIB CANVAS
# ═══════════════════════════════════════════════════════════════

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

# ═══════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lang = 'en'
        self.theme = 'light'
        self.models = {}
        self.hopfield = None
        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        self.setWindowTitle('Neural Networks & Deep Learning - Project 1')
        self.setGeometry(100, 100, 1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top bar
        top_bar = QHBoxLayout()
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(['English', 'فارسی'])
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        
        self.theme_btn = QPushButton('🌙 Dark')
        self.theme_btn.clicked.connect(self.toggle_theme)
        
        top_bar.addWidget(QLabel('Language:'))
        top_bar.addWidget(self.lang_combo)
        top_bar.addStretch()
        top_bar.addWidget(self.theme_btn)
        
        main_layout.addLayout(top_bar)

        # Tabs
        self.tabs = QTabWidget()
        self.tab1 = self.create_part1_tab()
        self.tab2 = self.create_part2_tab()
        
        self.tabs.addTab(self.tab1, 'Part 1: Logic Gates')
        self.tabs.addTab(self.tab2, 'Part 2: SR Latch')
        
        main_layout.addWidget(self.tabs)

    def create_part1_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left panel
        left_panel = QVBoxLayout()
        
        config_group = QGroupBox('Configuration')
        config_layout = QGridLayout()
        
        config_layout.addWidget(QLabel('System:'), 0, 0)
        self.system_combo = QComboBox()
        self.system_combo.addItems(['3-Input', '4-Input'])
        config_layout.addWidget(self.system_combo, 0, 1)
        
        config_layout.addWidget(QLabel('Mode:'), 1, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Binary', 'Bipolar'])
        config_layout.addWidget(self.mode_combo, 1, 1)
        
        self.train_btn = QPushButton('Train Network')
        self.train_btn.clicked.connect(self.train_network)
        config_layout.addWidget(self.train_btn, 2, 0, 1, 2)
        
        self.progress_bar = QProgressBar()
        config_layout.addWidget(self.progress_bar, 3, 0, 1, 2)
        
        config_group.setLayout(config_layout)
        left_panel.addWidget(config_group)
        
        # Truth table
        table_group = QGroupBox('Truth Table')
        table_layout = QVBoxLayout()
        self.truth_table = QTableWidget()
        table_layout.addWidget(self.truth_table)
        table_group.setLayout(table_layout)
        left_panel.addWidget(table_group)
        
        # Results
        results_group = QGroupBox('Results')
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        left_panel.addWidget(results_group)
        
        layout.addLayout(left_panel, 1)

        # Right panel - Graph
        right_panel = QVBoxLayout()
        graph_group = QGroupBox('Loss Curve')
        graph_layout = QVBoxLayout()
        self.canvas1 = MplCanvas(self, width=6, height=6)
        graph_layout.addWidget(self.canvas1)
        graph_group.setLayout(graph_layout)
        right_panel.addWidget(graph_group)
        
        layout.addLayout(right_panel, 1)

        return widget

    def create_part2_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left panel
        left_panel = QVBoxLayout()
        
        # SR Control
        sr_group = QGroupBox('SR Latch Control')
        sr_layout = QGridLayout()
        
        sr_layout.addWidget(QLabel('S (Set):'), 0, 0)
        self.s_spin = QSpinBox()
        self.s_spin.setRange(0, 1)
        sr_layout.addWidget(self.s_spin, 0, 1)
        
        sr_layout.addWidget(QLabel('R (Reset):'), 1, 0)
        self.r_spin = QSpinBox()
        self.r_spin.setRange(0, 1)
        sr_layout.addWidget(self.r_spin, 1, 1)
        
        sr_layout.addWidget(QLabel('Initial Q:'), 2, 0)
        self.q_spin = QSpinBox()
        self.q_spin.setRange(-1, 1)
        self.q_spin.setSingleStep(2)
        self.q_spin.setValue(1)
        sr_layout.addWidget(self.q_spin, 2, 1)
        
        sr_layout.addWidget(QLabel('Initial Q̄:'), 3, 0)
        self.qbar_spin = QSpinBox()
        self.qbar_spin.setRange(-1, 1)
        self.qbar_spin.setSingleStep(2)
        self.qbar_spin.setValue(-1)
        sr_layout.addWidget(self.qbar_spin, 3, 1)
        
        self.init_hopfield_btn = QPushButton('Initialize Hopfield')
        self.init_hopfield_btn.clicked.connect(self.init_hopfield)
        sr_layout.addWidget(self.init_hopfield_btn, 4, 0, 1, 2)
        
        self.run_sr_btn = QPushButton('Run SR Latch')
        self.run_sr_btn.clicked.connect(self.run_sr_latch)
        self.run_sr_btn.setEnabled(False)
        sr_layout.addWidget(self.run_sr_btn, 5, 0, 1, 2)
        
        sr_group.setLayout(sr_layout)
        left_panel.addWidget(sr_group)
        
        # State history
        history_group = QGroupBox('State History')
        history_layout = QVBoxLayout()
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(['Step', 'Q', 'Q̄'])
        history_layout.addWidget(self.history_table)
        history_group.setLayout(history_layout)
        left_panel.addWidget(history_group)
        
        # SR Results
        sr_results_group = QGroupBox('Results')
        sr_results_layout = QVBoxLayout()
        self.sr_results_text = QTextEdit()
        self.sr_results_text.setReadOnly(True)
        self.sr_results_text.setMaximumHeight(120)
        sr_results_layout.addWidget(self.sr_results_text)
        sr_results_group.setLayout(sr_results_layout)
        left_panel.addWidget(sr_results_group)
        
        layout.addLayout(left_panel, 1)

        # Right panel - Energy landscape
        right_panel = QVBoxLayout()
        energy_group = QGroupBox('Energy Landscape')
        energy_layout = QVBoxLayout()
        self.canvas2 = MplCanvas(self, width=6, height=6)
        energy_layout.addWidget(self.canvas2)
        energy_group.setLayout(energy_layout)
        right_panel.addWidget(energy_group)
        
        layout.addLayout(right_panel, 1)

        return widget

    def change_language(self, index):
        self.lang = 'fa' if index == 1 else 'en'
        self.update_texts()

    def toggle_theme(self):
        self.theme = 'dark' if self.theme == 'light' else 'light'
        self.apply_theme()

    def apply_theme(self):
        if self.theme == 'dark':
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #1e1e1e; color: #e0e0e0; }
                QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 10px; padding-top: 10px; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
                QPushButton { background-color: #0d47a1; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
                QPushButton:hover { background-color: #1565c0; }
                QPushButton:pressed { background-color: #0a3d91; }
                QComboBox, QSpinBox { background-color: #2d2d2d; color: #e0e0e0; border: 1px solid #555; padding: 5px; border-radius: 3px; }
                QTableWidget { background-color: #2d2d2d; color: #e0e0e0; gridline-color: #555; }
                QHeaderView::section { background-color: #0d47a1; color: white; padding: 5px; border: none; }
                QTextEdit { background-color: #2d2d2d; color: #e0e0e0; border: 1px solid #555; }
                QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; }
                QProgressBar::chunk { background-color: #0d47a1; }
                QLabel { color: #e0e0e0; }
            """)
            self.theme_btn.setText('☀️ Light')
            self.canvas1.fig.patch.set_facecolor('#1e1e1e')
            self.canvas1.axes.set_facecolor('#2d2d2d')
            self.canvas1.axes.tick_params(colors='#e0e0e0')
            self.canvas1.axes.spines['bottom'].set_color('#555')
            self.canvas1.axes.spines['top'].set_color('#555')
            self.canvas1.axes.spines['left'].set_color('#555')
            self.canvas1.axes.spines['right'].set_color('#555')
            self.canvas2.fig.patch.set_facecolor('#1e1e1e')
            self.canvas2.axes.set_facecolor('#2d2d2d')
            self.canvas2.axes.tick_params(colors='#e0e0e0')
            self.canvas2.axes.spines['bottom'].set_color('#555')
            self.canvas2.axes.spines['top'].set_color('#555')
            self.canvas2.axes.spines['left'].set_color('#555')
            self.canvas2.axes.spines['right'].set_color('#555')
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #f5f5f5; color: #212121; }
                QGroupBox { border: 1px solid #bdbdbd; border-radius: 5px; margin-top: 10px; padding-top: 10px; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
                QPushButton { background-color: #1976d2; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
                QPushButton:hover { background-color: #1e88e5; }
                QPushButton:pressed { background-color: #1565c0; }
                QComboBox, QSpinBox { background-color: white; color: #212121; border: 1px solid #bdbdbd; padding: 5px; border-radius: 3px; }
                QTableWidget { background-color: white; color: #212121; gridline-color: #e0e0e0; }
                QHeaderView::section { background-color: #1976d2; color: white; padding: 5px; border: none; }
                QTextEdit { background-color: white; color: #212121; border: 1px solid #bdbdbd; }
                QProgressBar { border: 1px solid #bdbdbd; border-radius: 3px; text-align: center; }
                QProgressBar::chunk { background-color: #1976d2; }
                QLabel { color: #212121; }
            """)
            self.theme_btn.setText('🌙 Dark')
            self.canvas1.fig.patch.set_facecolor('#f5f5f5')
            self.canvas1.axes.set_facecolor('white')
            self.canvas1.axes.tick_params(colors='#212121')
            self.canvas1.axes.spines['bottom'].set_color('#bdbdbd')
            self.canvas1.axes.spines['top'].set_color('#bdbdbd')
            self.canvas1.axes.spines['left'].set_color('#bdbdbd')
            self.canvas1.axes.spines['right'].set_color('#bdbdbd')
            self.canvas2.fig.patch.set_facecolor('#f5f5f5')
            self.canvas2.axes.set_facecolor('white')
            self.canvas2.axes.tick_params(colors='#212121')
            self.canvas2.axes.spines['bottom'].set_color('#bdbdbd')
            self.canvas2.axes.spines['top'].set_color('#bdbdbd')
            self.canvas2.axes.spines['left'].set_color('#bdbdbd')
            self.canvas2.axes.spines['right'].set_color('#bdbdbd')
        
        self.canvas1.draw()
        self.canvas2.draw()

    def update_texts(self):
        if self.lang == 'fa':
            self.tabs.setTabText(0, 'بخش ۱: گیت‌های منطقی')
            self.tabs.setTabText(1, 'بخش ۲: SR Latch')
            self.train_btn.setText('آموزش شبکه')
            self.init_hopfield_btn.setText('مقداردهی Hopfield')
            self.run_sr_btn.setText('اجرای SR Latch')
        else:
            self.tabs.setTabText(0, 'Part 1: Logic Gates')
            self.tabs.setTabText(1, 'Part 2: SR Latch')
            self.train_btn.setText('Train Network')
            self.init_hopfield_btn.setText('Initialize Hopfield')
            self.run_sr_btn.setText('Run SR Latch')

    def train_network(self):
        n_inputs = 3 if self.system_combo.currentText() == '3-Input' else 4
        mode = self.mode_combo.currentText().lower()
        
        X_bin = generate_inputs(n_inputs, 'binary')
        X = generate_inputs(n_inputs, mode)
        Y = get_targets(X_bin, n_inputs)
        
        self.update_truth_table(X, Y, n_inputs, mode)
        
        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.results_text.clear()
        
        self.thread = TrainingThread(n_inputs, mode, X, Y)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.training_finished)
        self.thread.start()

    def update_progress(self, epoch, loss):
        progress = min(int((epoch / 10000) * 100), 100)
        self.progress_bar.setValue(progress)

    def training_finished(self, model, epochs, acc):
        self.train_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        
        key = f"{self.system_combo.currentText()}_{self.mode_combo.currentText()}"
        self.models[key] = model
        
        n_inputs = 3 if '3-Input' in key else 4
        
        if self.lang == 'fa':
            result_text = f"آموزش کامل شد!\nدقت: {acc:.2f}%\nتعداد Epoch: {epochs}\n"
            if n_inputs == 3:
                result_text += "\nتوجه: سیستم 3-ورودی (Odd Parity) خطی‌جداپذیر نیست.\nیک Perceptron تک‌لایه نمی‌تواند آن را کامل یاد بگیرد.\nدقت ~50% طبیعی است."
            else:
                result_text += "\nسیستم 4-ورودی (Majority) خطی‌جداپذیر است."
        else:
            result_text = f"Training completed!\nAccuracy: {acc:.2f}%\nEpochs: {epochs}\n"
            if n_inputs == 3:
                result_text += "\nNote: 3-input system (Odd Parity) is NOT linearly separable.\nA single-layer perceptron cannot learn it perfectly.\nAccuracy ~50% is expected."
            else:
                result_text += "\n4-input system (Majority) is linearly separable."
        
        self.results_text.setText(result_text)
        
        self.canvas1.axes.clear()
        self.canvas1.axes.plot(model.loss_history, color='#1976d2', linewidth=2)
        self.canvas1.axes.set_xlabel('Epoch', color='#e0e0e0' if self.theme == 'dark' else '#212121')
        self.canvas1.axes.set_ylabel('Loss', color='#e0e0e0' if self.theme == 'dark' else '#212121')
        self.canvas1.axes.set_title(f'Loss Curve - {key}', color='#e0e0e0' if self.theme == 'dark' else '#212121')
        self.canvas1.axes.set_yscale('log')
        self.canvas1.axes.grid(True, alpha=0.3)
        self.canvas1.draw()
        
        try:
            self.canvas1.fig.savefig('part1_loss_curves.png', dpi=150, bbox_inches='tight', 
                                     facecolor=self.canvas1.fig.get_facecolor())
        except Exception as e:
            print(f"Error saving part1_loss_curves.png: {e}")

    def update_truth_table(self, X, Y, n_inputs, mode):
        labels = list('ABCD')[:n_inputs]
        self.truth_table.setRowCount(len(X))
        self.truth_table.setColumnCount(n_inputs + 1)
        self.truth_table.setHorizontalHeaderLabels(labels + ['OUT'])
        
        for i, (row, target) in enumerate(zip(X, Y)):
            for j, val in enumerate(row):
                if mode == 'bipolar':
                    item = QTableWidgetItem(f"{int(val):+d}")
                else:
                    item = QTableWidgetItem(str(int(val)))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.truth_table.setItem(i, j, item)
            
            out_item = QTableWidgetItem(str(int(target)))
            out_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.truth_table.setItem(i, n_inputs, out_item)

    def init_hopfield(self):
        patterns = [np.array([+1., -1.]), np.array([-1., +1.])]
        self.hopfield = HopfieldNetwork(n=2)
        self.hopfield.train(patterns)
        
        if self.lang == 'fa':
            self.sr_results_text.setText("شبکه Hopfield مقداردهی شد.\nالگوهای ذخیره شده: [+1,-1] و [-1,+1]")
        else:
            self.sr_results_text.setText("Hopfield network initialized.\nStored patterns: [+1,-1] and [-1,+1]")
        
        self.run_sr_btn.setEnabled(True)
        self.plot_energy_landscape()

    def run_sr_latch(self):
        if self.hopfield is None:
            return
        
        S = self.s_spin.value()
        R = self.r_spin.value()
        Q = float(self.q_spin.value())
        Qbar = float(self.qbar_spin.value())
        
        initial = np.array([Q, Qbar])
        bias = sr_bias(S, R)
        
        final, history, converged = self.hopfield.update_async(initial, bias=bias)
        
        self.history_table.setRowCount(len(history))
        for i, state in enumerate(history):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.history_table.setItem(i, 1, QTableWidgetItem(f"{int(state[0]):+d}"))
            self.history_table.setItem(i, 2, QTableWidgetItem(f"{int(state[1]):+d}"))
        
        energy = self.hopfield.energy(final, bias)
        
        if self.lang == 'fa':
            mode_str = {(0,0): 'نگهداری', (1,0): 'تنظیم', (0,1): 'بازنشانی', (1,1): 'نامعتبر'}[(S,R)]
            result = f"حالت: {mode_str}\nحالت نهایی: Q={int(final[0]):+d}, Q̄={int(final[1]):+d}\n"
            result += f"همگرا: {'بله' if converged else 'خیر'}\nانرژی: {energy:.3f}"
        else:
            mode_str = {(0,0): 'HOLD', (1,0): 'SET', (0,1): 'RESET', (1,1): 'INVALID'}[(S,R)]
            result = f"Mode: {mode_str}\nFinal state: Q={int(final[0]):+d}, Q̄={int(final[1]):+d}\n"
            result += f"Converged: {'Yes' if converged else 'No'}\nEnergy: {energy:.3f}"
        
        self.sr_results_text.setText(result)

    def plot_energy_landscape(self):
        if self.hopfield is None:
            return
        
        states = [np.array([+1., -1.]), np.array([-1., +1.]), 
                  np.array([+1., +1.]), np.array([-1., -1.])]
        state_labels = ['+1,-1', '-1,+1', '+1,+1', '-1,-1']
        
        energies_no_bias = [self.hopfield.energy(s) for s in states]
        
        x = np.arange(len(states))
        bar_color = '#1976d2' if self.theme == 'light' else '#42a5f5'
        text_color = '#212121' if self.theme == 'light' else '#e0e0e0'
        
        self.canvas2.axes.clear()
        bars = self.canvas2.axes.bar(x, energies_no_bias, color=bar_color, 
                                      edgecolor='white', linewidth=1.5, alpha=0.85)
        
        for bar, val in zip(bars, energies_no_bias):
            self.canvas2.axes.text(bar.get_x() + bar.get_width() / 2.0,
                                   bar.get_height() + 0.02,
                                   f'{val:.2f}', ha='center', va='bottom',
                                   color=text_color, fontsize=9)
        
        self.canvas2.axes.set_xticks(x)
        self.canvas2.axes.set_xticklabels(state_labels, color=text_color)
        self.canvas2.axes.set_ylabel('Energy', color=text_color)
        self.canvas2.axes.set_title('Hopfield Energy Landscape (no bias)', color=text_color)
        self.canvas2.axes.tick_params(colors=text_color)
        self.canvas2.axes.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        self.canvas2.axes.grid(True, axis='y', alpha=0.3)
        self.canvas2.draw()
        
        try:
            self.canvas2.fig.savefig('part2_energy_landscape.png', dpi=150, bbox_inches='tight',
                                     facecolor=self.canvas2.fig.get_facecolor())
        except Exception as e:
            print(f"Error saving part2_energy_landscape.png: {e}")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont('Segoe UI', 10))
    window = MainWindow()
    window.show() 
    sys.exit(app.exec())
