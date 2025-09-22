import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QComboBox
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

class Calculator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Simple Calculator')
        self.setWindowIcon(QIcon('calculator.png'))  # You might need to create an icon file
        self.setGeometry(100, 100, 400, 250)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
            }
            QLabel, QLineEdit, QComboBox, QPushButton {
                font-family: 'Arial';
                font-size: 14px;
            }
            QLineEdit, QComboBox {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            #resultLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Input fields
        input_layout = QHBoxLayout()
        self.num1_input = QLineEdit(self)
        self.num1_input.setPlaceholderText("Enter first number")
        self.num1_input.setAlignment(Qt.AlignCenter)
        self.op_combo = QComboBox(self)
        self.op_combo.addItems(['+', '-', '*', '/'])
        self.num2_input = QLineEdit(self)
        self.num2_input.setPlaceholderText("Enter second number")
        self.num2_input.setAlignment(Qt.AlignCenter)
        input_layout.addWidget(self.num1_input)
        input_layout.addWidget(self.op_combo)
        input_layout.addWidget(self.num2_input)

        # Result display
        self.result_label = QLabel('Result: ', self)
        self.result_label.setObjectName('resultLabel')
        self.result_label.setAlignment(Qt.AlignCenter)

        # Calculate button
        self.calc_button = QPushButton('Calculate', self)
        self.calc_button.clicked.connect(self.calculate)

        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.calc_button)

        self.setLayout(main_layout)

    def calculate(self):
        try:
            num1 = float(self.num1_input.text())
            num2 = float(self.num2_input.text())
            op = self.op_combo.currentText()

            if op == '+':
                result = num1 + num2
            elif op == '-':
                result = num1 - num2
            elif op == '*':
                result = num1 * num2
            elif op == '/':
                if num2 == 0:
                    self.result_label.setText('Result: Cannot divide by zero')
                    return
                result = num1 / num2
            else:
                self.result_label.setText('Result: Invalid operation')
                return

            self.result_label.setText(f'Result: {result}')
        except ValueError:
            self.result_label.setText('Result: Invalid input')
        except Exception as e:
            self.result_label.setText(f'Result: Error - {e}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    calc = Calculator()
    calc.show()
    sys.exit(app.exec_())