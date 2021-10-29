import random
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class FeedbackInterface(QWidget):
    def __init__(self, parent=None):
        super(FeedbackInterface, self).__init__(parent=parent)
        self.title = "Online Learning Feedback GUI"
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        self.setPalette(palette)

        # call the gridlayout function
        self.createGridLayout()
        self.loss_label.text = ""
        self.reward_label.text = ""
        self.bddl_goal_state.text = ""
        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.horizontalGroupBox)
        self.setLayout(windowLayout)
        self.show()  # this sets the main window to the screen size

    def createGridLayout(self):
        self.intro_text = QLabel(
            "Welcome to the feedback system for online learning. Using this system, you can give directional feedback"
            " to the robot. Please use the following keys on your keyboard to guide the robot in the appropriate direction.",
            self,
        )
        self.advice_label = QLabel(
            "Corrective Advice",
            self,
        )
        self.evaluative_feedback = QLabel(
            "Evaluative Feedback",
            self,
        )

        self.intro_text.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.intro_text.setWordWrap(True)

        self.advice_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.advice_label.setWordWrap(True)

        self.evaluative_feedback.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.evaluative_feedback.setWordWrap(True)

        self.loss_label = QLabel("0", self)
        self.reward_label = QLabel("0", self)
        self.bddl_goal_state = QLabel("0", self)
        self.robot_feedback = QLabel("0", self)
        self.horizontalGroupBox = QGroupBox()
        layout = QGridLayout()
        layout.addWidget(self.intro_text, 0, 0, 1, 6)
        layout.addWidget(self.advice_label, 1, 3)

        layout.addWidget(QPushButton("W = +Y"), 2, 1)
        layout.addWidget(QPushButton("R = Reset"), 2, 4)
        layout.addWidget(QPushButton("P = Pause"), 2, 5)

        layout.addWidget(QPushButton("A = -X"), 3, 0)
        layout.addWidget(QPushButton("S = +Z"), 3, 1)
        layout.addWidget(QPushButton("D = +X"), 3, 2)

        layout.addWidget(QPushButton("Z = -Z"), 4, 0)
        layout.addWidget(QPushButton("X = -Y"), 4, 1)

        layout.addWidget(self.evaluative_feedback, 5, 3)

        layout.addWidget(QPushButton("1 = -3"), 6, 0)
        layout.addWidget(QPushButton("2 = -2"), 6, 1)
        layout.addWidget(QPushButton("3 = -1"), 6, 2)
        layout.addWidget(QPushButton("4 = +1"), 6, 3)
        layout.addWidget(QPushButton("5 = +2"), 6, 4)
        layout.addWidget(QPushButton("6 = +3"), 6, 5)

        layout.addWidget(QPushButton("Loss: "), 7, 0)
        layout.addWidget(self.loss_label, 7, 1)

        layout.addWidget(QPushButton("Reward: "), 7, 2)
        layout.addWidget(self.reward_label, 7, 3)

        layout.addWidget(QPushButton("BDDL Goal State: "), 7, 4)
        layout.addWidget(self.bddl_goal_state, 7, 5)

        layout.addWidget(QPushButton("Robot Feedback: "), 8, 2)
        layout.addWidget(self.robot_feedback, 8, 3)

        rand_init_x = QLineEdit()
        rand_init_x.textChanged.connect(self.textchangedX)

        rand_init_y = QLineEdit()
        rand_init_y.textChanged.connect(self.textchangedY)

        rand_init_z = QLineEdit()
        rand_init_z.textChanged.connect(self.textchangedZ)

        layout.addWidget(QPushButton("Random Init X: "), 9, 0)
        layout.addWidget(rand_init_x, 9, 1)
        layout.addWidget(QPushButton("Random Init Y: "), 9, 2)
        layout.addWidget(rand_init_y, 9, 3)
        layout.addWidget(QPushButton("Random Init Z: "), 9, 4)
        layout.addWidget(rand_init_z, 9, 5)

        self.horizontalGroupBox.setLayout(layout)

    def updateLoss(self, loss_val):
        self.loss_label.setText(str(loss_val))

    def updateReward(self, reward_val):
        self.reward_label.setText(str(reward_val))

    def updateBDDL(self, bddl_val):
        self.bddl_goal_state.setText(str(bddl_val))

    def textchangedX(self, text):
        self.X = text

    def textchangedY(self, text):
        self.Y = text

    def textchangedZ(self, text):
        self.Z = text


def main():
    app = QApplication(sys.argv)
    ex = FeedbackInterface()
    ex.updateLoss(random.randint(0, 1000) * 10000)
    ex.updateReward(random.randint(0, 1000) * 10000)
    ex.updateBDDL(random.randint(0, 1000) * 10000)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
