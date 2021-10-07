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
        self.intro_text.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.intro_text.setWordWrap(True)

        self.loss_label = QLabel("0", self)
        self.reward_label = QLabel("0", self)
        self.bddl_goal_state = QLabel("0", self)
        self.horizontalGroupBox = QGroupBox()
        layout = QGridLayout()
        layout.addWidget(self.intro_text, 0, 0, 1, 6)

        layout.addWidget(QPushButton("W = +Y"), 1, 1)
        layout.addWidget(QPushButton("R = Resume"), 1, 4)
        layout.addWidget(QPushButton("P = Pause"), 1, 5)

        layout.addWidget(QPushButton("A = -X"), 2, 0)
        layout.addWidget(QPushButton("S = +Z"), 2, 1)
        layout.addWidget(QPushButton("D = +X"), 2, 2)

        layout.addWidget(QPushButton("Z = -Z"), 3, 0)
        layout.addWidget(QPushButton("X = -Y"), 3, 1)

        layout.addWidget(QPushButton("Loss: "), 4, 0)
        layout.addWidget(self.loss_label, 4, 1)

        layout.addWidget(QPushButton("Reward: "), 4, 2)
        layout.addWidget(self.reward_label, 4, 3)

        layout.addWidget(QPushButton("BDDL Goal State: "), 4, 4)
        layout.addWidget(self.bddl_goal_state, 4, 5)

        self.horizontalGroupBox.setLayout(layout)

    def updateLoss(self, loss_val):
        self.loss_label.setText(str(loss_val))

    def updateReward(self, reward_val):
        self.reward_label.setText(str(reward_val))

    def updateBDDL(self, bddl_val):
        self.bddl_goal_state.setText(str(bddl_val))


def main():
    app = QApplication(sys.argv)
    ex = FeedbackInterface()
    ex.updateLoss(random.randint(0, 1000) * 10000)
    ex.updateReward(random.randint(0, 1000) * 10000)
    ex.updateBDDL(random.randint(0, 1000) * 10000)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
