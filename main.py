from NaiveBayesTweets_ui import *
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvas
# from matplotlib.figure import Figure
import numpy as np
# import random

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as Navi
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from NaiveBayes import NaiveBayes

# import math
# import os

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.pushButtonEntrenamientoPrevio.clicked.connect(self.browseTrainedFile)
        self.pushButtonTrainFile.clicked.connect(self.browseTrainFile)
        self.pushButtonTesFile.clicked.connect(self.browseTestFile)
        self.pushButtonEntrenar.clicked.connect(self.train)
        self.pushButtonClasificar.clicked.connect(self.classify)

        self.file_name_train = ''
        self.file_name_test = ''
        self.file_name_trained = ''
        self.classifier = NaiveBayes()
        self.classifier.train('/home/lara/Desktop/Proba/NaiveBayes/ML_DATABASE.csv')

        # self.spinBoxNumeroClases.valueChanged.connect(self.numeroClasesValueChange)
        # self.spinBoxNumeroPatrones.valueChanged.connect(self.numeroPatronesValueChange)
        # self.spinBoxDimensionPatron.valueChanged.connect(self.numeroClasesValueChange)
        # self.tabWidget.currentChanged.connect(self.tabChange)
        # self.clasificador = clasificadorMetricas()

    def classify(self):
        tweet = self.plainTextEditTweet.toPlainText()
        r = self.classifier.classify(tweet)
        if self.radioButtonEntrenamientoNuevo.isChecked():
            if r == 1:
                self.labelResultado.setText('Clasificado como positivo con el entrenamiento nuevo')
            else:
                self.labelResultado.setText('Clasificado como negativo con el entrenamiento nuevo')
        else:
            if r == 1:
                self.labelResultado.setText('Clasificado como positivo con el entrenamiento por defecto')
            else:
                self.labelResultado.setText('Clasificado como negativo con el entrenamiento por defecto')
    def train(self):
        acc = self.classifier.train(self.file_name_train)
        self.label_resTrain.setText("Resultado del entrenamiento: " + str(acc*100) + "%")
    def browseTrainedFile(self):
        self.file_name_trained = QFileDialog.getOpenFileName(self, 'Open file', '/home/lara/Desktop/Proba/NaiveBayes')[0]
        self.plainTextEditPrevio.setPlainText(self.file_name_trained)

    def browseTrainFile(self):
        self.file_name_train = QFileDialog.getOpenFileName(self, 'Open file', '/home/lara/Desktop/Proba/NaiveBayes')[0]
        self.plainTextEditTrain.setPlainText(self.file_name_train)

    def browseTestFile(self):
        self.file_name_test = QFileDialog.getOpenFileName(self, 'Open file', '/home/lara/Desktop/Proba/NaiveBayes')[0]
        self.plainTextEditTest.setPlainText(self.file_name_test)



if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
