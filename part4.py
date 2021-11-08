import sys
import os
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from scipy.io import wavfile
import numpy as np
import simpleaudio as sa

#Code for band pass filter, implemented using the canonical form
#coefficients below were obtained using sympy and biliniear transform
def bpfilter(inp, samplerate, f_cl, f_ch):
    T = 1/samplerate
    f_cl = np.pi*2*f_cl
    f_ch = np.pi*2*f_ch
    a_0 = T**6*f_ch**3*f_cl**3 + 6*T**5*f_ch**3*f_cl**2 + 6*T**5*f_ch**2*f_cl**3 + 12*T**4*f_ch**3*f_cl + 36*T**4*f_ch**2*f_cl**2 + 12*T**4*f_ch*f_cl**3 + 8*T**3*f_ch**3 + 72*T**3*f_ch**2*f_cl + 72*T**3*f_ch*f_cl**2 + 8*T**3*f_cl**3 + 48*T**2*f_ch**2 + 144*T**2*f_ch*f_cl + 48*T**2*f_cl**2 + 96*T*f_ch + 96*T*f_cl + 64
    a_1 = 6*T**6*f_ch**3*f_cl**3 + 24*T**5*f_ch**3*f_cl**2 + 24*T**5*f_ch**2*f_cl**3 + 24*T**4*f_ch**3*f_cl + 72*T**4*f_ch**2*f_cl**2 + 24*T**4*f_ch*f_cl**3 - 96*T**2*f_ch**2 - 288*T**2*f_ch*f_cl - 96*T**2*f_cl**2 - 384*T*f_ch - 384*T*f_cl - 384
    a_2 = 15*T**6*f_ch**3*f_cl**3 + 30*T**5*f_ch**3*f_cl**2 + 30*T**5*f_ch**2*f_cl**3 - 12*T**4*f_ch**3*f_cl - 36*T**4*f_ch**2*f_cl**2 - 12*T**4*f_ch*f_cl**3 - 24*T**3*f_ch**3 - 216*T**3*f_ch**2*f_cl - 216*T**3*f_ch*f_cl**2 - 24*T**3*f_cl**3 - 48*T**2*f_ch**2 - 144*T**2*f_ch*f_cl - 48*T**2*f_cl**2 + 480*T*f_ch + 480*T*f_cl + 960
    a_3 = 20*T**6*f_ch**3*f_cl**3 - 48*T**4*f_ch**3*f_cl - 144*T**4*f_ch**2*f_cl**2 - 48*T**4*f_ch*f_cl**3 + 192*T**2*f_ch**2 + 576*T**2*f_ch*f_cl + 192*T**2*f_cl**2 - 1280
    a_4 = 15*T**6*f_ch**3*f_cl**3 - 30*T**5*f_ch**3*f_cl**2 - 30*T**5*f_ch**2*f_cl**3 - 12*T**4*f_ch**3*f_cl - 36*T**4*f_ch**2*f_cl**2 - 12*T**4*f_ch*f_cl**3 + 24*T**3*f_ch**3 + 216*T**3*f_ch**2*f_cl + 216*T**3*f_ch*f_cl**2 + 24*T**3*f_cl**3 - 48*T**2*f_ch**2 - 144*T**2*f_ch*f_cl - 48*T**2*f_cl**2 - 480*T*f_ch - 480*T*f_cl + 960
    a_5 = 6*T**6*f_ch**3*f_cl**3 - 24*T**5*f_ch**3*f_cl**2 - 24*T**5*f_ch**2*f_cl**3 + 24*T**4*f_ch**3*f_cl + 72*T**4*f_ch**2*f_cl**2 + 24*T**4*f_ch*f_cl**3 - 96*T**2*f_ch**2 - 288*T**2*f_ch*f_cl - 96*T**2*f_cl**2 + 384*T*f_ch + 384*T*f_cl - 384
    a_6 = 64 + T**6*f_ch**3*f_cl**3 - 6*T**5*f_ch**3*f_cl**2 - 6*T**5*f_ch**2*f_cl**3 + 12*T**4*f_ch**3*f_cl + 36*T**4*f_ch**2*f_cl**2 + 12*T**4*f_ch*f_cl**3 - 8*T**3*f_ch**3 - 72*T**3*f_ch**2*f_cl - 72*T**3*f_ch*f_cl**2 - 8*T**3*f_cl**3 + 48*T**2*f_ch**2 + 144*T**2*f_ch*f_cl + 48*T**2*f_cl**2 - 96*T*f_ch - 96*T*f_cl
    b_0 = 8*T**3*f_ch**3
    b_1 = 0
    b_2 = -24*T**3*f_ch**3
    b_3 = 0
    b_4 = 24*T**3*f_ch**3
    b_5 = 0
    b_6 = -8*T**3*f_ch**3
    x = [0, 0, 0, 0, 0, 0, 0]
    y = []
    for signals in inp:
        x[0] = (signals - x[1]*a_1/a_0 - x[2]*a_2/a_0 - x[3]*a_3/a_0 - x[4]*a_4/a_0 - x[5]*a_5/a_0 - x[6]*a_6/a_0)
        y.append(x[0]*b_0/a_0 + x[1]*b_1/a_0 + x[2]*b_2/a_0 + x[3]*b_3/a_0 + x[4]*b_4/a_0 + x[5]*b_5/a_0 + x[6]*b_6/a_0)
        x[6] = x[5]
        x[5] = x[4]
        x[4] = x[3]
        x[3] = x[2]
        x[2] = x[1]
        x[1] = x[0]
    return y

ui_path = os.path.dirname(os.path.abspath(__file__))
class MainWindow(QMainWindow):
    vawfile_fullpath = ""
    filtered = False
    filtered_data = []
    samplerate = 0
    plays = None
    data = None
    BSP = None
    def __init__(self):
        super(MainWindow,self).__init__()
        loadUi(ui_path+"\\gui.ui",self)
        self.browse.clicked.connect(self.browsefiles)
        self.save_button.clicked.connect(self.run)
        self.play_button.clicked.connect(self.play)
        self.filter_button.clicked.connect(self.filter)

    
    def notfiltered():
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText("Please Filter before you play or save")
        x = msg.exec_()

    def browsefiles(self):
        fname=QFileDialog.getOpenFileName(self,'Open file', ui_path, 'WAV Files (*.wav)')
        self.gui_filename.setText(fname[0])
        if fname[0]!=MainWindow.vawfile_fullpath:
            MainWindow.filtered = False
            MainWindow.vawfile_fullpath = fname[0]

    def play(self):
        if not MainWindow.filtered:
            MainWindow.notfiltered()
            return
        if MainWindow.plays is None:
            self.play_button.setText("Stop")
            if self.bsp_box.isChecked():
                MainWindow.BSP = MainWindow.data - np.array(MainWindow.filtered_data)
                MainWindow.plays = sa.play_buffer(MainWindow.BSP.astype(np.int16),1,2,MainWindow.samplerate)
            else:
                MainWindow.plays = sa.play_buffer(np.array(MainWindow.filtered_data).astype(np.int16), 1, 2, MainWindow.samplerate)
        else:
            if not MainWindow.plays.is_playing():
                self.play_button.setText("Stop")
                if self.bsp_box.isChecked():
                    MainWindow.BSP = MainWindow.data - np.array(MainWindow.filtered_data)
                    MainWindow.plays = sa.play_buffer(MainWindow.BSP.astype(np.int16),1,2,MainWindow.samplerate)
                else:
                    MainWindow.plays = sa.play_buffer(np.array(MainWindow.filtered_data).astype(np.int16), 1, 2, MainWindow.samplerate)
            elif MainWindow.plays.is_playing():
                self.play_button.setText("Play")
                sa.stop_all()


    def filter(self):
        if MainWindow.vawfile_fullpath == "":        
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("Please choose a file before you filter")
            x = msg.exec_()
            return
        samplerate, data = wavfile.read(self.gui_filename.text())
        length = data.shape[0] / samplerate
        f_cl = self.freq_box.value()-self.freq_box.value()*self.percentage_box.value()/100
        f_ch = self.freq_box.value()+self.freq_box.value()*self.percentage_box.value()/100
        MainWindow.filtered_data = bpfilter(data,samplerate,f_cl,f_ch)
        MainWindow.filtered = True
        MainWindow.samplerate = samplerate
        MainWindow.data = data
        print(MainWindow.data)
        
    def run(self):
        if not MainWindow.filtered:
            MainWindow.notfiltered()
            return
        fname = self.gui_filename.text()[:-4]+"B.wav"
        print(fname)
        if self.bsp_box.isChecked():
            BSP = MainWindow.data - np.array(MainWindow.filtered_data)
            wavfile.write(os.path.join(os.getcwd(), fname), MainWindow.samplerate, BSP.astype(np.int16))
        else:
            wavfile.write(os.path.join(os.getcwd(), fname), MainWindow.samplerate, np.array(MainWindow.filtered_data).astype(np.int16))

app=QApplication(sys.argv)
mainwindow=MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedHeight(420)
widget.setFixedWidth(410)
widget.show()
sys.exit(app.exec_())