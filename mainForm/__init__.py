from PyQt5 import QtWidgets,uic
from PyQt5.QtWidgets import *
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import random,math
#from mpl_toolkits.mplot3d import Axes3D
path = os.getcwd()
#from  Main_Window import Ui_MainWindow
dpath = path  + os.sep + "DataSet" + os.sep
mpath = path +os.sep+"ui"+os.sep +"Main_Window.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(mpath)
in_array=[] #input
weight1=np.array([-1.2,1,1]) #鍵結值
weight2=np.array([0.3,1,1])
weight3=np.array([0.5,0.4,0.8])
weight=[weight1,weight2,weight3]
d1=[] #期望值
x=[]

class MainUi(QtWidgets.QMainWindow, Ui_MainWindow):  # Python的多重繼承 MainUi 繼承自兩個類別
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.retranslateUi(self)
        self.title = '感知機類神經網路HW'
        self.iniGuiEvent()
        
        num_layer=int(self.menu2.currentText())
        self.table.setColumnCount(num_layer)
      
     
    # 定義事件觸發時要啟動的函式
    def iniGuiEvent(self):
        self.setWindowTitle(self.title)
        self.load.clicked.connect(self.load_onClick)
        self.clear_layout.clicked.connect(self.clear_layout_onClick)
        self.start.clicked.connect(self.plot3d)
        self.menu2.activated[int].connect(self.set_layer)
        
    # 點擊按鈕，讀取所選的文字檔
    def load_onClick(self):
        global in_array,d1
        p=[]
        f=open(dpath+self.menu1.currentText(),'r') 
        line=f.readline()
        while line:
            p=list(map(float,line.split()))
            p.insert(0,-1.0)
            d1.append(p.pop(3))
            in_array.append(p)
            line=f.readline()
        self.listWidget.insertItem(0,self.menu1.currentText())
        calculate.normalize(d1)
        
   
    #清除layout內的widget及初始化向量in_array,d
    def clear_layout_onClick(self):
        for i in reversed(range(self.ltest.count())): 
            self.ltest.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.test.count())): 
            self.test.itemAt(i).widget().setParent(None)
        #self.listWidget.removeItemWidget(self.listWidget.takeItem(0))
        self.listWidget.clear()
        self.label.setText("")
        self.label_2.setText("")
        global in_array,d1,weight1,weight2,weight3,weight
        in_array = []
        d1= []
        weight1=np.array([-1,0,1])
        weight1=np.array([-1.2,1,1]) #鍵結值
        weight2=np.array([0.3,1,1])
        weight3=np.array([0.5,0.4,0.8])
        weight=[weight1,weight2,weight3]
        
    #在LAYOUT裡畫圖
    def plot3d(self):
        global in_array,d,weight1
        #QMessageBox.about(self, "Title", "Message")
        n=float(self.learn.text())
        condition=int(self.condition.text())
        
        figure = plt.figure()
        #ax = Axes3D(figure)
        canvas = FigureCanvas(figure)
        self.ltest.addWidget(canvas)
        #ax.set_zlabel('Z')  
        plt.ylabel('X2')
        plt.xlabel('X1')
        
        #training
        random_x=in_array.copy()
        random_d=d1.copy()       
        #p,d=calculate.get_rand_array(random_x,random_d,2/3)
        p=np.array(random_x)
        d=np.array(random_d)
      
# =============================================================================
#         for k in range(condition):
#             j=k%len(p)
#             if (calculate.hard_limiter(np.dot(weight1,p[j])) != d[j]):
#                 if (d[j] == 1):
#                     weight1=weight1+n*p[j]
#                 elif (d[j] == 0):
#                     weight1=weight1-n*p[j]
# =============================================================================
        for k in range(condition):
            y=[]
            x=[]
            j=k%len(p)
            for i in range(len(weight)-1):
                y.append(calculate.sigmoid(np.dot(weight[i],p[j])))
            y.insert(0,-1)
            z=(calculate.sigmoid(np.dot(weight[-1],y)))
            a=(d[j]-z)*z*(1-z)
            x.insert(0,a)
            for i in range(len(y)-1):
                tmp=y[i+1]*(1-y[i+1])*x[-1]*weight[-1][i+1]
                x.insert(i,tmp)
            y=np.array(y)    
            weight[0]=weight[0]+n*x[0]*p[j]
            weight[1]=weight[1]+n*x[1]*p[j]
            weight[2]=weight[2]+n*x[2]*y
        
        #correct rate
        self.label.setText(str(calculate.correcrt_rate(p,d,weight1)))
        self.label_2.setText(str(calculate.correcrt_rate(random_x,random_d,weight1)))
        
        #input的位置
        p=np.array(in_array)    
        aa=[]
        bb=[]
        for i in range(len(p)):
            for j in range(len(weight)-1):
                aa.append(calculate.sigmoid(np.dot(weight[j],p[i])))
            aa.insert(0,-1)
            bb.append(calculate.sigmoid(np.dot(weight[-1],aa)))
            aa=[]
            
        for i in range(len(p)):
            if(bb[i] >= 0.5):
                plt.scatter(p[i][1], p[i][2], c='r')
            else:
                plt.scatter(p[i][1], p[i][2], c='g')
               
        
        
        xmin,xmax= plt.xlim()
        print(plt.xlim())
        ymin,ymax= plt.ylim()
        pmin=calculate.w_func(xmin,weight[0][0],weight[0][1],weight[0][2])
        pmax=calculate.w_func(xmax,weight[0][0],weight[0][1],weight[0][2])
        plt.axis([xmin,xmax,ymin,ymax])
        plt.plot([xmin,xmax],[pmin,pmax])
        plt.plot([xmin,xmax],[calculate.w_func(xmin,weight[1][0],weight[1][1],weight[1][2]),calculate.w_func(xmax,weight[1][0],weight[1][1],weight[1][2])])
        #plt.text((xmax+xmin)/2,(ymax+ymin)/2,"%.2fx1+%.2fx2=%.2f"%(weight1[1],weight1[2],weight1[0]))
        #plt.quiver((xmin+xmax)/2,(-weight1[1]/weight1[2])*(xmin+xmax)/2+weight1[0]/weight1[2],weight1[1],weight1[2])
        #顯示理想分類的圖
        figure1 = plt.figure()
        canvas1 = FigureCanvas(figure1)
        self.test.addWidget(canvas1)
        p=np.array(in_array)
        for i in range(len(p)):
            if(d1[i] == 1):
                plt.scatter(p[i][1], p[i][2], c='r')
            elif(d1[i] == 0):
                plt.scatter(p[i][1], p[i][2], c='g')
   #判斷有無先load data
    def is_load(self):
        global k
        if (k==1):
            self.plot3d
        elif (k==0):
            QMessageBox.about(self, "Title", "Message")
            
    def set_layer(self):
        headerlist=[]
        num_layer=int(self.menu2.currentText())
        self.table.setColumnCount(num_layer)
        for i in range(num_layer):
            headerlist.append("隱藏層"+str(i))
        self.table.setHorizontalHeaderLabels(headerlist)
        for i in range(num_layer):
            self.table.setItem(0,i,QTableWidgetItem(str(0)))

class calculate:
    #function
    def sigmoid(i):
        return 1/(1+math.exp(-i))
        
    def hard_limiter(i):
        if i >= 0:
            return 1
        elif i<0:
            return 0
        else :
            return 0
        
    #normalize
    def normalize(a):
        nmax = max(a)
        nmin = min(a)
        avg=(nmax+nmin)/2
        for  i in range(len(a)):
            if a[i] >= avg:
                a[i] = 1
            elif a[i]<avg:
                a[i] = 0

    # get random array
    def get_rand_array(x_array,e_array,num):
        garray=[]
        earray=[]
        for i in range(round(num*len(x_array))):
            r=random.randint(0,len(x_array))-1
            garray.append(x_array.pop(r))
            earray.append(e_array.pop(r))
        return garray,earray 
    
    #correct_rate
    def correcrt_rate(x_array,e_array,weight1):
        count=0
        for i in range(len(x_array)):
            if (calculate.hard_limiter(np.dot(weight1,x_array[i])) ==e_array[i]):
                count+=1
        return count/len(x_array)*100
    
    def w_func(x,s,w1,w2):
        return (s-x*w1)/w2
            
        
    
        