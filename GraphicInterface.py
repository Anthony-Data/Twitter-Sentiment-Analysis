# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:22:00 2021

@author: Antho
"""
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

df=pd.read_csv('NewListeTweets.csv')

X=df['text']
y=df['sentiment']

#sentiment_ordering = ['negative', 'neutral', 'positive']


#Modifier les paramètres

svm = SVC(cache_size = 170, C = 2.75, random_state = 1, kernel='linear', probability=True, break_ties=True)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=75, stratify=y)


# Create the tf-idf vectorizer
vectorizer = TfidfVectorizer()
# First fit the vectorizer with our training set
tfidf_train = vectorizer.fit_transform(X_train)
# Now we can fit our test data with the same vectorizer
tfidf_test = vectorizer.transform(X_test)

svm.fit(tfidf_train,y_train)
y_pred_svm=svm.predict(tfidf_test)


accuracy_svm = accuracy_score(y_test, y_pred_svm)
print('Accuracy SVM :',accuracy_svm)
cf_svm=classification_report(y_test,y_pred_svm)
print(cf_svm)

import numpy as np
from PyQt5.QtGui import QFont, QPixmap
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
import sys

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setStyleSheet("background-color: rgb(29,161,242);")
        self.setGeometry(500, 200, 500, 500)
        self.setWindowTitle('Analyse Sentiment')
        self.initUI()
    
    def initUI(self):
        
        self.label = QLabel(self)
        self.label.setText('Sentence to analyse')
        self.label.move(50,50)
        
        
        self.label2 = QLabel(self)
        self.label2.setText('Sentiment : ')
        self.label2.move(50,100)
        
        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText('Analyse')
        self.b1.move(200,200)
        self.b1.clicked.connect(self.clicked)
        self.b1.setStyleSheet("background-color: gray ; border: 1px solid black;")
        
        self.l1 = QtWidgets.QLineEdit(self)
        self.l1.move(160,50)
        self.l1.setStyleSheet("background-color: white; border: 1px solid black;")
        
        self.l2 = QLabel(self)
        self.l2.move(160,100)
        #self.l2.setStyleSheet("border: 1px solid black;")
        

        self.image = QLabel(self)
        self.pixmap = QPixmap('logo_twitter.png') 
        self.image.setPixmap(self.pixmap) 
        self.image.resize(self.pixmap.width(), self.pixmap.height())
        self.image.move(250,250)
        
    def clicked(self):
        #self.label.setText('u pressed the button')
        
        NewDf = pd.DataFrame(columns = ['text'])
    
        text_list = []
        Xnew = self.l1.text()
        
        NewDf = NewDf.append({'text': Xnew}, ignore_index=True)
        text_list.append(Xnew)
        
        
        Xnew = NewDf['text']
        
        tfidf_new = vectorizer.transform(Xnew)
        y_pred_new = svm.predict(tfidf_new)
        list_sentiment = np.array(y_pred_new).tolist()
        
        sentiment = list_sentiment[0]
        if list_sentiment[0] == 'positive':
            self.l2.setStyleSheet("color : green")
            self.l2.setText(sentiment)
        elif list_sentiment[0] == 'negative':
            self.l2.setStyleSheet("color : red")
            self.l2.setText(sentiment)
        else :
            self.l2.setStyleSheet("color : black")
            self.l2.setText(sentiment)
        
        
        self.l2.setFont(QFont('Arial', 20))
        
        
        dict = {}
        dict["Text"] = text_list
        dict["Sentiments"] = list_sentiment
        
        NewDf = pd.DataFrame.from_dict(dict)
        
        NewDf.to_csv('NewListText.csv')
        #self.update()
        
    #def update(self):
        #self.label.adjustSize()

def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    
    
    win.show()
    sys.exit(app.exec_())
  
    
if __name__ == '__main__':
    window()





