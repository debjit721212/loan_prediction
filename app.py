from flask import Flask, app, request, render_template
import pickle
import pandas as pd
#import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score  
#import warnings
#warnings.simplefilter("ignore")

app = Flask(__name__)

df1 = pd.read_csv("loan_prediction.csv")
df1 = pd.DataFrame(df1.drop(["Unnamed: 0","Risk_Flag","skfold"],axis=1))

#q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])

def predict():
    
    '''
    Income
    Age
    Experience,
    House_Ownership,
    Car_Ownership,
    Profession,
    CITY,
    STATE,
    CURRENT_JOB_YRS,
    CURRENT_HOUSE_YRS,
    Status
  
    '''
    

    
    inputQuery1 =int(request.form['query1'])
    inputQuery2 = int(request.form['query2'])
    inputQuery3 = int(request.form['query3'])
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = int(request.form['query9'])
    inputQuery10 =int( request.form['query10'])
    inputQuery11 = request.form['query11']
    
    
    
    model = pickle.load(open("model.pkl", "rb"))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10,inputQuery11]]
    
    new_df = pd.DataFrame(data, columns = ['Income' , 'Age', 'Experience','House_Ownership','Car_Ownership',
                                           'Profession','CITY','STATE','CURRENT_JOB_YRS','CURRENT_HOUSE_YRS','Status'])
    
    
    df2 = pd.concat([df1, new_df], ignore_index = True)
    
    new_df__dummies = pd.get_dummies(df2[['House_Ownership', 'Car_Ownership', 'Profession',
                                          'CITY','STATE','Status']], prefix="ohe", prefix_sep="_")
    
    df3 =  pd.DataFrame( df2.drop(columns= ['House_Ownership', 'Car_Ownership', 'Profession',
                                            'CITY', 'STATE','Status']))
    
    final_df = pd.concat([new_df__dummies, df3], axis=1)
    scaler = preprocessing.MinMaxScaler()
    final_df= scaler.fit_transform(final_df)
    final_df = pd.DataFrame(final_df)
    
    single = model.predict(final_df.tail(1))
    probablity = model.predict_proba(final_df.tail(1))[:,1]
    
    if single==1:
        o1 = "Congratulation !, You are eligible for bank loan."
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = " Sorry !, You are not elegible for bank loan. "
        o2 = "Confidence: {}".format((1 - probablity)*100)
        
    return render_template('home.html', output1=o1, output2=o2, 
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'], 
                           query7 = request.form['query7'], 
                           query8 = request.form['query8'], 
                           query9 = request.form['query9'], 
                           query10 = request.form['query10'], 
                           query11 = request.form['query11'])
    
if __name__ == "__main__":
    app.run(threaded=True, port=5000)
