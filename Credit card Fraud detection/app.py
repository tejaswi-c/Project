from flask import Flask, request, jsonify, render_template, redirect, session
import os
import pickle
import numpy as np
import pandas as pd
import datetime
import time

from geopy.geocoders import ArcGIS
nom=ArcGIS()
PEOPLE_FOLDER = os.path.join('static', 'image')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
app.secret_key = 'fraud'    
model1=pickle.load(open('model.pkl','rb'))

user = {"username": "abc", "password":"xyz"}
@app.route('/login', methods = ['POST', 'GET'])
def login():
    if(request.method == 'POST'):
        username = request.form.get('username')
        password = request.form.get('password')     
        if username == user['username'] and password == user['password']:
            session['user'] = username
            return redirect('/dashboard')
        return "<h1>Wrong username or password</h1>"    #if the username or password does not matches 

    return render_template("login.html")
@app.route('/dashboard')
def show_index():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'cc3.PNG')
    return render_template("home.html", user_image = full_filename)
@app.route('/dashboard')
def dashboard():
    if('user' in session and session['user'] == user['username']):
        return render_template("home.html")
    #here we are checking whether the user is logged in or not
    return '<h1>You are not logged in.</h1>'  #if the user is not in the session
@app.route('/logout')
def logout():
    session.pop('user')         #session.pop('user') help to remove the session from the browser
    return redirect('/login')
@app.route('/index')
def hello_world():
    return render_template("index.html")


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 10)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
 
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        card = request.form.get("cc_num")
        amt = request.form.get("amt")
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        mloc = to_predict_list[-2]
        cloc = to_predict_list[-1]
        df3 = nom.geocode(mloc, timeout=10)
        df4 = nom.geocode(cloc, timeout=10)
        mlat = df3[1][0]
        mlong = df3[1][1]
        clat = df4[1][0]
        clong = df4[1][1]
        lat_diff = clat - mlat
        long_diff = clong - mlong
        dis = np.sqrt(pow((lat_diff*110),2) + pow((long_diff*110),2))
        to_predict_list = to_predict_list[:-2]
        #to_predict_list.append(dis)
        to_predict_list.insert(9,dis)
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)       
        if int(result)== 1:
            prediction ='Fraud'
            return render_template("resultf.html", card=card,amt=amt)
        else:
            prediction ='Not Fraud'           
            return render_template("resultnf.html", card=card,amt=amt)
# def predict():
 
#     int_features = [x for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model1.predict(final_features)
 
#     output = prediction[0]
 
#     return render_template('index.html', prediction_text='Tree type is {}'.format(output))
# def predict():
    # features = []
    # if request.method == "POST":
    #     cc_num = request.form["cc_num"]
    #     features.append(cc_num)
        #date = request.form["date_time"]
        #hour = pd.DatetimeIndex(date).hour
        #month = pd.DatetimeIndex(date).month
        # features.append(hour)
        # features.append(month)
        # dayofweek = pd.DatetimeIndex(date).dayofweek + 1
        # features.append(dayofweek)
        # df5 = pd.DataFrame()
        # df5["unix_time"] = (time.mktime(date.timetuple()))
        # df5["recency"] = df5.groupby(by="cc_num")["unix_time"].diff()
        # df5.loc[df5.recency.isnull(),["recency"]] = -1
        # df5.recency = df5.recency.apply(lambda x: float((x/60)/60))
        # if df5.loc[(df5["recency"]<1)]:
        #     recency_segment = 4.0
        # elif df5.loc[((df5["recency"]>1) & (df5["recency"]<6))]:
        #     recency_segment = 5.0
        # elif df5.loc[((df5["recency"]>6) & (df5["recency"]<12))]:
        #     recency_segment = 1.0
        # elif df5.loc[((df5["recency"]>12) & (df5["recency"]<24))]:
        #     recency_segment = 2.0
        # elif df5.loc[((df5["recency"]<0))]:
        #     recency_segment = 3.0
        # features.append(recency_segment)
        # category = request.form["category"]
        # features.append(category)
        # amt = request.form["amt"]
        # features.append(amt)
        # gender = request.form["gender"]
        # features.append(gender)
        # age = request.form["age"]
        # features.append(age)
        # hour = request.form["hour"]
        # features.append(hour)
        # month = request.form["month"]
        # features.append(month)
        # dayofweek = request.form["dayofweek"]
        # features.append(dayofweek)
        
        #df = pd.DataFrame()
        # pop = (int(df1["city_pop"].loc[df1['city'] == city]))
        # if pop<10000:
        #     pop_dense = 0.0
        # elif pop>10000 and pop<50000:
        #     pop_dense = 0.1
        # elif pop>50000:
        #     pop_dense = 0.2  
        # pop_dense = request.form["pop_dense"]
        # features.append(pop_dense)
        # displacement = request.form["displacement"]
        # features.append(displacement)
    #     if x == 'mloc':
    #         df3 = nom.geocode(x)
    #         mlat = df3[1][0]
    #         mlong = df3[1][1]
    #     if x == 'cloc':
    #         df4 = nom.geocode(x)
    #         clat = df4[1][0]
    #         clong = df4[1][1]
    #     #lat_diff = abs(clat - mlat)
    #     #long_diff = abs(clong - mlong)
    #     lat_diff = 0.969904	
    #     long_diff = 0.107519	
    #     features.append(np.sqrt(pow((lat_diff*110),2) + pow((long_diff*110),2)))
    #     if x == 'category' or x == 'amt' or x == 'age':
    #         features.append(x)
    # final=[np.array(features)]
    # print(features)
    # print(final)
    # prediction=model1.predict(final)
    # #prediction = model1.predict(features)
    # output = prediction[0]

    # return render_template('index.html', prediction_text='The customer is found to be of class {}'.format(output))
    #int_features=features[:]
    #final=[np.array(int_features)]
    #print(int_features)
    #print(final)
    # prediction=model1.predict(features)
    # output='{0:.{1}f}'.format(prediction[0][1], 2)
    # print(output)
    '''if output == 0:
        return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")'''
if __name__ == "__main__":
    app.run(debug=True)
    