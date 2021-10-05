from flask import Flask,render_template,request
import numpy as np 
import joblib
from sklearn.preprocessing import StandardScaler



app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/breast_cancer')
def breast_cancer():
    return render_template('breast_cancer.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/predict_liver')
def predict_liver():
    a1 = int(request.args.get('a'))
    b1 = int(request.args.get('b'))
    c1 = float(request.args.get('c'))
    d1 = float(request.args.get('d'))
    e1 = int(request.args.get('e'))
    f1 = int(request.args.get('f'))
    g1 = int(request.args.get('g'))
    h1 = float(request.args.get('h'))
    i1 = float(request.args.get('i'))
    j1 = float(request.args.get('j')) 
  
    input_data = (a1,b1,c1,d1,e1,f1,g1,h1,i1,j1)
    scaler = StandardScaler()
    input_data_asnparray = np.asarray(input_data)
    reshaped_input_data = input_data_asnparray.reshape(1,-1)
    std_input_data = scaler.fit_transform(reshaped_input_data)
    loaded_model = joblib.load('liver_model.pkl')
    prediction = loaded_model.predict(std_input_data)
    if prediction[0]==0:
        outcome_data = 'Have the disease'
    else:
        outcome_data = 'Does not have the disease'
        
    return render_template('predict_liver.html',result=outcome_data)

@app.route('/predict_diabetes')
def predict_diabetes():
    a1 = int(request.args.get('a'))
    b1 = int(request.args.get('b'))
    c1 = int(request.args.get('c'))
    d1 = int(request.args.get('d'))
    e1 = int(request.args.get('e'))
    f1 = float(request.args.get('f'))
    g1 = float(request.args.get('g'))
    h1 = int(request.args.get('h')) 
  
    input_data = (a1,b1,c1,d1,e1,f1,g1,h1)
    scaler = StandardScaler()
    input_data_asnparray = np.asarray(input_data)
    reshaped_input_data = input_data_asnparray.reshape(1,-1)
    std_input_data = scaler.fit_transform(reshaped_input_data)
    loaded_model = joblib.load('diabetes_model.pkl')
    prediction = loaded_model.predict(std_input_data)
    if prediction[0]==0:
        outcome_data = 'Have the disease'
    else:
        outcome_data = 'Does not have the disease'
    return render_template('predict_diabetes.html',result=outcome_data)

@app.route('/predict_breast_cancer')
def predict_breast_cancer():
    a1 = float(request.args.get('a'))
    b1 = float(request.args.get('b'))
    c1 = float(request.args.get('c'))
    d1 = float(request.args.get('d'))
    e1 = float(request.args.get('e')) 
  
    input_data = (a1,b1,c1,d1,e1)
    scaler = StandardScaler()
    input_data_asnparray = np.asarray(input_data)
    reshaped_input_data = input_data_asnparray.reshape(1,-1)
    std_input_data = scaler.fit_transform(reshaped_input_data)
    loaded_model = joblib.load('breast_cancer_model.pkl')
    prediction = loaded_model.predict(std_input_data)
    if prediction[0]==0:
        outcome_data = 'Have the disease'
    else:
        outcome_data = 'Does not have the disease'
    
    return render_template('predict_breast_cancer.html',result=outcome_data)








if __name__ == '__main__':
    app.run(debug =  True)