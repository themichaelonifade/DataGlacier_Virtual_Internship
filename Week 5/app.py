from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    fixedacidity = request.form.get('fixedacidity'),
    volatileacidity = request.form.get('volatileacidity'),
    citricacid = request.form.get('citricacid'),
    residualsugar =request.form.get('residualsugar'),
    chlorides = request.form.get('chlorides'),
    freesulfurdioxide = request.form.get('freesulfurdioxide'),
    totalsulfurdioxide = request.form.get('totalsulfurdioxide'),
    density = request.form.get('density'),
    ph = request.form.get('ph'),
    sulphates = request.form.get('sulphates'),
    alcohol = request.form.get('alcohol')

    fixedacidity = float(fixedacidity[0]) if fixedacidity is not None and len(fixedacidity) > 0 else 0
    volatileacidity = float(volatileacidity[0]) if volatileacidity is not None and len(volatileacidity) else 0
    citricacid = float(citricacid[0]) if citricacid is not None and len(citricacid) else 0
    residualsugar = float(residualsugar[0]) if residualsugar is not None and len(residualsugar) else 0
    chlorides = float(chlorides[0]) if chlorides is not None and len(chlorides) else 0
    freesulfurdioxide = int(freesulfurdioxide[0]) if freesulfurdioxide is not None and len(freesulfurdioxide) else 0
    totalsulfurdioxide = int(totalsulfurdioxide[0]) if totalsulfurdioxide is not None and len(totalsulfurdioxide) else 0
    density = float(density[0]) if density is not None and len(density) else 0
    ph = float(ph[0]) if ph is not None and len(ph) else 0
    sulphates = float(sulphates[0]) if sulphates is not None and len(sulphates) else 0
    alcohol = float(alcohol[0]) if alcohol is not None and len(alcohol) else 0

    features_arr = np.array([[fixedacidity, volatileacidity,citricacid, residualsugar,chlorides, freesulfurdioxide, totalsulfurdioxide, density, ph, sulphates, alcohol]])
    prediction = model.predict(features_arr)
    output = round(prediction[0], 2)

    return render_template('result.html', prediction_text=f'The predicted wine quality is: {output}')

if __name__ == '__main__':
    app.debug=True
    app.run()