from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pickle
from flask_cors import CORS
from columns_list import X_list
app = Flask(__name__)
CORS(app)

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

# @app.route('/house', methods=['GET'])
# def house():
#     return render_template('house.html')

@app.route("/api", methods=['GET'])
def api():
    if 'location' and 'area' and 'rooms' and 'bathrooms' in request.args:
        location = request.args['location']
        area = request.args['area']
        rooms = request.args['rooms']
        bathrooms = request.args['bathrooms']
        price = round(predict_price(location, area, rooms, bathrooms), 2)

        if price <= 0:
            result = {
                'location': location,
                'area': area,
                'rooms': rooms,
                'bathrooms': bathrooms,
                'price': 'Cannot Predict Price with this values, Try again with other values',
                'status': 'warning'
            }
            return jsonify(result)
        else:
            result = {
                'location': location,
                'area': area,
                'rooms': rooms,
                'bathrooms': bathrooms,
                'price': price,
                'status': 'success'
            }
            return jsonify(result)
    else:
        result = {
            'status': 'failed'
        }
        return jsonify(result)

def predict_price(location, area, rooms, bathrooms):
    load_model = pickle.load(open('banglore_home_prices_model.pickle', 'rb'))
    loc_index = X_list.index(location)
    x = np.zeros(len(X_list))
    x[0] = area
    x[1] = bathrooms
    x[2] = rooms
    if loc_index >= 0:
        x[loc_index] = 1

    return load_model.predict([x])[0]


if __name__ == "__main__":
    app.run(debug=True)
