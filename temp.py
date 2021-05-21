import json
import numpy as np
import pickle
from columns_list import X_list
import time

start_time = time.time()

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

pp = predict_price('Abbigere', '1000', '2', '1')
print(round(pp, 2))


print("--- %s seconds ---" % (time.time() - start_time))