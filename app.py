import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from lightfm import LightFM
from lightfm.data import Dataset

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    reviews = pd.read_csv("product_reviews.csv")
    dataset = Dataset()
    dataset.fit(users = (x for x in reviews.userID),
                        items = (x for x in reviews.productID))

    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
    new_user = 'newUser'

    # change the item_id to actual input
    item_id = 2060358


    data = Dataset()
    data.fit_partial(users=[new_user], items=[item_id])
    (new_int, new_weights) = data.build_interactions(list(zip([new_user], [item_id])))
    # model.fit_partial(interactions=new_int)
    user_id_map2, user_feature_map2, item_id_map2, item_feature_map2 = data.mapping()
    items = list(range(1975))
    predict = model.predict(user_id_map2[new_user], items)

    recs = {list(item_id_map.keys())[i]: predict[i] for i in range(len(predict))} 
    recs = dict(sorted(recs.items(), key=lambda item: item[1], reverse=True))
    output = list(recs.keys())[:4]
    face_prediction = output[0]
    eye_prediction = output[1]
#    prediction_text = "The top 4 recommended products are: {}".format(output)
    return render_template('index.html', prediction_text=output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    print(data)

    new_user = 'newUser'

    # change the item_id to actual input
    item_id = 2060358   

    data = Dataset()
    data.fit_partial(users=[new_user], items=[item_id])
    (new_int, new_weights) = data.build_interactions(list(zip([new_user], [item_id])))
    user_id_map, user_feature_map, item_id_map, item_feature_map = data.mapping()
    items = list(range(1975))
    predict = model.predict(user_id_map[new_user], items)

    

    recs = {list(item_id_map.keys())[i]: predict[i] for i in range(len(predict))} 
    recs = dict(sorted(recs.items(), key=lambda item: item[1], reverse=True))
    output = list(recs.keys())[:4]

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
