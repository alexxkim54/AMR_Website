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
    item_id = request.form['input_products'].split(',')
    item_id = [id for id in item_id if id in reviews.productID]

    budget = float(request.form['myRange'])
    skin_type = request.form['option']
    skin_tone = request.form['option1']
    allergens = request.form.getlist('allergies')
    allergens = [x.lower() for x in allergens]

    data = Dataset()
    data.fit_partial(users=[new_user], items=item_id)
    (new_int, new_weights) = data.build_interactions(list(zip([new_user], item_id)))
    user_id_map2, user_feature_map2, item_id_map2, item_feature_map2 = data.mapping()
    items = list(range(1975))
    predict = model.predict(user_id_map2[new_user], items)

    recs = {list(item_id_map.keys())[i]: predict[i] for i in range(len(predict))} 
    recs = dict(sorted(recs.items(), key=lambda item: item[1], reverse=True))
    products = pd.read_csv('products.csv')

    new_recs = {}
    for key, value in recs.items():
        try:
            price = products[products['productID'] == float(key)].price.values[0]
        except IndexError:
            continue
        if float(price[1:]) <= budget:
            new_recs[key]= value
    
    recs = {}
    for key, value in new_recs.items():
        for allergen in allergens:
            try:
                allergen_present = products[products['productID'] == float(key)]['ingredients_'+allergen].values[0]
            except IndexError:
                continue
            if allergen_present == 0:
                recs[key]= value
                
    new_recs = {}
    for key, value in recs.items():
        for allergen in allergens:
            try:
                temp = reviews[reviews['productID'] == key]
                temp = temp[temp['skin_tone'] == skin_tone]
                temp = temp[temp['skin_type'] == skin_type]
                rating = temp.rating.mean()
            except IndexError:
                continue
            if rating >= 3:
                new_recs[key]= value
                
    recs = new_recs

    face_recs = {}
    eye_recs = {}
    cheek_recs = {}
    lips_recs = {}
    for key, value in recs.items():
        try:
            label = products[products['productID'] == float(key)].Label.values[0]
        except IndexError:
            continue
        if label == 'face':
            face_recs[key] = value
        if label == 'eye':
            eye_recs[key] = value
        if label == 'cheek':
            cheek_recs[key] = value
        if label == 'lips':
            lips_recs[key] = value

    face = list(face_recs.keys())[0]
    eye = list(eye_recs.keys())[0]
    cheek = list(cheek_recs.keys())[0]
    lips = list(lips_recs.keys())[0]

    face_brand = products[products['productID'] == face].brand
    face_url = products[products['productID'] == face].URL
    face_name = products[products['productID'] == face].name
    face_price = products[products['productID'] == face].price

    eye_brand = products[products['productID'] == eye].brand
    eye_url = products[products['productID'] == eye].URL
    eye_name = products[products['productID'] == eye].name
    eye_price = products[products['productID'] == eye].price

    cheek_brand = products[products['productID'] == cheek].brand
    cheek_url = products[products['productID'] == cheek].URL
    cheek_name = products[products['productID'] == cheek].name
    cheek_price = products[products['productID'] == cheek].price

    lips_brand = products[products['productID'] == lips].brand
    lips_url = products[products['productID'] == lips].URL
    lips_name = products[products['productID'] == lips].name
    lips_price = products[products['productID'] == lips].price

    output = face_url + ',' + eye_url + ',' + cheek_url + ',' + lips_url
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
