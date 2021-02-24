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
    allergens = [x[:-5].lower() for x in allergens]

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


    face = list(face_recs.keys())
    face_url = "We could not find any face products that match your preferences."
    eye_url = "We could not find any eye products that match your preferences."
    cheek_url = "We could not find any cheek products that match your preferences."
    lips_url = "We could not find any lip products that match your preferences."

    if len(face) > 0:
        face = face[0]
        face_brand = products[products['productID'] == face].brand.values[0]
        face_url = products[products['productID'] == face].URL.values[0]
        face_name = products[products['productID'] == face].name.values[0]
        face_price = products[products['productID'] == face].price.values[0]

    eye = list(eye_recs.keys())
    if len(eye) > 0:
        eye = eye[0]
        eye_brand = products[products['productID'] == eye].brand.values[0]
        eye_url = products[products['productID'] == eye].URL.values[0]
        eye_name = products[products['productID'] == eye].name.values[0]
        eye_price = products[products['productID'] == eye].price.values[0]

    cheek = list(cheek_recs.keys())
    if len(cheek) > 0:
        cheek = cheek[0]
        cheek_brand = products[products['productID'] == cheek].brand.values[0]
        cheek_url = products[products['productID'] == cheek].URL.values[0]
        cheek_name = products[products['productID'] == cheek].name.values[0]
        cheek_price = products[products['productID'] == cheek].price.values[0]

    lips = list(lips_recs.keys())
    if len(lips) > 0:
        lips = lips[0]
        lips_brand = products[products['productID'] == lips].brand.values[0]
        lips_url = products[products['productID'] == lips].URL.values[0]
        lips_name = products[products['productID'] == lips].name.values[0]
        lips_price = products[products['productID'] == lips].price.values[0]

    brand_output = [face_brand, eye_brand, cheek_brand, lips_brand]
    product_output = [face_name, eye_name, cheek_name, lips_name]
    price_output = [face_price, eye_price, cheek_price, lips_price]
    url_output = [face_url, eye_url, cheek_url, lips_url]
    return render_template('index.html', brand_text=brand_output, product_text=product_output, price_text=price_output, link_text=url_output)

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
