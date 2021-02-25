import streamlit as st
import numpy as np
import pandas as pd
import pickle
from lightfm import LightFM
from lightfm.data import Dataset

st.set_page_config(layout='centered')
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("static/css/style.css")

html = f"<button><img src='https://www.sephora.com/img/ufe/rich-profile/skintone-porcelain.png'></button>"

st.title('Makeup Recommender')

st.header('About the Recommender')
st.write('This is a one-stop shop experience where you can discover an array of products to create an entire makeup look. Forget spending hours browsing through items because our model recommends four items: a face, eye, lip, and cheek product. These products are based on similar products that you enjoy, products that similar users have purchased, as well as products that are personalized to your skin type, skin tone, ingredient preferences, and budget. We utilize collaborative filtering recommendations along with content-based filtering to ensure your satisfaction and success when creating your desired look.')
st.subheader('First, we need to ask you some questions...')
st.write('')
st.write('Which best describes your skin type?')
st.write('Not sure? To find out more about skin types click [here](https://www.masterclass.com/articles/how-to-identify-your-skin-type#what-are-the-most-common-skin-types)')
skin_type = st.radio('', ('Oily', 'Dry', 'Combination', 'Normal'))
st.write('')
st.write('')
st.write('')
st.write('Which best describes your skin tone?')
porcelain, fair, light, medium, tan, olive, deep, dark, ebony = st.beta_columns(9)
with porcelain:
        st.image('./skintone-porcelain.png', caption='Porcelain')
with fair:
        st.image('./skintone-fair.png', caption='Fair')
with light:
        st.image('./skintone-light.png', caption='Light')
with medium:
        st.image('./skintone-medium.png', caption='Medium')
with tan:
        st.image('./skintone-tan.png', caption='Tan')
with olive:
        st.image('./skintone-olive.png', caption='Olive')
with deep:
        st.image('./skintone-deep.png', caption='Deep')
with dark:
        st.image('./skintone-dark.png', caption='Dark')
with ebony:
        st.image('./skintone-ebony.png', caption='Ebony')
skin_tone = st.radio('', ('Porcelain', 'Fair', 'Light', 'Medium', 'Tan', 'Olive', 'Deep', 'Dark', 'Ebony'))
st.write('')
st.write('')
st.write('')
st.write('What is your budget per product?')
budget = st.slider('', 0, 240, 0, 5)
st.write('')
st.write('')
st.write('')
st.write('Do you wish to avoid any of the following ingredients?')
allergies = st.multiselect('Select all those that apply.', ['Rubber', 'Fragrances', 'Preservatives', 'Metals'])
st.write('')
st.write('')
st.write('')
st.write('We would love to know which products you currently enjoy! Knowing what you already love will help us make better recommendations.')
products = st.text_input('Please enter at least one Sephora product ID below (separated by commas).')
st.write('')
st.write('')
clicked = st.button('Find me a set!')

st.sidebar.title('Contact Us :email:')
st.sidebar.text('Shayal Singh, scs008@ucsd.edu')
st.sidebar.text('Alex Kim, aek005@ucsd.edu')
st.sidebar.text('Justin Lee, jul290@ucsd.edu')
st.sidebar.title('Repo :lipstick:')
st.sidebar.write('Find our GitHub repo [here](https://github.com/alexxkim54/AMR_Website)')

if clicked:
    if (len(products) == 0):
        st.error('Please enter at least one Sephora product ID')
    else:
        with st.spinner('About to make you blush...'):
            model = pickle.load(open('model.pkl', 'rb'))
            reviews = pd.read_csv("product_reviews.csv")
            dataset = Dataset()
            dataset.fit(users = (x for x in reviews.userID),
                                items = (x for x in reviews.productID))

            user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
            new_user = 'newUser'

            # change the item_id to actual input
            item_id = products.split(',')
            item_id = [id for id in item_id if id in reviews.productID]

            allergies = [x.lower() for x in allergies]

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
            if len(allergies) == 0:
                recs = new_recs
            else:
                for key, value in new_recs.items():
                    for allergen in allergies:
                        try:
                            allergen_present = products[products['productID'] == float(key)]['ingredients_'+allergen].values[0]
                        except IndexError:
                            continue
                        if allergen_present == 0:
                            recs[key]= value

            new_recs = {}
            for key, value in recs.items():
                # for allergen in allergens:
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


            face_url = "We could not find any face products that match your preferences."
            face_brand = ''
            face_name = ''
            face_price = ''
            eye_url = "We could not find any eye products that match your preferences."
            eye_brand = ''
            eye_name = ''
            eye_price = ''
            cheek_url = "We could not find any cheek products that match your preferences."
            cheek_brand = ''
            cheek_name = ''
            cheek_price = ''
            lips_url = "We could not find any lip products that match your preferences."
            lips_brand = ''
            lips_name = ''
            lips_price = ''

            face_list = list(face_recs.keys())
            if len(face_list) > 0:
                face = face_list[0]
                face_brand = products[products['productID'] == face].brand.values[0]
                face_url = products[products['productID'] == face].URL.values[0]
                face_name = products[products['productID'] == face].name.values[0]
                face_price = products[products['productID'] == face].price.values[0]

            eye_list = list(eye_recs.keys())
            if len(eye_list) > 0:
                eye = eye_list[0]
                eye_brand = products[products['productID'] == eye].brand.values[0]
                eye_url = products[products['productID'] == eye].URL.values[0]
                eye_name = products[products['productID'] == eye].name.values[0]
                eye_price = products[products['productID'] == eye].price.values[0]

            cheek_list = list(cheek_recs.keys())
            if len(cheek_list) > 0:
                cheek = cheek_list[0]
                cheek_brand = products[products['productID'] == cheek].brand.values[0]
                cheek_url = products[products['productID'] == cheek].URL.values[0]
                cheek_name = products[products['productID'] == cheek].name.values[0]
                cheek_price = products[products['productID'] == cheek].price.values[0]

            lips_list = list(lips_recs.keys())
            if len(lips_list) > 0:
                lips = lips_list[0]
                lips_brand = products[products['productID'] == lips].brand.values[0]
                lips_url = products[products['productID'] == lips].URL.values[0]
                lips_name = products[products['productID'] == lips].name.values[0]
                lips_price = products[products['productID'] == lips].price.values[0]

            face_col, eye_col, cheek_col, lips_col = st.beta_columns(4)
            with face_col:
                if len(face_list) == 0:
                    st.write('We could not find any products that match your preferences.')
                else:
                    st.header("Face")
                    st.markdown('**Brand:** ' + face_brand)
                    st.markdown('**Product:** ' + face_name)
                    st.markdown('**Price:** ' + face_price)
                    st.write('[Product Link](face_url)')

            with eye_col:
                if len(eye_list) == 0:
                    st.write('We could not find any products that match your preferences.')
                else:
                    st.header("Eye")
                    st.write('**Brand:** ' + eye_brand)
                    st.write('**Product:** ' + eye_name)
                    st.write('**Price:** ' + eye_price)
                    st.write('[Product Link](eye_url)')

            with cheek_col:
                if len(cheek_list) == 0:
                    st.write('We could not find any products that match your preferences.')
                else:
                    st.header("Cheek")
                    st.write('**Brand:** ' + cheek_brand)
                    st.write('**Product:** ' + cheek_name)
                    st.write('**Price:** ' + cheek_price)
                    st.write('[Product Link](cheek_url)')

            with lips_col:
                if len(lips_list) == 0:
                    st.write('We could not find any products that match your preferences.')
                else:
                    st.header("Lips")
                    st.write('**Brand:** ' + lips_brand)
                    st.write('**Product:** ' + lips_name)
                    st.write('**Price:** ' + lips_price)
                    st.write('[Product Link](lips_url)')
