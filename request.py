import requests

url = 'https://makeup-recommender-jas.herokuapp.com/results'
r = requests.post(url,json={'rate':5, 'sales_in_first_month':200, 'sales_in_second_month':400})

print(r.json())
