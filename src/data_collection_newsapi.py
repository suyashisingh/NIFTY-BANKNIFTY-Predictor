import requests

api_key = "26180a45-7384-4ac4-970b-54f9f14b7c77"  # Paste your key here

url = "https://eventregistry.org/api/v1/article/getArticles"
params = {
    "apiKey": api_key,
    "resultType": "articles",
    "keyword": "NIFTY",
    "lang": "eng",
    "articlesSortBy": "date"
}

response = requests.get(url, params=params)
data = response.json()
print(data)
