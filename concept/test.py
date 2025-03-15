import requests

content=requests.get("https://gist.githubusercontent.com/datasense0823/8cb645216814fcdedbcc1b37d94df56b/raw/b787b5955d98e1987b7077874faf701b5fc62bd5/shagunnagpal.json")

print(content.text)