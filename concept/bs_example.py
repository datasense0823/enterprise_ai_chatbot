#Basic Example to Use Beautiful Soup with Requests

import requests
from bs4 import BeautifulSoup

# 1. Fetching the webpage
url = "https://quotes.toscrape.com/"
response = requests.get(url)
html_content = response.content

# 2. Parsing the HTML
soup = BeautifulSoup(html_content, 'html.parser')

# 3. Extracting all quotes
quotes = soup.find_all('span', class_='text')

# 4. Printing each quote
for quote in quotes:
    print(quote.text)
