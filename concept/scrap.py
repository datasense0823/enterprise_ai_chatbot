import requests
from bs4 import BeautifulSoup

# URL of the page you want to scrape
url = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"

# Fetch page content
response = requests.get(url)
html_content = response.content

# Parse HTML
soup = BeautifulSoup(html_content, "html.parser")

# Remove script and style tags (unnecessary for text)
for script in soup(["script", "style"]):
    script.extract()

# Get all visible text
text = soup.get_text(separator=' ', strip=True)  # separator=' ' to ensure spacing

# Clean excessive whitespaces (optional)
import re
clean_text = re.sub(r'\s+', ' ', text)

# Print the cleaned text
print(clean_text)
