import requests
from bs4 import BeautifulSoup
# <div class="text-[24px] font-medium text-label-1 dark:text-dark-label-1">90</div>
# Set the URL to scrape
url = 'https://leetcode.com/Kool_Cool/'

# Make a GET request to the URL
response = requests.get(url)

# Parse the HTML response using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find an element on the page using a CSS selector
element = soup.findAll("div" , class_ ="text-[24px] font-medium text-label-1 dark:text-dark-label-1")

# Print the text content of the element
print(element[0].text)
