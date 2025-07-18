import re
import pycountry

# Create a list of country names
country_names = [country.name for country in pycountry.countries]

# Function to remove special characters and spaces
def clean_name(name):
    return re.sub(r'[^A-Za-z]', '', name)

# Dictionary to store letter frequency for each country
country_letter_frequencies = {}

# Iterate through each country
for country_name in country_names:
    cleaned_name = clean_name(country_name.lower())
    letter_count = {}
    
    # Count the frequency of each letter
    for letter in cleaned_name:
        if letter in letter_count:
            letter_count[letter] += 1
        else:
            letter_count[letter] = 1
    
    # Store the letter frequency dictionary for the country
    country_letter_frequencies[country_name] = letter_count

# Print the final result
print(country_letter_frequencies)