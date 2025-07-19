import json
import random
import re
from collections import Counter
import pycountry

def clean_name(name):
    return re.sub(r'[^A-Za-z]', '', name)

def get_max_repeated_letters(country_name):
    cleaned_name = clean_name(country_name.lower())
    letter_count = Counter(cleaned_name)
    max_count = max(letter_count.values()) if letter_count else 0
    return max_count, letter_count

def find_countries_with_most_repeated_letters(country_list):
    country_data = {}
    for country in country_list:
        max_count, letter_count = get_max_repeated_letters(country)
        country_data[country] = {
            'max_repeated': max_count,
            'letter_count': letter_count
        }
    
    max_repeated = max(data['max_repeated'] for data in country_data.values())
    winners = [country for country, data in country_data.items() 
               if data['max_repeated'] == max_repeated]
    
    return winners, max_repeated, country_data

def generate_answer(winners, max_repeated, country_data):
    if len(winners) == 1:
        country = winners[0]
        letters = [letter for letter, count in country_data[country]['letter_count'].items() 
                  if count == max_repeated]
        letter_str = "'" + "' and '".join(letters) + "'"
        return f"{country} has the most repeated letters with {max_repeated} occurrences of {letter_str}."
    else:
        winner_details = []
        for country in winners:
            letters = [letter for letter, count in country_data[country]['letter_count'].items() 
                      if count == max_repeated]
            letter_str = "'" + "' and '".join(letters) + "'"
            winner_details.append(f"{country} ({letter_str})")
        
        if len(winners) == 2:
            letter_details = []
            for country in winners:
                for letter, count in country_data[country]['letter_count'].items():
                    if count == max_repeated:
                        letter_details.append(f"'{letter}' in {country}")
            return f"{' and '.join(winners)} are tied with the most repeated letters, each having a letter that appears {max_repeated} times ({', '.join(letter_details)})."
        else:
            return f"{', '.join(winners[:-1])}, and {winners[-1]} are tied with the most repeated letters, each having {max_repeated} repeated letters."

def generate_qa_pairs(num_examples=50):
    country_names = [country.name for country in pycountry.countries]
    examples = []
    
    for _ in range(num_examples):
        selected_countries = random.sample(country_names, 3)
        winners, max_repeated, country_data = find_countries_with_most_repeated_letters(selected_countries)
        
        country_list = '", "'.join(selected_countries)
        question = f'Which country has the most repeated letters among "{country_list}"?'
        answer = generate_answer(winners, max_repeated, country_data)
        
        examples.append({
            "question": f"Question: {question}",
            "answer": f"Answer: {answer}"
        })
    
    return examples

def export_examples():
    examples = generate_qa_pairs()
    
    with open('examples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(examples)} question-answer pairs and exported to examples.json")

if __name__ == "__main__":
    export_examples()