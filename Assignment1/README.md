# 📅 Prompt Engineering

## **Prompting** - file you need to submit is prompts.md

- Prompt engineer the following question so it consistently answers the question right the first time.
  - What country has the same letter repeated the most in its name?

- Try different models, which one performs better?
- Try using GPT-4o and prompt engineering to get a more accurate answer


In your prompts.md file include:

- The best performing models you tried
- The prompt engineered version to get GPT-4o to work better
- The actual answer to the question 

### Implementation Files

#### Prompt Versions
- **prompt_v1.py** - Examples include the answer Saint Vincent and the Grenadines
- **prompt_v2.py** - Examples for optimization do not include the answer Saint Vincent and the Grenadines
- **ProgramOfThought_v1.py** - Example using program of thought to create a solution doesn't use any packages
-- **ProgramOfThought_v1.py** - Example using program of thought to create a solution adding the constraint that it must use pycountry. (Note: some prompt optimization still needed getting an incorrect answer but executed code is close to correct)


#### Output Directory
Contains results from running the different prompt versions:
- Text files with model responses and analysis results

#### Prompts Directory
Contains optimized prompt configurations:
- JSON files with the engineered prompts for each version


## Vibe Coding - file you need to submit is main.py

- Follow the exercises in lab 2 where we vibecoded a FastAPI api

- Make sure the API has an /ask endpoint for ChatGPT integration
- Make sure you set up a Milvus account [here](https://milvus.io/) and get RAG setup as well


### Submission

Zip up to the two files 
- prompts.md
- main.py 

And then upload to DataExpert.io


