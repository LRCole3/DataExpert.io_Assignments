import os
import warnings
from dotenv import load_dotenv
import dspy
from dspy.primitives.python_interpreter import PythonInterpreter

# Suppress the OpenSSL warning
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Use .env and dspy to setup openai apikey
load_dotenv('.env')
api_key = os.getenv('OPENAI_API_KEY')

lm_35 = dspy.LM("openai/gpt-3.5-turbo", api_key=api_key)
dspy.configure(lm=lm_35)

class CountryAnalysisSignature(dspy.Signature):
    instruction: str = dspy.InputField(desc="If you need to install packages, use micropip.install at the beginning of your code")
    question: str = dspy.InputField(desc="A problem to solve")
    answer: str = dspy.OutputField(desc="The solution to the problem")

def test_country_repeated_letters():
    """Test ProgramOfThought to find the country with the most repeated letters"""
    print("=== Testing Country with Most Repeated Letters ===")
    
    interpreter = PythonInterpreter(
        enable_network_access=["pypi.org", "cdn.jsdelivr.net", "files.pythonhosted.org"],
        sync_files=True
    )
    
    program = dspy.ProgramOfThought(
        signature=CountryAnalysisSignature,
        max_iters=15,
        interpreter=interpreter
    )
    
    question = """What country has the same letter repeated the most in its name? Use the pycountry library to get a comprehensive list of countries. To solve this, I need to:
    1. Create a list of country names
    2. make sure to remove all special characters and spaces when counting
    3. For each country, count how many times each letter appears
    4. Find the maximum number of times any letter is repeated in each country
    5. Find and return which country has the highest maximum repetition count
    """
    
    try:
        result = program(
            instruction="If you need to install packages, use micropip.install at the beginning of your code",
            question=question
        )
        dspy.inspect_history(n=5)
        print(f"Result: {result.answer}")
        return result.answer
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_country_repeated_letters()