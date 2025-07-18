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

def test_country_repeated_letters():
    """Test ProgramOfThought to find the country with the most repeated letters"""
    print("=== Testing Country with Most Repeated Letters ===")
    
    interpreter = PythonInterpreter(
        enable_network_access=["pypi.org", "cdn.jsdelivr.net", "files.pythonhosted.org"],
        sync_files=True
    )
    
    program = dspy.ProgramOfThought(
        signature="question -> answer",
        max_iters=10,
        interpreter=interpreter
    )
    
    question = "What country has the same letter repeated the most in its name?"
    
    try:
        result = program(question=question)
        dspy.inspect_history(n=5)
        print(f"Result: {result.answer}")
        return result.answer
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_country_repeated_letters()