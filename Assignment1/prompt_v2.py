import dspy
import os
import warnings
from dotenv import load_dotenv
from dspy.teleprompt import BootstrapFewShot

# Suppress the OpenSSL warning
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Use .env and dspy to setup openai apikey
load_dotenv('../.env')
api_key = os.getenv('OPENAI_API_KEY')

# Test question
question = "What country has the same letter repeated the most in its name?"
expected_answer = "Saint Vincent and the Grenadines"

print("=" * 80)
print("TESTING PROMPT WITHOUT OPTIMIZATION")
print("=" * 80)

# Test gpt-3.5 with dspy predict
print("\n1. Testing GPT-3.5 with dspy predict:")
lm_35 = dspy.LM("openai/gpt-3.5-turbo", api_key=api_key)
dspy.configure(lm=lm_35)
qa_35_predict = dspy.Predict('question -> answer')
result_35_predict = qa_35_predict(question=question)
print(f"Result: {result_35_predict.answer}")

# Test gpt-3.5 with dspy chainofthought
print("\n2. Testing GPT-3.5 with dspy chainofthought:")
qa_35_cot = dspy.ChainOfThought('question -> answer')
result_35_cot = qa_35_cot(question=question)
print(f"Reasoning: {result_35_cot.reasoning}")
print(f"Result: {result_35_cot.answer}")

# Test gpt-4o with dspy predict
print("\n3. Testing GPT-4o with dspy predict:")
lm_4o = dspy.LM("openai/gpt-4o", api_key=api_key)
dspy.configure(lm=lm_4o)
qa_4o_predict = dspy.Predict('question -> answer')
result_4o_predict = qa_4o_predict(question=question)
print(f"Result: {result_4o_predict.answer}")

# Test gpt-4o with dspy chainofthought
print("\n4. Testing GPT-4o with dspy chainofthought:")
qa_4o_cot = dspy.ChainOfThought('question -> answer')
result_4o_cot = qa_4o_cot(question=question)
print(f"Reasoning: {result_4o_cot.reasoning}")
print(f"Result: {result_4o_cot.answer}")

print("\n" + "=" * 80)
print("CREATING EXAMPLES FOR DSPY OPTIMIZATION")
print("=" * 80)

# Create examples for dspy to create an optimized prompt (deliberately excluding the correct answer)
trainset = [
    # Asia-Pacific countries with repeated letters
    dspy.Example(question="Using a list of all recognized countries. Compare the countries to see which has the most repeated letters in its name. For example, between  Philippines and the Marshall Islands, which has the most repeated letters?", answer="Philippines", reasoning= "Because the Philippines has p repeated 4 times while Marshall Islands most preeated letter is a 3 times").with_inputs("question"),
    dspy.Example(question="Using the list of all recognized countries. Compare the countries to see which has the most repeated letters in its name. For example, between  Mexico and Canada?", answer="Canada", reasoning= "Because Canada has the most repeated letter is a 3 times while Mexico most repeated letter is a 1 times").with_inputs("question"),
    dspy.Example(question="Using the list of all recognized countries. Compare the countries to see which has the most repeated letters in its name. For example, between Rwanda and Saint Vincent and the Grenadines?", answer="Saint Vincent and the Grenadines", reasoning= "Because Saint Vincent and the Grenadines has the letter n repeated 6 times while Rwanda most repeated letter is a 2 times").with_inputs("question"),
    dspy.Example(question="Using the list of all recognized countries. Compare the countries to see which has the most repeated letters in its name. For example, between the United States Of America and Egypt?", answer="United States of America", reasoning="After counting the repetition of letters in the United States of America you get A three times while Egypts most repeated letter is only repeated once").with_inputs("question"),
    dspy.Example(question="Using the list of all recognized countries. Compare the countries to see which has the most repeated letters in its name. For example, between Marshall Islands and Saint Vincent and the Grenadines?", answer="Saint Vincent and the Grenadines", reasoning= "Because Saint Vincent and the Grenadines has the letter n repeated 6 times while Marshall Islands most repeated letter is a 3 times").with_inputs("question"),
   
]

print(f"Created {len(trainset)} training examples")

print("\n" + "=" * 80)
print("CREATING OPTIMIZED PROMPT WITH DSPY")
print("=" * 80)

# Use dspy and examples to create an optimized prompt
class CountryQuestion(dspy.Signature):
    """Identify the country with the most repeated letters in its name"""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="The country name with the most repeated letters")

class CountryClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CountryQuestion)
    
    def forward(self, question):
        return self.predict(question=question)

# Create the module to be optimized
country_classifier = CountryClassifier()

# Configure the BootstrapFewShot optimizer
bootstrap_few_shot = BootstrapFewShot(metric=None, max_bootstrapped_demos=4, max_labeled_demos=5)


# Use GPT-4o for optimization
dspy.configure(lm=lm_4o)

# Compile the optimized module
compiled_classifier = bootstrap_few_shot.compile(country_classifier, trainset=trainset)
compiled_classifier.save("optimized_v2.json")

print("Optimized prompt created successfully")

print("\n" + "=" * 80)
print("TESTING OPTIMIZED PROMPT (10 TRIES EACH)")
print("=" * 80)

def test_accuracy(model_func, model_name, num_tries=10):
    correct = 0
    results = []
    for i in range(num_tries):
        try:
            result = model_func(question=question)
            answer = result.answer if hasattr(result, 'answer') else str(result)
            results.append(answer)
            if "Saint Vincent and the Grenadines" in answer:
                correct += 1
        except Exception as e:
            print(f"Error on try {i+1}: {e}")
            results.append("ERROR")
    
    accuracy = (correct / num_tries) * 100
    print(f"{model_name}: {results})")
    print(f"{model_name}: {correct}/{num_tries} correct ({accuracy:.1f}%)")
    return accuracy, results

# Test optimized prompt 10 times - GPT-3.5 with dspy predict
print("\n1. Testing optimized prompt - GPT-3.5 with dspy predict:")
dspy.configure(lm=lm_35)
compiled_classifier_35 = bootstrap_few_shot.compile(country_classifier, trainset=trainset)
qa_35_opt_predict = dspy.Predict(CountryQuestion)
acc_35_opt_predict, _ = test_accuracy(qa_35_opt_predict, "GPT-3.5 Optimized Predict")

# Test optimized prompt 10 times - GPT-3.5 with dspy chainofthought
print("\n2. Testing optimized prompt - GPT-3.5 with dspy chainofthought:")
acc_35_opt_cot, _ = test_accuracy(compiled_classifier_35, "GPT-3.5 Optimized ChainOfThought")

# Test optimized prompt 10 times - GPT-4o with dspy predict
print("\n3. Testing optimized prompt - GPT-4o with dspy predict:")
dspy.configure(lm=lm_4o)
compiled_classifier_4o = bootstrap_few_shot.compile(country_classifier, trainset=trainset)
qa_4o_opt_predict = dspy.Predict(CountryQuestion)
acc_4o_opt_predict, _ = test_accuracy(qa_4o_opt_predict, "GPT-4o Optimized Predict")

# Test optimized prompt 10 times - GPT-4o with dspy chainofthought
print("\n4. Testing optimized prompt - GPT-4o with dspy chainofthought:")
acc_4o_opt_cot, _ = test_accuracy(compiled_classifier_4o, "GPT-4o Optimized ChainOfThought")

print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)
print(f"GPT-3.5 Optimized Predict: {acc_35_opt_predict:.1f}% accuracy")
print(f"GPT-3.5 Optimized ChainOfThought: {acc_35_opt_cot:.1f}% accuracy")
print(f"GPT-4o Optimized Predict: {acc_4o_opt_predict:.1f}% accuracy")
print(f"GPT-4o Optimized ChainOfThought: {acc_4o_opt_cot:.1f}% accuracy")