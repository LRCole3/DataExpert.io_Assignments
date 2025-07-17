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

output_file = "Assignment1/output/prompt_v1_results.txt"
with open(output_file, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("TESTING PROMPT WITHOUT OPTIMIZATION\n")
    f.write("=" * 80 + "\n")

# Test gpt-3.5 with dspy predict
with open(output_file, "a") as f:
    f.write("\n1. Testing GPT-3.5 with dspy predict:\n")
lm_35 = dspy.LM("openai/gpt-3.5-turbo", api_key=api_key)
dspy.configure(lm=lm_35)
qa_35_predict = dspy.Predict('question -> answer')
result_35_predict = qa_35_predict(question=question)
with open(output_file, "a") as f:
    f.write(f"Result: {result_35_predict.answer}\n")

# Test gpt-3.5 with dspy chainofthought
with open(output_file, "a") as f:
    f.write("\n2. Testing GPT-3.5 with dspy chainofthought:\n")
qa_35_cot = dspy.ChainOfThought('question -> answer')
result_35_cot = qa_35_cot(question=question)
with open(output_file, "a") as f:
    f.write(f"Reasoning: {result_35_cot.reasoning}\n")
    f.write(f"Result: {result_35_cot.answer}\n")

# Test gpt-4o with dspy predict
with open(output_file, "a") as f:
    f.write("\n3. Testing GPT-4o with dspy predict:\n")
lm_4o = dspy.LM("openai/gpt-4o", api_key=api_key)
dspy.configure(lm=lm_4o)
qa_4o_predict = dspy.Predict('question -> answer')
result_4o_predict = qa_4o_predict(question=question)
with open(output_file, "a") as f:
    f.write(f"Result: {result_4o_predict.answer}\n")

# Test gpt-4o with dspy chainofthought
with open(output_file, "a") as f:
    f.write("\n4. Testing GPT-4o with dspy chainofthought:\n")
qa_4o_cot = dspy.ChainOfThought('question -> answer')
result_4o_cot = qa_4o_cot(question=question)
with open(output_file, "a") as f:
    f.write(f"Reasoning: {result_4o_cot.reasoning}\n")
    f.write(f"Result: {result_4o_cot.answer}\n")

with open(output_file, "a") as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write("CREATING EXAMPLES FOR DSPY OPTIMIZATION\n")
    f.write("=" * 80 + "\n")

# Create examples for dspy to create an optimized prompt
trainset = [
    dspy.Example(question="What country has the same letter repeated the most in its name?", answer="Saint Vincent and the Grenadines").with_inputs("question"),
    dspy.Example(question="Which nation has the highest frequency of repeated letters in its name?", answer="Saint Vincent and the Grenadines").with_inputs("question"),
    dspy.Example(question="What is the country with the most letter repetitions in its official name?", answer="Saint Vincent and the Grenadines").with_inputs("question"),
    dspy.Example(question="Which country's name contains the same letter appearing most frequently?", answer="Saint Vincent and the Grenadines").with_inputs("question"),
    dspy.Example(question="What nation has a name with the highest count of duplicate letters?", answer="Saint Vincent and the Grenadines").with_inputs("question")
]

with open(output_file, "a") as f:
    f.write(f"Created {len(trainset)} training examples\n")

with open(output_file, "a") as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write("CREATING OPTIMIZED PROMPT WITH DSPY\n")
    f.write("=" * 80 + "\n")

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
compiled_classifier.save("Assignment1/prompts/optimized_v1.json")

with open(output_file, "a") as f:
    f.write("Optimized prompt created successfully\n")

with open(output_file, "a") as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write("TESTING OPTIMIZED PROMPT (10 TRIES EACH)\n")
    f.write("=" * 80 + "\n")

def test_accuracy(model_func, model_name, num_tries=10):
    correct = 0
    results = []
    for i in range(num_tries):
        try:
            result = model_func(question=question)
            answer = result.answer if hasattr(result, 'answer') else str(result)
            reasoning = result.reasoning if hasattr(result, 'reasoning') else None
            results.append({'answer': answer, 'reasoning': reasoning})
            if "Saint Vincent and the Grenadines" in answer:
                correct += 1
        except Exception as e:
            with open(output_file, "a") as f:
                f.write(f"Error on try {i+1}: {e}\n")
            results.append({'answer': "ERROR", 'reasoning': None})
    
    accuracy = (correct / num_tries) * 100
    with open(output_file, "a") as f:
        f.write(f"{model_name}: {[r['answer'] for r in results]})\n")
        f.write(f"{model_name}: {correct}/{num_tries} correct ({accuracy:.1f}%)\n")
        # Print reasoning for Chain Of Thought tests
        if "ChainOfThought" in model_name:
            for i, result in enumerate(results):
                if result['reasoning']:
                    f.write(f"  Try {i+1} reasoning: {result['reasoning']}\n")
    return accuracy, results

# Test optimized prompt 10 times - GPT-3.5 with dspy predict
with open(output_file, "a") as f:
    f.write("\n1. Testing optimized prompt - GPT-3.5 with dspy predict:\n")
dspy.configure(lm=lm_35)
compiled_classifier_35 = bootstrap_few_shot.compile(country_classifier, trainset=trainset)
qa_35_opt_predict = dspy.Predict(CountryQuestion)
acc_35_opt_predict, _ = test_accuracy(qa_35_opt_predict, "GPT-3.5 Optimized Predict")

# Test optimized prompt 10 times - GPT-3.5 with dspy chainofthought
with open(output_file, "a") as f:
    f.write("\n2. Testing optimized prompt - GPT-3.5 with dspy chainofthought:\n")
acc_35_opt_cot, _ = test_accuracy(compiled_classifier_35, "GPT-3.5 Optimized ChainOfThought")

# Test optimized prompt 10 times - GPT-4o with dspy predict
with open(output_file, "a") as f:
    f.write("\n3. Testing optimized prompt - GPT-4o with dspy predict:\n")
dspy.configure(lm=lm_4o)
compiled_classifier_4o = bootstrap_few_shot.compile(country_classifier, trainset=trainset)
qa_4o_opt_predict = dspy.Predict(CountryQuestion)
acc_4o_opt_predict, _ = test_accuracy(qa_4o_opt_predict, "GPT-4o Optimized Predict")

# Test optimized prompt 10 times - GPT-4o with dspy chainofthought
with open(output_file, "a") as f:
    f.write("\n4. Testing optimized prompt - GPT-4o with dspy chainofthought:\n")
acc_4o_opt_cot, _ = test_accuracy(compiled_classifier_4o, "GPT-4o Optimized ChainOfThought")

with open(output_file, "a") as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write("SUMMARY OF RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write(f"GPT-3.5 Optimized Predict: {acc_35_opt_predict:.1f}% accuracy\n")
    f.write(f"GPT-3.5 Optimized ChainOfThought: {acc_35_opt_cot:.1f}% accuracy\n")
    f.write(f"GPT-4o Optimized Predict: {acc_4o_opt_predict:.1f}% accuracy\n")
    f.write(f"GPT-4o Optimized ChainOfThought: {acc_4o_opt_cot:.1f}% accuracy\n")

print(f"Results written to {output_file}")