# 📅 Prompt Engineering Assignment

## Overview
This assignment focuses on prompt engineering to consistently answer the following question:
**"What country has the same letter repeated the most in its name?"**

## Objectives
- Prompt engineer the question to get consistent, accurate answers
- Test different models to determine which performs better
- Use GPT-4o with prompt engineering for improved accuracy

## Submission Requirements
Submit your findings in **prompts.md** including:
- The best performing models you tried
- The prompt engineered version to get GPT-4o to work better
- The actual answer to the question

## Implementation Files

### Prompt Versions
- **prompt_v1.py** - Initial version with examples including Saint Vincent and the Grenadines
- **prompt_v2.py** - Optimized version without the answer in examples
- **ProgramOfThought_v1.py** - Program of thought approach without external packages
- **ProgramOfThought_v2.py** - Program of thought approach using pycountry package (Note: requires prompt optimization for correct results)

### Directory Structure
- **Output/** - Contains results from running different prompt versions (text files with model responses and analysis)
- **Prompts/** - Contains optimized prompt configurations (JSON files with engineered prompts)
- **Configs** - Contains config files to run different secenarios (Work in progress to modularized code to use this)
- **Examples** - Contains JSON files of examples to use with dspy opizations
- **util** - Contains supporting functions for example file to generate Q/A pairs








