# -*- coding: utf-8 -*-
"""LangGraph - Build_a_Reflective_Self_Correcting_Code_Generation_AI_Agent.ipynb

# Reflection Agentic AI Systems


---

"""# Build a Reflective Self-Correcting Code Generation AI Agent

In this project we will be building an intelligent self-correction code generation AI Agent which can generate working code for user problems, leverage the reflection pattern to examine and test the code and improve and correct it.

![](https://i.imgur.com/koLDCZL.png)

### Reflective Self-Correcting Code Generation AI Agent

This project focuses on building a **Reflective Self-Correcting Code Generation AI Agent**, designed to iteratively generate, execute, and refine code to achieve accurate solutions. The workflow integrates reflective reasoning and error analysis to ensure robust and functional code generation. The workflow includes the following components:

1. **Code Generation**:
   - The system utilizes **OpenAI GPT-4o** to generate code based on the user's input and problem requirements.
   - Code is generated following a predefined schema, including:
     - **Prefix**: Problem description or setup requirements.
     - **Imports**: Necessary libraries or dependencies.
     - **Code**: Functional implementation of the solution.

2. **Code Execution and Reflection**:
   - The generated code is executed, and the results are analyzed.
   - If an error occurs, feedback is provided to refine the code:
     - **Error Feedback**: Captures errors or unexpected behaviors during execution.
     - **Reflection**: GPT-4o reflects on the error feedback to iteratively improve the solution.

3. **Iterative Correction Loop**:
   - The system repeats the generate-execute-reflect cycle until:
     - A successful solution is achieved (no errors).
     - The attempt count exceeds a predefined threshold (`N`), ensuring the loop terminates if a solution cannot be reached.

4. **Attempt Counter**:
   - Tracks the number of attempts (`K`) and increments with each iteration.
   - If the attempt count exceeds the predefined threshold, the process halts, and the user is informed.

5. **Final Validation**:
   - Once the code passes execution without errors, the refined solution is presented to the user as the final output.

6. **User Feedback Loop**:
   - If the user is unsatisfied with the solution, the process can be restarted to refine the code further or adjust the requirements.

## Install OpenAI, LangGraph and LangChain dependencies
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install langchain #==0.3.14
# !pip install langchain-openai #==0.3.0
# !pip install langchain-community #==0.3.14
# !pip install langgraph #==0.2.64

"""## Enter Open AI API Key"""

# from getpass import getpass

# OPENAI_KEY = getpass('Enter Open AI API Key: ')

"""## Setup Environment Variables"""

# import os

# os.environ['OPENAI_API_KEY'] = OPENAI_KEY

import os
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

"""## Build Code Generator"""

from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel, Field


# Initialize GPT-5-mini
llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0)

# Prompt
CODE_GEN_SYS_PROMPT = [
        (
            "system",
            """You are a coding assistant.
                Ensure any code you provide can be executed with all required imports and variables defined.
                Make sure point 3 below has some code to run and execute any code or functions which you define

                Structure your answer as follows:
                  1) a prefix describing the code solution
                  2) the imports (if no imports needed keep it empty string)
                  3) the functioning code blocks

                Here is the user question:""",
        )
]

# Data model
class Code(BaseModel):
    """Schema for code solutions to questions about coding."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Just the import statements of the code")
    code: str = Field(description="Code blocks not including import statements")

# LLM
code_generator = llm.with_structured_output(Code)

type(code_generator)

question = "code to find the factorial of a given number"
messages = [("user", question)]

result = code_generator.invoke(CODE_GEN_SYS_PROMPT + messages)
result

print(result.code)

print(result.imports)

def factorial(n):
    """
    Compute factorial of a non-negative integer n iteratively.
    Raises ValueError for negative or non-integer inputs.
    """
    if not isinstance(n, int):
        raise ValueError("factorial is only defined for integers")
    if n < 0:
        raise ValueError("factorial is not defined for negative integers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Example usage: compute factorials for sample numbers
sample_numbers = [0, 1, 5, 10]
for num in sample_numbers:
    print(f"{num}! = {factorial(num)}")

import math
# Compare with math.factorial to verify correctness
for num in sample_numbers:
    assert factorial(num) == math.factorial(num)

"""We create a chain that returns structured data for code solutions. We’ll feed it the user’s question plus the system directive in the prompt above.

## Build Agent Graph Node Functions

The key focus here will be to create the function implementations of the main nodes in our graph which will include:

-  **Generate** a code solution
-  **Check code** imports and code execution and add error messages if any
- **Conditional routing** to regenerate code by reflecting on the errors if any OR stop generation

### Create Agent State Schema
"""

from typing import TypedDict, Annotated, List
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import HumanMessage

class CodeGenState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error_flag : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        code_solution : Code solution
        attempts : Number of tries
    """
    error_flag: str
    messages: Annotated[List[AnyMessage], add_messages]
    code_solution: str
    attempts: int

"""### Node 1: Generate Code"""

def generate_code(state: CodeGenState) -> CodeGenState:
    """Generate code solution from GPT-4o, structured as prefix/imports/code."""
    print("--- GENERATING CODE SOLUTION ---")
    msgs = state["messages"]
    attempts_so_far = state["attempts"]

    # Call code_generation_chain
    code_soln = code_generator.invoke(CODE_GEN_SYS_PROMPT + msgs)

    # We'll record the chain's answer as a new assistant message in conversation.
    new_msg_content = (f"Here is my solution attempt:\n\nDescription: {code_soln.prefix}\n\n"
                       f"Imports: {code_soln.imports}\n\n"
                       f"Code:\n{code_soln.code}")

    msgs.append(("assistant", new_msg_content))
    attempts_so_far += 1

    return {
        "messages": msgs,
        "code_solution": code_soln,
        "attempts": attempts_so_far
    }

"""### Node 2: Check Code

We try to `exec` the imports, then `exec` the code. If errors occur, we pass them back to the conversation. Otherwise, success.
"""

# sample code of how exec works
user_code = """
x = 5
y = 15
print('Result:', x * y)
"""
exec(user_code)

def check_code_execution(state: CodeGenState) -> CodeGenState:
    print("--- CHECKING CODE EXECUTION ---")
    msgs = state["messages"]
    code_soln = state["code_solution"]
    imports_str = code_soln.imports
    code_str = code_soln.code
    attempts = state["attempts"]

    # Attempt to import:
    try:
        exec(imports_str)
    except Exception as e:
        # Import failed
        print("---CODE IMPORT CHECK: FAILED---")
        error_msg = f"""Import test failed!
                        Here is the exception trace details:
                        {e}.

                        Please fix the import section."""

        msgs.append(("user", error_msg))
        return {
            "code_solution": code_soln,
            "attempts": attempts,
            "messages": msgs,
            "error_flag": "yes"
        }

    # Attempt to run code:
    try:
        scope = {}
        exec(f"{imports_str}\n{code_str}", scope)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_msg =  f"""Your code solution failed the code execution test!
                            Here is the exception trace details:
                            {e}

                            Reflect on this error and your prior attempt to solve the problem.

                            (1) State what you think went wrong with the prior solution
                            (2) try to solve this problem again.

                            Return the FULL SOLUTION.

                            Use the code tool to structure the output with a prefix, imports, and code block."""

        msgs.append(("user", error_msg))
        return {
            "code_solution": code_soln,
            "attempts": attempts,
            "messages": msgs,
            "error_flag": "yes"
        }

    # If no errors:
    print("--- NO ERRORS FOUND ---")
    return {
            "code_solution": code_soln,
            "attempts": attempts,
            "messages": msgs,
            "error_flag": "no"
    }

"""### Conditional Routing to Decide Next Step"""

MAX_ATTEMPTS = 3

def decide_next(state: CodeGenState) -> str:
    """If error or attempts < MAX_ATTEMPTS => go generate. Else end."""
    err = state["error_flag"]
    attempts = state["attempts"]
    if err == "no" or attempts >= MAX_ATTEMPTS:
        print("--- DECISION: FINISH ---")
        return "__end__"
    else:
        print("--- DECISION: RETRY ---")
        return "generate_code"

"""## Build the Reflection Agentic Graph

We'll define the nodes and edges in LangGraph, then run it.
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

graph = StateGraph(CodeGenState)

# Add nodes:
graph.add_node("generate_code", generate_code)
graph.add_node("check_code", check_code_execution)

# Edges:
graph.set_entry_point("generate_code")
graph.add_edge("generate_code", "check_code")
graph.add_conditional_edges(
    "check_code",
    decide_next,
    [END, "generate_code"]
)

coder_agent = graph.compile()

from IPython.display import display, Image, Markdown

display(Image(coder_agent.get_graph().draw_mermaid_png()))

"""## Run and Test the Agent"""

from langchain_core.messages import HumanMessage

def call_reflection_coding_agent(agent, prompt, verbose=False):
    events = agent.stream(
        {"messages": [HumanMessage(content=prompt)], "attempts": 0},
        stream_mode="values",
    )

    print('Running Agent. Please wait...')
    for event in events:
        if verbose:
            event["messages"][-1].pretty_print()

    print('\n\nFinal Solution:')
    print("\nDescription:\n" + event["code_solution"].prefix +
          "\nCode:\n"+event["code_solution"].imports + '\n\n' + event["code_solution"].code)

prompt = "write some code to demonstrate how to do a pivot table in pandas"
call_reflection_coding_agent(coder_agent, prompt, verbose=True)

prompt = "write some code to scrape data from any wikipedia page"
call_reflection_coding_agent(coder_agent, prompt, verbose=True)

prompt = """Give me working code to get data from Twitter using API"""
call_reflection_coding_agent(coder_agent, prompt, verbose=True)

prompt = """Give me working code to do sentiment analysis using transformers"""
call_reflection_coding_agent(coder_agent, prompt, verbose=True)

