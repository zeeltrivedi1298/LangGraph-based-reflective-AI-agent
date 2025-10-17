# -*- coding: utf-8 -*-
"""
Reflective Self-Correcting Code Generation AI Agent
---------------------------------------------------

This script builds a self-correcting AI code generation agent using LangGraph, LangChain, and OpenAI models.
It demonstrates how an AI can generate code, execute it, detect errors, and refine its own output iteratively.

Workflow:
1. Generate initial code using GPT-5-mini
2. Execute and validate code
3. Reflect and retry if errors occur
4. Stop after success or reaching maximum retries

Dependencies:
    pip install langchain langchain-openai langchain-community langgraph pydantic

"""

import os
from typing import TypedDict, Annotated, List

# --- ENV SETUP ---
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field

# Ensure your API key is available as environment variable
# os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_KEY>"

# --- LLM SETUP ---
llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0)

CODE_GEN_SYS_PROMPT = [
    (
        "system",
        """You are a coding assistant.
        Ensure any code you provide can be executed with all required imports and variables defined.
        Structure your answer as follows:
          1) prefix: description of the solution
          2) imports: all necessary imports
          3) code: runnable implementation
        Here is the user question:""",
    )
]

# --- DATA MODEL ---
class Code(BaseModel):
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Import statements")
    code: str = Field(description="Executable code")

# Initialize LLM for structured output
code_generator = llm.with_structured_output(Code)

# --- STATE DEFINITION ---
class CodeGenState(TypedDict):
    error_flag: str
    messages: Annotated[List[AnyMessage], add_messages]
    code_solution: Code
    attempts: int


# --- NODE 1: CODE GENERATION ---
def generate_code(state: CodeGenState) -> CodeGenState:
    """Generate code solution using GPT."""
    print("\n--- GENERATING CODE SOLUTION ---")
    msgs = state["messages"]
    attempts = state["attempts"]

    code_soln = code_generator.invoke(CODE_GEN_SYS_PROMPT + msgs)

    new_msg = (
        f"Here is my solution attempt:\n\n"
        f"Description: {code_soln.prefix}\n\n"
        f"Imports: {code_soln.imports}\n\n"
        f"Code:\n{code_soln.code}"
    )

    msgs.append(("assistant", new_msg))
    attempts += 1

    return {
        "messages": msgs,
        "code_solution": code_soln,
        "attempts": attempts,
        "error_flag": "no"
    }


# --- NODE 2: EXECUTION & ERROR CHECK ---
def check_code_execution(state: CodeGenState) -> CodeGenState:
    """Execute generated code and detect runtime/import errors."""
    print("\n--- CHECKING CODE EXECUTION ---")
    msgs = state["messages"]
    code_soln = state["code_solution"]
    imports_str = code_soln.imports
    code_str = code_soln.code
    attempts = state["attempts"]

    # Try imports
    try:
        exec(imports_str, {})
    except Exception as e:
        print("--- IMPORT ERROR ---")
        error_msg = f"Import failed: {e}\nPlease fix the import section."
        msgs.append(("user", error_msg))
        return {
            "messages": msgs,
            "code_solution": code_soln,
            "attempts": attempts,
            "error_flag": "yes"
        }

    # Try main code
    try:
        scope = {}
        exec(f"{imports_str}\n{code_str}", scope)
    except Exception as e:
        print("--- CODE EXECUTION FAILED ---")
        error_msg = (
            f"Code execution failed with error: {e}\n"
            f"Reflect on this error and improve your previous solution.\n"
            f"Return a full corrected version with prefix, imports, and code."
        )
        msgs.append(("user", error_msg))
        return {
            "messages": msgs,
            "code_solution": code_soln,
            "attempts": attempts,
            "error_flag": "yes"
        }

    print("--- CODE EXECUTED SUCCESSFULLY ---")
    return {
        "messages": msgs,
        "code_solution": code_soln,
        "attempts": attempts,
        "error_flag": "no"
    }


# --- ROUTING DECISION ---
MAX_ATTEMPTS = 3

def decide_next(state: CodeGenState) -> str:
    """Decide whether to retry or finish."""
    if state["error_flag"] == "no" or state["attempts"] >= MAX_ATTEMPTS:
        print("--- DECISION: FINISH ---")
        return "__end__"
    print("--- DECISION: RETRY ---")
    return "generate_code"


# --- GRAPH DEFINITION ---
def build_reflective_agent():
    """Build the LangGraph reflective code generation agent."""
    graph = StateGraph(CodeGenState)
    graph.add_node("generate_code", generate_code)
    graph.add_node("check_code", check_code_execution)
    graph.set_entry_point("generate_code")
    graph.add_edge("generate_code", "check_code")
    graph.add_conditional_edges("check_code", decide_next, [END, "generate_code"])
    return graph.compile()


# --- AGENT RUNNER ---
def call_reflection_coding_agent(agent, prompt: str, verbose: bool = False):
    """Run the self-correcting code agent for a given prompt."""
    print("\n\n===========================================")
    print(f"Running agent for prompt: {prompt}")
    print("===========================================\n")

    events = agent.stream(
        {"messages": [HumanMessage(content=prompt)], "attempts": 0},
        stream_mode="values",
    )

    for event in events:
        if verbose:
            print("\n--- Agent Event ---")
            print(event["messages"][-1])

    print("\n✅ Final Solution:")
    print("\nDescription:\n" + event["code_solution"].prefix)
    print("\nImports:\n" + event["code_solution"].imports)
    print("\nCode:\n" + event["code_solution"].code)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    coder_agent = build_reflective_agent()

    # Example prompts
    prompts = [
        "write some code to demonstrate how to do a pivot table in pandas",
        "write some code to scrape data from any wikipedia page",
        "Give me working code to get data from Twitter using API",
        "Give me working code to do sentiment analysis using transformers"
    ]

    for p in prompts:
        call_reflection_coding_agent(coder_agent, p, verbose=True)
