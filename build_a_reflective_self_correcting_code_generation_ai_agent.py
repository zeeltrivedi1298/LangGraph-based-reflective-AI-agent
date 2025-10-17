#!/usr/bin/env python3
"""
Reflection Coding Agent (LangChain + LangGraph)

What this script does:
1) Instantiates an OpenAI chat model via LangChain.
2) Defines a structured output schema (prefix/imports/code) for code generation.
3) Builds a small LangGraph that:
   - Generates code for a given prompt
   - Executes the returned imports + code to catch errors
   - If errors occur, feeds them back to the model to regenerate (up to MAX_ATTEMPTS)
4) Runs a few sample prompts.

Requirements (versions roughly aligned with your notebook):
  pip install "langchain>=0.3.0" "langchain-openai>=0.2.0" \
              "langchain-community>=0.3.0" "langgraph>=0.2.60"

Environment:
  - Set OPENAI_API_KEY in your environment.
  - Optional: set OPENAI_MODEL (defaults to a reasonable model).
"""

import os
import sys
from typing import Annotated, List, TypedDict, Optional

# --- 1) ENV + MODEL SETUP -----------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit(
        "ERROR: Please set the environment variable OPENAI_API_KEY before running.\n"
        "Example (macOS/Linux): export OPENAI_API_KEY='sk-...'\n"
    )

# Pick a sensible default model; you can override via env var.
# If you specifically have access to "gpt-5-mini", set OPENAI_MODEL=gpt-5-mini
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- 2) IMPORTS ---------------------------------------------------------------

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import HumanMessage

# --- 3) STRUCTURED OUTPUT SCHEMA ---------------------------------------------

class Code(BaseModel):
    """Schema for code solutions to questions about coding."""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Just the import statements of the code")
    code: str = Field(description="Code blocks not including import statements")

# --- 4) MODEL + SYSTEM PROMPT -------------------------------------------------

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

CODE_GEN_SYS_PROMPT = [
    (
        "system",
        """You are a coding assistant.
Ensure any code you provide can be executed with all required imports and variables defined.
Make sure point 3 below has executable code and run-ready functions.

Structure your answer EXACTLY as:
  1) a prefix describing the code solution
  2) the imports (if no imports needed keep it empty string)
  3) the functioning code blocks

Return results using the provided structured output schema.
Here is the user question:""",
    )
]

# Ask the LLM to adhere to our Code schema
code_generator = llm.with_structured_output(Code)

# --- 5) GRAPH STATE -----------------------------------------------------------

class CodeGenState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error_flag : "yes"/"no" for whether execution failed
        messages   : conversation history (prompts, reflections)
        code_solution : last Code object returned by the model
        attempts   : number of tries so far
    """
    error_flag: str
    messages: Annotated[List[AnyMessage], add_messages]
    code_solution: Code
    attempts: int

# --- 6) NODES -----------------------------------------------------------------

def generate_code(state: CodeGenState) -> CodeGenState:
    """Generate code solution structured as prefix/imports/code."""
    print("\n--- GENERATING CODE SOLUTION ---")
    msgs = state["messages"]
    attempts_so_far = state["attempts"]

    # Call the LLM with the growing conversation
    code_soln: Code = code_generator.invoke(CODE_GEN_SYS_PROMPT + msgs)

    # Record assistant message for traceability
    new_msg_content = (
        "Here is my solution attempt:\n\n"
        f"Description: {code_soln.prefix}\n\n"
        f"Imports: {code_soln.imports}\n\n"
        f"Code:\n{code_soln.code}"
    )
    msgs.append(("assistant", new_msg_content))
    attempts_so_far += 1

    return {
        "messages": msgs,
        "code_solution": code_soln,
        "attempts": attempts_so_far,
        "error_flag": state.get("error_flag", "no"),
    }

def check_code_execution(state: CodeGenState) -> CodeGenState:
    """Exec the imports, then exec the code; if errors, push them back as new user messages."""
    print("\n--- CHECKING CODE EXECUTION ---")
    msgs = state["messages"]
    code_soln: Code = state["code_solution"]
    imports_str = code_soln.imports or ""
    code_str = code_soln.code or ""
    attempts = state["attempts"]

    # First attempt to import dependencies
    try:
        exec(imports_str, {})
    except Exception as e:
        print("--- CODE IMPORT CHECK: FAILED ---")
        error_msg = (
            "Import test failed!\n"
            "Here is the exception details:\n"
            f"{e}\n\n"
            "Please fix the import section."
        )
        msgs.append(("user", error_msg))
        return {
            "code_solution": code_soln,
            "attempts": attempts,
            "messages": msgs,
            "error_flag": "yes",
        }

    # Then attempt to run the provided code
    try:
        scope = {}
        exec(f"{imports_str}\n{code_str}", scope)
    except Exception as e:
        print("--- CODE BLOCK CHECK: FAILED ---")
        error_msg = (
            "Your code solution failed the execution test!\n"
            "Here are the exception details:\n"
            f"{e}\n\n"
            "Reflect on this error and your prior attempt to solve the problem.\n"
            "(1) State what you think went wrong with the prior solution\n"
            "(2) Try to solve this problem again.\n\n"
            "Return the FULL SOLUTION using the structured output with prefix, imports, and code."
        )
        msgs.append(("user", error_msg))
        return {
            "code_solution": code_soln,
            "attempts": attempts,
            "messages": msgs,
            "error_flag": "yes",
        }

    print("--- NO ERRORS FOUND ---")
    return {
        "code_solution": code_soln,
        "attempts": attempts,
        "messages": msgs,
        "error_flag": "no",
    }

# --- 7) CONDITIONAL ROUTING ---------------------------------------------------

MAX_ATTEMPTS = 3

def decide_next(state: CodeGenState) -> str:
    """If success or attempts exhausted => end, else cycle back to generate."""
    err = state["error_flag"]
    attempts = state["attempts"]
    if err == "no" or attempts >= MAX_ATTEMPTS:
        print("--- DECISION: FINISH ---")
        return END
    else:
        print("--- DECISION: RETRY ---")
        return "generate_code"

# --- 8) BUILD GRAPH -----------------------------------------------------------

graph = StateGraph(CodeGenState)
graph.add_node("generate_code", generate_code)
graph.add_node("check_code", check_code_execution)

graph.set_entry_point("generate_code")
graph.add_edge("generate_code", "check_code")
graph.add_conditional_edges("check_code", decide_next, [END, "generate_code"])

coder_agent = graph.compile()

# --- 9) RUNNER ---------------------------------------------------------------

def call_reflection_coding_agent(agent, prompt: str, verbose: bool = False):
    """Convenience function to run the agent on a prompt and print the final solution."""
    print("\n==================================================")
    print(f"PROMPT: {prompt}")
    print("==================================================\n")

    events = agent.stream(
        {"messages": [HumanMessage(content=prompt)], "attempts": 0, "error_flag": "no"},
        stream_mode="values",
    )

    last_event: Optional[CodeGenState] = None
    print("Running agent...")
    for event in events:
        last_event = event
        if verbose and "messages" in event and event["messages"]:
            try:
                # Pretty print the latest message if possible
                msg = event["messages"][-1]
                if hasattr(msg, "pretty_print"):
                    msg.pretty_print()
            except Exception:
                pass

    if not last_event:
        print("No result from the agent.")
        return

    solution: Code = last_event["code_solution"]
    print("\n\nFinal Solution")
    print("--------------------------------------------------")
    print("Description:\n", solution.prefix)
    print("\nImports:\n", solution.imports)
    print("\nCode:\n", solution.code)
    print("--------------------------------------------------\n")



if __name__ == "__main__":
   
    demos = [
        "code to find the factorial of a given number",
        "write some code to demonstrate how to do a pivot table in pandas",
        "write some code to scrape data from any wikipedia page",
        "Give me working code to do sentiment analysis using transformers",
    ]
    for p in demos:
        call_reflection_coding_agent(coder_agent, p, verbose=False)
