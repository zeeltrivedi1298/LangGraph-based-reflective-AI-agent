# Build_a_reflective_self_correcting_code_generation_ai_agent


Reflective Self-Correcting Code Generation AI Agent (LangGraph)

This project implements a Reflective, Self-Correcting AI Agent using LangGraph
, LangChain
, and OpenAI models.
The agent follows the Reflection Pattern: it generates code, executes it, detects errors, and refines the solution iteratively until it works

langgraph_project_5_build_a_ref…

.

🚀 Features

Code Generation – Uses GPT-based models to create Python code solutions.

Execution & Reflection – Runs the generated code, captures errors, and provides feedback.

Iterative Correction Loop – Repeats generation and refinement until code runs without errors.

Stopping Conditions – Terminates after success or after a maximum attempt threshold.

User Feedback – Allows restarting refinement if the user isn’t satisfied.
