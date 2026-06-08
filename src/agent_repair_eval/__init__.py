"""Agent Repair Eval — RecoverFlow.

This package records objective execution-feedback trajectories for LLM coding agents and
then analyzes the resulting state sequences with Markov-style models.

Quick start (Colab / Jupyter):
    from agent_repair_eval.colab import run_eval
    results = run_eval("Qwen/Qwen2.5-Coder-0.5B-Instruct", n_problems=20)
"""

__version__ = "0.1.0"

from agent_repair_eval.colab import run_eval, display_results, compare_runs  # noqa: F401
