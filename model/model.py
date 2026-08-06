# core/graph.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import mlflow

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.base import BaseCheckpointSaver


def build_chat_graph(
    *,
    model_id: str,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    run_name_model: str = "chat_model",
    **llm_kwargs,
):
    """
    Build a minimal LangGraph chat pass:

        START -> call_model -> END

    This function:
      - Creates the LLM
      - Builds the graph
      - Returns (graph, ainvoke)

    No assumptions about task, prompts, datasets, or evaluation.
    """

    # --- LLM creation (integrated) ---

    base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in environment or arguments.")

    llm: BaseChatModel = ChatOpenAI(
        model=model_id,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        **llm_kwargs,
    )

    llm_to_use = llm.with_config({"run_name": run_name_model})

    # --- Node ---

    @mlflow.trace(name="call_model")
    def call_model(state: MessagesState, *, config: Optional[Dict[str, Any]] = None):
        messages: List[AnyMessage] = state["messages"]
        ai = llm_to_use.invoke(messages, config=config)
        return {"messages": [ai]}

    # --- Graph ---

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)

    graph = builder.compile(checkpointer=checkpointer)

    # --- Convenience wrapper ---

    @mlflow.trace(name="graph_ainvoke")
    async def ainvoke(messages: Sequence[AnyMessage], *, config: Optional[Dict[str, Any]] = None) -> MessagesState:
        return await graph.ainvoke({"messages": list(messages)}, config=dict(config or {}))

    return graph, ainvoke
