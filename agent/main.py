from typing import Dict, TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    name: str
    age: int
    skills: list
    result: str

def first_node(state: AgentState) -> AgentState:
    state["result"] = f"Welcome {state['name']}!. "
    return state

def second_node(state: AgentState) -> AgentState:
    state["result"] = state["result"] + f"You are {state['age']} years old."
    return state

def third_node(state: AgentState) -> AgentState:
    state["result"] = state["result"] + f" You have the following skills {(", ").join(state["skills"])}"
    return state

graph = StateGraph(AgentState)

graph.add_node("personalize_name", first_node)
graph.add_node("user_age", second_node)
graph.add_node("user_skill", third_node)

graph.add_edge("personalize_name", "user_age")
graph.add_edge("user_age", "user_skill")

graph.set_entry_point("personalize_name")
graph.set_finish_point("user_skill")

app = graph.compile()

result = app.invoke({"name": "Charles", "age": 23, "skills": ["python", "JavaScript"]})
print(result)
