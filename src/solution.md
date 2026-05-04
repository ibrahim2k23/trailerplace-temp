# LangGraph + Session State Solution for Trailer Recommendation Agent

This document outlines how to restructure your conversational agent using **LangGraph** to solve hallucination issues caused by an overly bloated system prompt.

By using LangGraph, we break the monolithic system prompt into smaller, specialized agents (Nodes) that are only activated when their specific context is needed. We use a Session State to pass only the necessary information forward.

## 1. Session State Definition

The `SessionState` will act as the memory for the ongoing conversation, storing key flags so the LLM doesn't have to guess where it is in the flow.

```python
from typing import TypedDict, Annotated, Sequence, Any
from langchain_core.messages import BaseMessage
import operator

class SessionState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    intent: str | None           # 'sales', 'finance', 'service', 'trade_in'
    category: str | None         # e.g., 'Equipment', 'Dump', 'Utility', etc.
    slots_collected: dict[str, Any] # e.g., {"haul_weight_lbs": 10000}
    search_results: list[dict]   # Stores latest Pinecone search results
    is_interested: bool          # Set when a user shows interest in a specific listing
```

## 2. Graph Nodes & Their Respective Prompts

We will divide the logic into three main nodes:
1. **Master Router Node**: Greets, handles general FAQs, and identifies the trailer category.
2. **Category Specialist Node**: Loaded dynamically based on the `category`. Only asks questions relevant to that specific trailer.
3. **Recommendation Node**: Activates after the search is performed, focusing entirely on selling and logging interest.

### Node A: Master Router Node
**Role:** Initial entry point. Figures out the intent. If the user wants a trailer, it identifies the category.

**Prompt:**
```text
You are a friendly trailer sales assistant for TrailerPlace (Wharton TX). 
Your goal is to route the customer or find out what broad type of trailer they need.

1. INTENT ROUTING:
If the user asks about financing, service, or trade-ins, provide our contact info (979-532-1486) and offer to connect them.

2. CATEGORY IDENTIFICATION:
If the user wants a trailer, figure out what category they need based on their use case. 
- "skid steer" -> Equipment
- "landscape" -> Utility
- "V-nose" -> Enclosed
- "fuel tank" -> Diesel Tank
- "garbage" -> Dump

3. UNKNOWN CATEGORY (Edge Case):
If they say "I need a trailer" but don't specify what for, DO NOT guess. 
Ask: "What will you be hauling?" or "What kind of work are you doing?"

Once you know the category, use the `set_state_category` tool to update the system and pass them to the specialist.
```

### Node B: Category Specialist Node (Dynamic Prompting)
**Role:** Once a category is identified, this node takes over. The magic here is that we **only inject the rules for the active category** into the prompt.

**Dynamic Prompt Template:**
```text
You are the {category} Trailer Specialist for TrailerPlace.
The user is looking for a {category} trailer.

YOUR TASK: Collect the required information to perform an inventory search.
Ask ONE question at a time. Do not overwhelm the user.

{category_specific_rules}

CRITICAL RULES:
- If the item they are hauling is inherently lightweight (golf carts, mowers), DO NOT ask for weight. Silently set payload needs to 1000 lbs.
- Do not ask questions they have already answered in the chat.

Once all required information is gathered, call the `search_trailers` tool.
```

**Example of `{category_specific_rules}` injected for "Dump" trailers:**
```text
- Required Questions: What material are you hauling? Do you know the rough weight?
- Optional Questions: Do you prefer a scissor, telescopic, or standard hoist?
```

**Example of `{category_specific_rules}` injected for "Equipment" trailers:**
```text
- Required Questions: What exact equipment are you hauling? What is the rough weight? What length do you need?
- Optional Questions: Hitch preference (Bumper pull/Gooseneck)? Loading style (ramps, deckover)?
```

### Node C: Recommendation & Handoff Node
**Role:** Presents the `search_results` stored in the state, and handles the `log_product_interest` flow.

**Prompt:**
```text
You are assisting a customer who has just received search results for trailers.
Here are the options we found:
{search_results}

YOUR TASK:
1. Present the options clearly, using bullet points for specs directly from the data.
2. Add a 1-2 sentence explanation of why it fits their needs.
3. Ask ONE closing question like "Does any of these suit what you're looking for?"

If the user expresses interest in a specific unit, use the `log_product_interest` tool immediately. Wait for the tool to succeed before telling them the team will reach out.
```

## 3. Handling Edge Cases & State Transitions

- **Edge Case 1: User doesn't mention which type of trailer they want.**
  - *Handling:* The graph stays in the `Master Router Node` until the LLM successfully invokes a `set_state_category` tool. The Master Router is instructed to ask clarifying questions directly.

- **Edge Case 2: User changes their mind midway (e.g., "Actually I want a Dump trailer").**
  - *Handling:* The `Category Specialist Node` is equipped with a `clear_category` tool. If it detects a major pivot out of its domain, it triggers this tool, which updates the state (`category = None`) and forces the LangGraph router back to the `Master Router Node`.

- **Edge Case 3: Hallucinating Specs / Recommendations.**
  - *Handling:* Because the `Category Specialist` does nothing but collect slots and run searches, it never attempts to sell imaginary trailers. The `Recommendation Node` handles the presentation and is strictly bound by the `{search_results}` injected into its prompt.

## 4. LangGraph Flow Architecture (Python Pseudocode)

```python
from langgraph.graph import StateGraph, END

# Define conditions
def route_after_master(state: SessionState):
    if state["category"] is not None:
        return "specialist_node"
    return END # Wait for user input

def route_after_specialist(state: SessionState):
    if state.get("category") is None: # User changed their mind
        return "master_router_node"
    if len(state.get("search_results", [])) > 0:
        return "recommendation_node"
    return END

# Build Graph
builder = StateGraph(SessionState)

builder.add_node("master_router_node", master_agent_func)
builder.add_node("specialist_node", specialist_agent_func)
builder.add_node("recommendation_node", recommendation_agent_func)

# Base Setup
builder.set_entry_point("master_router_node")

# Edges
builder.add_conditional_edges("master_router_node", route_after_master)
builder.add_conditional_edges("specialist_node", route_after_specialist)
builder.add_edge("recommendation_node", END)

# Compile
graph = builder.compile()
```

## Summary
Yes, this approach is **highly recommended** and absolutely possible. By implementing this state-machine architecture with LangGraph:
1. You drastically reduce token usage per API call.
2. The LLM's attention mechanism isn't overwhelmed by rules it doesn't need for the current trailer type, virtually eliminating hallucinations.
3. You have explicit guardrails governing when a search can trigger and when recommendations can be presented.
