import json

from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT

DELETE_MEMORY_SYSTEM_PROMPT = """
You are a memory manager specializing in identifying, managing, and optimizing memories. Your primary task is to analyze a list of existing relationships and determine which ones should be deleted based on the new information provided.

Input:
1. Existing Memories: A list of current memories, each containing past information.
2. New Text: The new information to be integrated into the existing memories.
3. Use "USER_ID" as node for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Guidelines:
1. Identification: Use the new information to evaluate existing memories.
2. Deletion Criteria: Delete a memory only if it meets at least one of these conditions:
   - Outdated or Inaccurate: The new information is more recent or accurate.
   - Contradictory: The new information conflicts with or negates the existing information.
3. DO NOT DELETE if their is a possibility of memories to coexist.
4. Comprehensive Analysis:
   - Thoroughly examine each existing memory against the new information and delete as necessary.
   - Multiple deletions may be required based on the new information.
5. Semantic Integrity:
   - Ensure that deletions maintain or improve the overall semantic structure of the graph.
   - Avoid deleting memories that are NOT contradictory/outdated to the new information.
6. Temporal Awareness: Prioritize recency when timestamps are available.
7. Necessity Principle: Only DELETE memories that must be deleted and are contradictory/outdated to the new information to maintain an accurate and coherent memory graph.
Note: DO NOT DELETE if their is a possibility of the memories to coexist. 

For example: 
Existing Memory: alice loves to eat pizza
New Information: Alice also loves to eat burger.
Do not delete in the above example because there is a possibility that Alice loves to eat BOTH Pizza and Burger.

Memory Format for Existing Memories:
id: text

Provide a list of deletion instructions, each specifying the memory to be deleted.
"""


def get_fact_retrieval_messages(message):
    return FACT_RETRIEVAL_PROMPT, f"Input: {message}"

def get_delete_memory_messages(retrieved_old_memory, parsed_messages):
    retrived_old_memory_string = ""
    for memory in retrieved_old_memory:
        retrived_old_memory_string += f"{memory['id']}: {memory['text']}\n"
    return DELETE_MEMORY_SYSTEM_PROMPT, f"Existing Memories: {retrived_old_memory_string}\nNew Information: {parsed_messages}"

def parse_messages(messages):
    response = ""
    for msg in messages:
        if msg["role"] == "system":
            response += f"system: {msg['content']}\n"
        if msg["role"] == "user":
            response += f"user: {msg['content']}\n"
        if msg["role"] == "assistant":
            response += f"assistant: {msg['content']}\n"
    return response

def format_entities(entities):
    if not entities:
        return ""
    
    formatted_lines = []
    for entity in entities:
        simplified = f"{entity['source']} -- {entity['relation'].upper()} -- {entity['destination']}"
        formatted_lines.append(simplified)

    return "\n".join(formatted_lines)