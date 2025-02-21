from openai import OpenAI
import os

client = OpenAI()


def optimize_prompt_with_feedback(
    base_prompt: str,
    feedback: str,
    optimizer_type: str = "prompt_memory",
    model: str = "gpt-4o-mini",
    max_steps: int = 3
) -> str:
    """
    Optimize a prompt using feedback and specified optimization strategy.

    Args:
        base_prompt (str): Original prompt to optimize
        feedback (str): User feedback for improvement
        optimizer_type (str): One of ["gradient", "metaprompt", "prompt_memory"]
        model (str): OpenAI model to use
        max_steps (int): Maximum optimization steps for gradient/metaprompt

    Returns:
        str: Optimized prompt
    """
    client = OpenAI()

    # Gradient optimization prompt template
    GRADIENT_PROMPT = """Analyze this prompt and feedback to identify areas for improvement:

Current Prompt:
{prompt}

User Feedback:
{feedback}

Think about:
1. What specific issues does the feedback highlight?
2. How can the prompt be modified to address these issues?
3. What minimal changes would make the most impact?

Provide your analysis and recommendations."""

    # Metaprompt optimization template
    METAPROMPT = """You are optimizing a prompt based on user feedback.

Current Prompt:
{prompt}

User Feedback:
{feedback}

Create an enhanced version of the prompt that:
1. Addresses the specific feedback provided
2. Maintains the original intent
3. Makes minimal but effective changes

Return only the improved prompt."""

    # Prompt memory template (single-shot)
    MEMORY_PROMPT = """Update this prompt based on the user's feedback:

Current Prompt:
{prompt}

User Feedback:
{feedback}

Return an improved version that incorporates the feedback while maintaining the original purpose."""

    def get_llm_response(prompt: str) -> str:
        """Helper to make OpenAI API call"""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    if optimizer_type == "gradient":
        # Two-step optimization
        current_prompt = base_prompt
        for step in range(max_steps):
            # Step 1: Analysis
            analysis = get_llm_response(
                GRADIENT_PROMPT.format(prompt=current_prompt, feedback=feedback)
            )

            # Step 2: Improvement
            improvement_prompt = f"""Based on this analysis, generate an improved version of the prompt:

Analysis:
{analysis}

Current Prompt:
{current_prompt}

Return only the improved prompt."""
            current_prompt = get_llm_response(improvement_prompt)

        return current_prompt

    elif optimizer_type == "metaprompt":
        # Iterative meta-learning approach
        current_prompt = base_prompt
        for step in range(max_steps):
            current_prompt = get_llm_response(
                METAPROMPT.format(prompt=current_prompt, feedback=feedback)
            )
        return current_prompt

    elif optimizer_type == "prompt_memory":
        # Single-shot optimization
        return get_llm_response(
            MEMORY_PROMPT.format(prompt=base_prompt, feedback=feedback)
        )

    else:
        raise ValueError(
            f"Unsupported optimizer type: {optimizer_type}. "
            "Use 'gradient', 'metaprompt', or 'prompt_memory'"
        )

tool_feedback = {
    "type": "function",
    "strict": True,
    "function": {
        "name": "is_feedback",
        "description": (
            "Check if the user message contains feedback or instructions about "
            "how the agent/LLM should behave or respond. Feedback typically "
            "includes suggestions to modify behavior, style, tone, or approach. "
            "Common patterns include 'be more X', 'try to X', 'please use X', "
            "or phrases with 'should', 'need to', 'too', 'more/less'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string", 
                    "description": "The user message to analyze for feedback."
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0-1 for classification",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": ["response", "confidence"]
        },
        "returns": {
            "type": "boolean",
            "description": "True if message contains feedback, False otherwise"
        }
    }
}


tool_statement = {
    "type": "function",
    "strict": True,
    "function": {
        "name": "is_statement",
        "description": (
            "Determine if the user message is an actionable request or task. "
            "This includes direct commands ('write', 'create', 'find'), "
            "questions that require action ('can you...', 'could you...'), and "
            "implicit requests ('I need...', 'I want...'). Returns true for "
            "messages asking the agent to perform a task (e.g. 'write an email', "
            "'summarize this', 'help me with...'). Returns false for feedback "
            "about behavior ('be more concise'), questions about capabilities "
            "('can you access...'), or simple statements ('I like pizza')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "The user message to analyze"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0-1 for classification",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": ["response", "confidence"]
        }
    }
}


prompt = """
You are an expert classifier that determines if user messages are feedback or task 
requests. Feedback messages provide guidance on how to improve behavior/responses. 
Task requests ask you to perform a specific action.

Key characteristics of feedback:
- Suggests changes to communication style, tone, or behavior
- Often starts with phrases like "be more...", "try to...", "please use..."
- Focuses on HOW you should respond, not WHAT to do
- Usually contains words like "should", "need to", "too", "more/less"

Key characteristics of task requests:
- Asks you to perform a specific action or provide information
- Often starts with action verbs like "write", "create", "explain", "find"
- May be phrased as questions ("can you...") or statements ("I need...")
- Focuses on WHAT you should do, not HOW to do it

Examples of feedback:
- "Be more concise in your responses"
- "Please use simpler language"
- "Your tone should be more formal"
- "Include more specific details"
- "You're too verbose - keep it shorter"
- "Make sure to add examples"

Examples of task requests:
- "Write an email to my boss"
- "Summarize this article"
- "Help with my homework"
- "Create a shopping list"
- "Tell me a joke"
- "Explain photosynthesis"

Classify the user message based on these criteria. Use the appropriate function call:
- is_feedback() for feedback about behavior/style
- is_statement() for actionable task requests

Important: If you are unsure about the classification, default to is_statement().
Consider the context - if the message could be interpreted as both feedback and a task,
prioritize the task interpretation.

For example:
"Write it more formally" -> is_statement() (primary goal is to rewrite something)
"Make it shorter" -> is_feedback() (pure style guidance)
"Can you write this again but shorter?" -> is_statement() (explicit rewrite request)
"""


class Email:
    def __init__(self,):
        self.base_prompt = "Write an email"

    def llm_call(self, task_request):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"{task_request}\n\n{self.base_prompt}"}]
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    email = Email()
