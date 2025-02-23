from openai import OpenAI
from mem0.utils.factory import LlmFactory

# Prompt templates
ITERATIVE_ANALYSIS_PROMPT = """Analyze this prompt and feedback to suggest targeted improvements.
Current Prompt:
{prompt}

User Feedback:
{feedback}

Analysis Framework:
1. Key Issues: Identify specific problems/recommendations highlighted in the feedback
2. Proposed Solutions: Suggest concrete changes to address each issue
3. Impact Assessment: Focus on high-leverage modifications that will yield the greatest improvements
4. Implementation Priority: Recommend which changes should be made first

Provide an analysis with actionable recommendations for prompt enhancement."""

SEQUENTIAL_REFINEMENT_PROMPT = """Enhance this prompt based on the provided feedback:
# CURRENT PROMPT:
```
{prompt}
```

# USER FEEDBACK:
```
{feedback}
```

Please provide an improved version that:
1. Incorporates the specific feedback points only
2. Maintains clarity and conciseness
3. Always be brief and to the point.

The updated prompt should incorporate feedback while preserving core details.
Just return the updated prompt and nothing else."""

SINGLE_SHOT_PROMPT = """Enhance this prompt based on the provided feedback:
# CURRENT PROMPT:
```
{prompt}
```

# USER FEEDBACK:
```
{feedback}
```

Please provide an improved version that:
1. Incorporates the specific feedback points only
2. Maintains clarity and conciseness
3. Always be brief and to the point.
4. Update the information only when there is a conflict. If it is something additional, add it.

The updated prompt should incorporate feedback while preserving core details."""

IMPROVEMENT_PROMPT = """Based on this analysis, create an enhanced version of the prompt.
# ANALYSIS:
```
{analysis}
```

# CURRENT PROMPT:
```
{current_prompt}
```

Guidelines:
- Address the key points from the analysis
- Keep the core intent and purpose
- Make focused, impactful changes
- Ensure clarity and conciseness

Return the improved prompt only."""


class PromptOptimizer:
    def __init__(self, model: LlmFactory = LlmFactory()):
        self.llm = model

    def _get_llm_response(self, prompt: str) -> str:
        """Helper to make OpenAI API call"""
        response = self.llm.generate_response(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        return response

    def _iterative_analysis(self, base_prompt: str, feedback: str, max_steps: int) -> str:
        """Optimize prompt using iterative analysis approach"""
        current_prompt = base_prompt
        for _ in range(max_steps):
            analysis = self._get_llm_response(
                ITERATIVE_ANALYSIS_PROMPT.format(prompt=current_prompt, feedback=feedback)
            )
            current_prompt = self._get_llm_response(
                IMPROVEMENT_PROMPT.format(
                    analysis=analysis,
                    current_prompt=current_prompt
                )
            )
        return current_prompt

    def _sequential_refinement(self, base_prompt: str, feedback: str, max_steps: int) -> str:
        """Optimize prompt using sequential refinement approach"""
        current_prompt = base_prompt
        for _ in range(max_steps):
            current_prompt = self._get_llm_response(
                SEQUENTIAL_REFINEMENT_PROMPT.format(prompt=current_prompt, feedback=feedback)
            )
        return current_prompt

    def _single_shot(self, base_prompt: str, feedback: str) -> str:
        """Optimize prompt using single shot approach"""
        return self._get_llm_response(
            SINGLE_SHOT_PROMPT.format(prompt=base_prompt, feedback=feedback)
        )

    def optimize(
        self,
        base_prompt: str,
        feedback: str,
        augmentation: str = "single_shot",
        max_steps: int = 3
    ) -> str:
        """
        Optimize a prompt using feedback and specified optimization strategy.

        Args:
            base_prompt (str): Original prompt to optimize
            feedback (str): User feedback for improvement
            optimizer_type (str): One of ["iterative_analysis", "sequential_refinement", 
                                "single_shot"]
            max_steps (int): Maximum optimization steps for iterative optimizations

        Returns:
            str: Optimized prompt
        """
        optimization_methods = {
            "iterative_analysis": lambda: self._iterative_analysis(
                base_prompt, feedback, max_steps
            ),
            "sequential_refinement": lambda: self._sequential_refinement(
                base_prompt, feedback, max_steps
            ),
            "single_shot": lambda: self._single_shot(base_prompt, feedback)
        }

        if augmentation not in optimization_methods:
            raise ValueError(
                f"Unsupported augmentation type: {augmentation}. "
                "Use 'iterative_analysis', 'sequential_refinement', or 'single_shot'"
            )

        return optimization_methods[augmentation]()


def augment_prompt_with_feedback(
    base_prompt: str,
    feedback: str,
    augmentation: str = "single_shot",
    max_steps: int = 3,
    model: LlmFactory = LlmFactory()
) -> str:
    """Wrapper function for backward compatibility"""
    optimizer = PromptOptimizer(model=model)
    return optimizer.optimize(base_prompt, feedback, augmentation, max_steps)
