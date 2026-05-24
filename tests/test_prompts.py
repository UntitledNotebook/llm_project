from llm_project.data.prompts import build_mmlu_prompt


def test_build_mmlu_prompt_describes_single_choice_boxed_answers():
    prompt = build_mmlu_prompt(
        "abstract_algebra",
        "Which object is a group?",
        ["Option one", "Option two", "Option three", "Option four"],
    )

    assert "single-choice question" in prompt
    assert "There is exactly one correct answer" in prompt
    assert "`\\boxed{A}` or `\\boxed{1}`" in prompt
    assert "`\\boxed{B}` or `\\boxed{2}`" in prompt
    assert "`\\boxed{C}` or `\\boxed{3}`" in prompt
    assert "`\\boxed{D}` or `\\boxed{4}`" in prompt
    assert "Show your reasoning" in prompt
    assert prompt.endswith("Solution:\n")
