from llm_project.evaluation.mmlu_eval import extract_mmlu_answer, mmlu_answers_match


def test_extract_mmlu_boxed_letters():
    assert extract_mmlu_answer("Reasoning goes here. \\boxed{A}") == "A"
    assert extract_mmlu_answer("Reasoning goes here. \\boxed{a}") == "A"


def test_extract_mmlu_boxed_numbers():
    assert extract_mmlu_answer("Reasoning goes here. \\boxed{1}") == "A"
    assert extract_mmlu_answer("Reasoning goes here. \\boxed{4}") == "D"


def test_extract_mmlu_explicit_final_marker():
    assert extract_mmlu_answer("Reasoning. The final answer is: B") == "B"
    assert extract_mmlu_answer("Reasoning. Answer is 3.") == "C"


def test_extract_mmlu_rejects_invalid_or_unmarked_answers():
    assert extract_mmlu_answer("Reasoning goes here. \\boxed{E}") is None
    assert extract_mmlu_answer("Reasoning goes here. \\boxed{0}") is None
    assert extract_mmlu_answer("There are 4 choices, so I choose A.") is None
    assert extract_mmlu_answer("Unrelated text.") is None


def test_mmlu_answers_match_normalized_final_answer():
    assert mmlu_answers_match("Reasoning. \\boxed{1}", "A")
    assert mmlu_answers_match("Reasoning. \\boxed{D}", "D")
    assert not mmlu_answers_match("Reasoning. \\boxed{2}", "A")
