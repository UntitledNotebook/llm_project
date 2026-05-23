from llm_project.math_utils import answers_match, extract_answer


def test_extract_gsm8k_marker():
    assert extract_answer("We compute 2+2=4. #### 4") == "4"


def test_extract_boxed():
    assert extract_answer("Therefore \\boxed{42}.") == "42"


def test_fraction_match():
    assert answers_match("1/2", "0.5")


def test_comma_match():
    assert answers_match("1,000", "1000")
