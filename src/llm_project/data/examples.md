# Dataset Answer Format Examples

Crawled on 2026-05-24 to compare answer formats across:

- `AI-MO/NuminaMath-CoT`, filtered with `where="source"='gsm8k'`
- `openai/gsm8k`, config `main`
- `cais/mmlu`, subset `elementary_mathematics`

The target generation format should follow the NuminaMath-CoT convention: show reasoning, then end with a final answer in `\boxed{...}`. For final evaluation datasets, keep the raw gold labels for scoring, but prompt/generated completions should use the same boxed final-answer convention.

## Format Summary

| Dataset | Raw gold / solution format | Recommended completion ending |
| --- | --- | --- |
| NuminaMath-CoT, `source=gsm8k` | Assistant solution text usually ends with `\boxed{answer}`. | `Therefore, the final answer is \boxed{6}.` |
| Original GSM8K | Natural-language solution with calculator annotations and a final `#### answer` marker. | Convert the final marker to `\boxed{answer}` in model-facing examples. |
| MMLU | Multiple-choice `answer` is a class-label index whose names are `A`, `B`, `C`, `D`. | For generative prompts, end with the selected letter, e.g. `\boxed{C}`. |

For numeric tasks, prefer only the canonical scalar inside the box, without units or currency symbols: use `\boxed{83200}` instead of `\boxed{\$83,200}`. This keeps answer extraction simple and avoids mixing formatting tokens with the value being scored.

## Side-by-Side GSM8K Examples

### Example 1: Carl Candy Bars

NuminaMath-CoT row 123, `source=gsm8k`:

```text
Problem:
Every time Carl earned $0.50 he would go to the corner market and buy a candy bar. Carl's neighbor said he would pay him $0.75 every week for taking out his trash. At the end of four weeks, how many candy bars will Carl be able to buy?

Final line:
So, the final answer, encapsulated as required, is $\boxed{6}$.
```

Original GSM8K row 7172, same question:

```text
Answer excerpt:
He is paid $0.75 every week for 4 weeks, so Carl made .75 * 4 = $<<.75*4=3.00>>3.00 in 4 weeks
Each candy bar cost $0.50 and he made $3.00 so 3.00/.50 = <<3.00/.50=6>>6 candy bars
#### 6
```

Normalized target ending:

```text
Therefore, Carl can buy \boxed{6} candy bars.
```

### Example 2: Ruth Math Class Hours

NuminaMath-CoT row 175, `source=gsm8k`:

```text
Problem:
Ruth goes to school 8 hours a day and 5 days a week. She is in math class 25% of this time. How many hours per week does she spend in math class?

Final line:
Therefore, Ruth spends $\boxed{10}$ hours per week in math class.
```

Original GSM8K row 5061, same question:

```text
Answer excerpt:
She is in school for 40 hours because 5 x 8 = <<5*8=40>>40
She is in math class for 10 hours a week because 40 x .25 = <<40*.25=10>>10
#### 10
```

Normalized target ending:

```text
Therefore, Ruth spends \boxed{10} hours per week in math class.
```

## MMLU Examples

MMLU stores the gold answer as a class-label index, not a free-form solution string. The `cais/mmlu` `elementary_mathematics` rows below expose class-label names `A`, `B`, `C`, `D`, so `answer: 2` means `C`.

### Example 3: Equation

Raw MMLU row 0:

```text
Question:
What is the value of p in 24 = 2p?

Choices:
A. p = 4
B. p = 8
C. p = 12
D. p = 24

Raw answer:
2, i.e. C
```

Normalized target ending:

```text
The correct choice is C. Therefore, the final answer is \boxed{C}.
```

### Example 4: Daily Miles

Raw MMLU row 1:

```text
Question:
Ms. Perez drove a total of 40 miles in 5 days. She drove the same number of miles each day. How many miles did Ms. Perez drive each day?

Choices:
A. 5
B. 7
C. 8
D. 9

Raw answer:
2, i.e. C
```

Normalized target ending:

```text
The correct choice is C. Therefore, the final answer is \boxed{C}.
```

## Source URLs

- NuminaMath-CoT filtered rows: `https://datasets-server.huggingface.co/filter?dataset=AI-MO%2FNuminaMath-CoT&config=default&split=train&where=%22source%22%3D%27gsm8k%27&offset=0&length=3`
- Original GSM8K Carl row: `https://datasets-server.huggingface.co/search?dataset=openai%2Fgsm8k&config=main&split=train&query=Carl%27s%20neighbor&offset=0&length=5`
- Original GSM8K Ruth row: `https://datasets-server.huggingface.co/search?dataset=openai%2Fgsm8k&config=main&split=train&query=Ruth%20goes%20to%20school%208%20hours&offset=0&length=3`
- MMLU rows: `https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu&config=elementary_mathematics&split=test&offset=0&length=5`
