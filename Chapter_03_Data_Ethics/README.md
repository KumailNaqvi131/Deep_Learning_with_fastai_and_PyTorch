# Chapter 3: Data Ethics

The book's chapter is mostly about examples and process: recourse and accountability, feedback loops, bias, disinformation, how to identify ethical issues in your project, the power of diversity, and the role of policy. Three small simulations make the technical ideas concrete.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Representation bias: a good overall metric hides failure on an under-represented group | Bias (Gender Shades) | `01_representation_bias.py` |
| 2 | A feedback loop: predictive policing with equal true crime rates diverges; correcting for how data is collected | Feedback Loops | `02_feedback_loops.py` |
| 3 | Historical bias and fairness metrics (selection rate, equal opportunity); removing the protected column doesn't fix proxies | Bias, Fairness, Accountability, and Transparency | `03_fairness_metrics.py` |

## Run

```bash
cd Chapter_03_Data_Ethics
python 01_representation_bias.py
python 02_feedback_loops.py
python 03_fairness_metrics.py
```

## Key takeaways
- Break your metrics down by subgroup. An aggregate number can hide a model that fails some people.
- If a model's outputs influence its future training data, expect feedback loops. Collect data independently and monitor what the model changes.
- Labels can encode past discrimination, and other columns can act as proxies for protected attributes. The fix has to come from the data and the process, not just from the code.
- Questions to ask about any project (from the book): should we even be doing this? What bias is in the data? Can the code and data be audited? What are the error rates for different groups? What is the accuracy of a simple rule-based alternative? What processes handle appeals and mistakes? How diverse is the team?
