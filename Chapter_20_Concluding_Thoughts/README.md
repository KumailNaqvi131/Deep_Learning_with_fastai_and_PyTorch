# Chapter 20: Concluding Thoughts (and Appendices A and B)

This chapter has no code. It's about what to do next.

## The book's advice
- **Keep practising: train models on your own data.** Pick a project you care about, in a domain you know. Most of what you need is in Chapters 1-2: gather data, train quickly, clean the data with the model's help, iterate.
- **Read code and papers.** Part IV of the book (and `common/` in this repo) showed that the library is a few hundred lines of ideas you can now read. Read papers, reimplement them, check your results against the paper.
- **Write and share.** Explain what you learned in a blog post (Appendix A shows how to set up a blog with GitHub Pages). Writing for someone one step behind you is the best way to consolidate what you know.
- **Join a community.** The fast.ai forums, study groups, Kaggle. Help others: answering questions teaches you as much as asking them.
- **Go further:** the fast.ai courses (Practical Deep Learning for Coders, and "From Deep Learning Foundations to Stable Diffusion") continue where the book stops.

## Appendix B: the data project checklist (summary)
Before starting a data project, answer these questions:
1. **Strategy:** what objective does this serve? What actions (levers) will the model's predictions inform? What data could connect them? (the Drivetrain approach, Chapter 2)
2. **Data:** what's available, where, how clean, how biased? Is it labelled? Will the data in production look like the training data?
3. **Analytics:** what's the simplest baseline? How will you validate (a validation set that looks like production, Chapter 1)? What metrics matter to the people affected?
4. **Implementation:** how will the model be used in practice (Chapter 2)? What's the cost of errors? How will humans stay in the loop?
5. **Maintenance:** how will you monitor the model and detect drift and feedback loops (Chapters 2 and 3)? Who maintains it?
6. **Constraints:** legal, ethical (Chapter 3), privacy, compute and time budgets.

## Where to go from here in this repo
- Re-run any chapter with `--real` to train on the book's real datasets.
- Swap the synthetic data for your own images in `data/real/bears/<class>/` (Chapter 2) or any folder of labelled images, and reuse the scripts.
- Read `common/learner.py` and `common/vision.py` side by side with Chapters 16 and 19.
