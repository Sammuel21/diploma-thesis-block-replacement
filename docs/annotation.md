Thesis: Diploma
Degree: CS + DS (mAIN)
Faculty: FMFI UK Bratislava

Thesis title:
- Compressing Large Language Models via Replacement of MLP Blocks


Thesis annotation and goals:

In modern Large Language Models (LLMs) based on the Transformer architecture, Multi-Layer Perceptron (MLP) blocks typically account for approximately 80% of the total parameters. While essential for the model's expressive power, these blocks are significant bottlenecks for memory storage and inference latency. Instead of standard compression techniques such as quantization or unstructured pruning, we plan to use an alternative approach that treats each MLP block as an isolated function and attempts to replace it with a significantly smaller, more efficient approximation trained on calibration data.

The goals of this thesis are:
1) Design and implement a methodology to replace individual MLP blocks with smaller substitutes. This involves training smaller, "drop-in" network structures (e.g., shallower MLPs, linear layers, or hybrids) to mimic the original block's function, using local calibration data (input/output pairs) captured from the frozen, pre-trained model.
2) Test and evaluate the improvements and trade-offs of this replacement strategy. This includes an analysis of variations in the compression strategy and an evaluation of their impact on model quality.
