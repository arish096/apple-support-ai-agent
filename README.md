# Apple Support AI Agent

## Hiver SDE Intern — Take-Home Assignment

An AI customer-support agent built from historical AppleSupport conversations from the Twitter Customer Support dataset.

The system performs three core tasks:

1. Classifies the customer's issue into a small intent taxonomy.
2. Retrieves similar historical AppleSupport conversations and uses their resolutions as evidence.
3. Decides whether the case can be automatically handled or should be escalated to a human.

The design intentionally prioritizes safe escalation over maximizing automation.

---

## 1. Problem Framing

The goal is to build a support agent for one brand using real historical customer-support conversations.

For each incoming customer message, the agent should:

- identify the primary support intent,
- find similar historical customer cases,
- ground the proposed response in historical AppleSupport resolutions,
- decide whether the case is safe to auto-handle,
- otherwise escalate it to a human with a reason.

The central design principle is:

> High retrieval similarity is not sufficient evidence for automatic handling.

A message can be similar to historical conversations while still lacking enough actionable evidence for a safe automated response.

---

## 2. Dataset

The project uses the Twitter Customer Support dataset.

I selected AppleSupport as the target brand.

The original dataset contains conversations between customers and support accounts. I constructed customer-to-AppleSupport response pairs by identifying AppleSupport replies and pairing them with the preceding customer message.

This produced:

- 106,646 AppleSupport customer → support-reply pairs
- 104,823 unique customer messages
- 1,823 repeated customer messages
- 0 duplicate customer+reply pairs

For development and modeling, a 10,000-example sample was used rather than repeatedly processing the complete dataset.

This keeps the pipeline lightweight and reproducible while retaining a substantial amount of historical support data.

---

## 3. Intent Taxonomy

I defined a compact 12-class intent taxonomy based on recurring AppleSupport issues in the data.

| Intent | Description |
|---|---|
| `battery_power_issue` | Battery drain, charging and power-related problems |
| `ios_update_issue` | iOS updates, update failures and update-related problems |
| `app_issue` | Application crashes, freezes or application-specific problems |
| `device_hardware_issue` | Physical/device-level hardware problems |
| `apple_id_account_issue` | Apple ID, login, password and account problems |
| `calls_network_issue` | Calls, cellular service, network and signal problems |
| `icloud_issue` | iCloud-related problems |
| `keyboard_autocorrect_issue` | Keyboard, typing and autocorrect problems |
| `apple_music_media_issue` | Apple Music and media-related problems |
| `purchase_billing_issue` | Purchases, charges and billing problems |
| `accessibility_feature_issue` | Accessibility features and related problems |
| `other_support` | Cases that do not clearly fit another intent |

When multiple issues appear in one message, the primary issue is selected.

---

## 4. System Architecture

```text
Customer Message
       |
       v
Intent Classifier
       |
       v
TF-IDF Historical Retrieval
       |
       v
Top Historical Support Cases
       |
       v
Evidence Filtering
       |
       +----------------------------+
       |                            |
       v                            v
Specific + useful evidence       Weak/vague evidence
       |                            |
       v                            v
Similarity >= threshold          ESCALATE
       |
       v
AUTO-HANDLE
       |
       v
Historical Support Reply

The system has two main components.

Intent classification

A TF-IDF representation with word n-grams is used with Logistic Regression.

Historical retrieval

Customer messages from historical AppleSupport conversations are indexed using TF-IDF vectors.

The incoming message is compared against historical customer messages using cosine similarity.

5. Evidence Selection

The agent retrieves the top historical cases and applies additional filtering.

A historical case is considered usable evidence when:

Its historical intent matches the predicted intent.
The historical response is not overly generic.
The response contains actionable guidance.

Examples of actionable guidance include:

Settings
restarting
updating
checking
resetting
installing/removing
backup
troubleshooting steps

This prevents the system from treating a generic response such as "DM us and we'll help" as sufficient evidence for automation.

6. Auto-Handle vs Escalation

The agent uses a conservative decision policy.

ESCALATE when:
the customer message is too vague,
no sufficiently useful historical evidence is found,
historical evidence exists but similarity is below the safety threshold.
AUTO-HANDLE when:
the message is sufficiently specific,
matching historical evidence exists,
the historical response contains actionable guidance,
similarity is at or above the safety threshold.

The current similarity threshold is:

0.30

The threshold is only checked after evidence-quality filtering.

Therefore, similarity alone does not trigger automatic handling.

7. Example

Input:

My iPhone battery is draining very quickly after the latest iOS update.

The agent returns:

Predicted intent:
battery_power_issue

Decision:
ESCALATE

Similarity:
0.475

The case is escalated because a sufficiently similar historical case with matching intent and actionable guidance was not selected.

This demonstrates an important property of the system:

A similarity score above the threshold does not automatically mean that the case is safe to handle.

8. Golden Evaluation Set

A manually reviewed evaluation set was created from AppleSupport examples.

The final cleaned benchmark contains:

199 examples

The benchmark was manually audited and subsequently cleaned for obvious inconsistencies and duplicate examples.

The final benchmark is not treated as a perfectly independent gold standard. Some cleaning involved rule-based corrections after manual auditing.

This limitation is explicitly considered when interpreting the results.

9. Evaluation

The evaluation measures:

Intent classification
Accuracy
Macro F1
Weighted F1
Agent behavior
Auto-handle rate
Escalation rate
Retrieval similarity
Historical intent agreement
Qualitative evaluation
Failure-mode analysis
Historical evidence inspection
LLM-as-judge evaluation

The golden-set evaluation excludes exact self-matches during retrieval to reduce evaluation leakage.

10. Baseline Comparison

The current benchmark results are:

Approach	Accuracy	Macro F1
Majority-class baseline	45.0%	0.062
Keyword baseline	40.0%	0.227
TF-IDF + Logistic Regression	37.5%	0.090

The majority baseline predicts the most common class for every example.

The keyword baseline uses simple intent-specific keyword rules.

The learned TF-IDF classifier does not currently outperform these simple baselines.

This is an important negative result rather than something hidden from the evaluation.

11. Agent Results

On the cleaned 199-example benchmark, the current agent produced:

Intent accuracy:
19.1%

Historical intent agreement:
73.4%

Auto-handle:
44 / 199
22.1%

Escalate:
155 / 199
77.9%

Average selected similarity:
0.367

The agent is deliberately conservative.

The relatively high escalation rate reflects the requirement that historical evidence must be both relevant and actionable before an automatic response is allowed.

12. What Is Misleading About My Headline Number?

The 19.1% intent accuracy should not be interpreted as a clean estimate of real-world intent understanding.

There are several limitations:

The evaluation set contains only 199 examples.
The classes are highly imbalanced.
Some examples are ambiguous and can reasonably belong to more than one category.
The benchmark was manually audited and then rule-cleaned rather than being an independently labelled gold set.
The intent classifier was trained using weak keyword-based labels rather than a large independently hand-labelled training set.
The current learned classifier does not beat the simple baselines.

Therefore, the 19.1% number is most useful as a diagnostic signal and for comparing iterations, rather than as a definitive measure of production-level intent understanding.

The agent's safety behavior should be evaluated separately from intent accuracy.

13. Top 5 Failure Modes
Failure Mode 1 — Battery vs iOS Update Confusion

Messages frequently mention both battery behavior and an iOS update.

Example:

since I installed the new IOS my phone is crashing everyday,
stopped working and start getting hot.

The classifier can over-weight the update vocabulary and predict ios_update_issue even when the benchmark label emphasizes another issue.

Hypothesis

The current TF-IDF model is sensitive to high-frequency issue words but does not understand which problem is primary.

Improvement

Use better multi-intent handling and train on independently labelled examples.

Failure Mode 2 — App / Hardware / Billing Boundary Confusion

Some customer messages contain device failures, application problems and purchase-related language together.

Example:

My iPhone X is already broken after like three weeks.
It just turned off and never turned back on.

The classifier can incorrectly associate such messages with purchase or billing language because of vocabulary overlap in the training data.

Hypothesis

Weak labels and sparse examples for some classes create unstable class boundaries.

Improvement

Create a larger, balanced, human-labelled training set.

Failure Mode 3 — Keyboard / Autocorrect Overlap With App and iOS Issues

Keyboard problems frequently mention updates, applications or unusual characters.

Example:

why does the letter after h show up like this...

These short messages can be difficult to distinguish from broader iOS or application issues.

Hypothesis

TF-IDF captures word overlap but not the underlying semantic relationship between typing, autocorrect and system updates.

Improvement

Use sentence embeddings or a stronger text classifier and include more keyboard-specific labelled examples.

Failure Mode 4 — Short or Context-Dependent Messages

Messages such as:

Can you please help

contain almost no information about the actual problem.

The system deliberately escalates these cases.

This is a desirable safety behavior even if the retriever finds a high-similarity historical example.

Hypothesis

A similarity model can find lexical similarity without knowing whether enough information exists to resolve the current issue.

Improvement

Keep the specificity gate and consider asking a clarifying question before escalating.

Failure Mode 5 — Retrieval Similarity Does Not Equal Resolution Relevance

A high similarity score does not guarantee that the historical reply provides a safe resolution.

The agent therefore filters retrieved responses for:

matching intent,
non-generic content,
actionable guidance.
Hypothesis

Customer-message similarity and solution similarity are different concepts.

Improvement

Use a dedicated cross-encoder or reranker trained to measure customer-problem-to-resolution relevance.

14. Safety Design

The most important safety property is that the system should not maximize automation at the expense of response quality.

For example:

High similarity
        +
Weak/generic historical reply
        =
ESCALATE

rather than:

High similarity
        =
AUTO-HANDLE

The system also escalates vague requests.

This makes the system more conservative and easier to audit.

15. LLM-as-Judge

An LLM-as-judge evaluation harness was implemented using OpenRouter.

The judge evaluates:

whether the AUTO-HANDLE / ESCALATE decision was appropriate,
reply quality on a 1–5 scale,
whether the reply is grounded in historical evidence,
whether the response is safe to auto-handle,
a short explanation for the judgment.

The evaluation is designed to complement, rather than replace, deterministic metrics.

The free-model daily request limit of the external judge API was reached during development, so the full judge run was not completed in the current development run.

Therefore, no incomplete LLM-judge result is presented as a headline metric.

16. Human Agreement

LLM-as-judge scores should not be treated as ground truth.

A small human-reviewed subset should be used to measure judge-human agreement.

The intended process is:

Agent output
     |
     +----> LLM Judge
     |
     +----> Human Reviewer
                |
                v
        Compare judgments

Agreement should be reported for:

decision correctness,
groundedness,
safe-to-auto-handle.

This provides evidence for whether the automated judge is a reasonable proxy for human evaluation.

17. What I Would Do With One More Week
Day 1–2: Better labelled training data

Create a larger independently hand-labelled dataset with balanced coverage across all 12 intents.

This would reduce the dependence on weak keyword labels.

Day 3: Better classifier

Compare:

TF-IDF + Logistic Regression
TF-IDF + Linear SVM
sentence embeddings
lightweight transformer classifier

using the same fixed evaluation set.

Day 4: Better retrieval/reranking

Move from pure TF-IDF similarity toward semantic retrieval and add a reranking stage that evaluates whether the historical resolution actually addresses the new problem.

Day 5: Better response generation

Instead of copying a historical response directly, generate a response constrained by retrieved evidence.

The generator should not introduce unsupported troubleshooting steps.

Day 6: Evaluation

Expand the golden set and run:

intent accuracy
macro F1
retrieval quality
auto-handle precision
escalation quality
human-vs-LLM judge agreement
Day 7: Error-driven iteration

Focus model improvements on the highest-frequency confusion pairs instead of optimizing aggregate accuracy alone.

18. Decision Log
Decision 1 — Select AppleSupport as the target brand

AppleSupport provided a large number of customer-support interactions, making it practical to build a retrieval-based prototype.

Decision 2 — Pair each AppleSupport reply with the preceding customer message

This provides a simple customer → support resolution structure suitable for retrieval.

Decision 3 — Use a compact 12-intent taxonomy

A small taxonomy makes the system easier to evaluate and reduces extremely sparse categories.

Decision 4 — Use the primary issue when multiple issues appear

This creates a single-label classification problem while acknowledging that real support messages can be multi-intent.

Decision 5 — Use TF-IDF for the first retrieval system

TF-IDF is lightweight, interpretable and fast enough for the take-home prototype.

Decision 6 — Use cosine similarity

Cosine similarity provides a simple normalized measure of lexical overlap between customer messages.

Decision 7 — Exclude exact self-matches during golden evaluation

This prevents the evaluation query from retrieving its own historical example.

Decision 8 — Filter generic historical replies

Responses such as "DM us" are not considered sufficient resolution evidence.

Decision 9 — Require actionable guidance

A historical reply must contain useful troubleshooting or next-step guidance before it can support automatic handling.

Decision 10 — Add a message-specificity gate

Very short or vague customer requests are escalated instead of being automatically answered.

Decision 11 — Use conservative automation

The system only auto-handles cases when evidence passes multiple checks.

Decision 12 — Report negative model results honestly

The learned classifier currently does not beat the simple baselines, so this result is reported rather than hidden.

Decision 13 — Separate intent accuracy from agent safety

A classifier can make an incorrect intent prediction while the agent still safely escalates the case.

Decision 14 — Treat LLM-as-judge as a secondary evaluation

LLM judgments are useful for qualitative evaluation but should not replace deterministic metrics or human review.

19. Project Structure
hiver-ai-support-agent/
│
├── agent.py
├── app.py
├── analyze_data.py
├── build_conversations.py
├── check_brand.py
├── clean_golden.py
├── create_review_set.py
├── create_targeted_candidates.py
├── discover_intents.py
├── evaluate_model.py
├── find_brands.py
├── label_data.py
├── llm_judge.py
├── prepare_training_data.py
├── review_candidates.py
│
├── data/
│   ├── apple_support_conversations.csv
│   ├── golden_audited.csv
│   ├── golden_cleaned.csv
│   ├── golden_predictions.csv
│   ├── golden_review_candidates.csv
│   ├── golden_set.csv
│   ├── labeled_sample.csv
│   ├── sample.csv
│   ├── targeted_candidates.csv
│   ├── training_data.csv
│   └── twcs.csv
│
└── models/
    ├── cross_validate.py
    ├── intent_classifier.joblib
    ├── reply_retriever.joblib
    ├── similarity_classifier.joblib
    ├── similarity_model.py
    ├── train_model.py
    └── validate_similarity.py
20. Installation

From the project directory:

pip install pandas numpy scikit-learn joblib streamlit
21. Running the Agent
Interactive command-line mode
python agent.py

Select:

1 - Test one customer message

Then enter a customer-support message.

22. Running the Web UI

Start Streamlit:

streamlit run app.py

The UI provides:

customer message input,
predicted intent,
AUTO-HANDLE / ESCALATE decision,
similarity score,
decision reason,
draft reply,
selected historical evidence,
top retrieved historical cases,
explanation of the decision process.
23. Running the Golden Evaluation
python agent.py

Select:

2 - Evaluate golden set

The evaluation results are saved to:

data/golden_predictions.csv
24. Reproducibility

The headline pipeline is designed to run using the prepared sampled data and saved models rather than requiring the full raw dataset to be processed again.

The main agent and evaluation pipeline use:

Python
pandas
NumPy
scikit-learn
joblib
Streamlit

The external LLM judge is optional for running the core agent.

25. Limitations

The current prototype has several limitations:

The intent classifier is trained using weak keyword-based labels.
The golden benchmark is relatively small.
Class distribution is highly imbalanced.
Some examples are inherently ambiguous.
TF-IDF retrieval is lexical rather than deeply semantic.
Historical responses can themselves be generic or incomplete.
The current model does not outperform the simple baselines.
The full LLM-as-judge evaluation was limited by the free external API request quota.
Human-vs-LLM judge agreement still needs to be measured on a dedicated human-reviewed subset.

These limitations are intentionally reported because the objective is to provide evidence about what works and what does not, rather than presenting an inflated headline metric.

26. Summary

The current prototype demonstrates an end-to-end support-agent pipeline:

Customer message
      ↓
Intent classification
      ↓
Historical case retrieval
      ↓
Evidence filtering
      ↓
Safety decision
      ↓
Draft response

The strongest design choice is the conservative escalation policy.

Instead of assuming that a similar historical message is enough to answer automatically, the agent requires matching intent, useful historical resolution evidence and sufficient similarity.

The current classifier still needs substantial improvement. In particular, the results show that weakly supervised intent classification is the main bottleneck, while historical retrieval and conservative escalation provide a useful foundation for a safer support workflow.