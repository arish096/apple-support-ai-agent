# Apple Support AI Agent

An AI-powered customer support agent built for the Hiver SDE Intern Take-Home Assignment.

The system analyzes incoming Apple Support customer messages, predicts the primary support intent, retrieves similar historical Apple Support interactions, drafts a grounded response, and decides whether the request is safe to **AUTO-HANDLE** or should be **ESCALATED** to a human.

> **Core principle:** When historical evidence is weak, ambiguous, or insufficiently similar, the agent prefers escalation over unsupported automation.

---

## Features

- Intent classification for Apple Support queries
- Historical customer-support case retrieval
- Evidence-based response drafting
- Similarity-based confidence scoring
- Safety-first AUTO-HANDLE vs ESCALATE decision
- Explicit escalation reasons
- Leakage-aware golden-set evaluation
- LLM-as-judge evaluation harness
- Interactive Streamlit demo
- CLI-based inference and evaluation

---

## Architecture

```text
Customer Message
       |
       v
Intent Classification
       |
       v
Historical Case Retrieval
       |
       v
Evidence Filtering
       |
       v
Grounded Reply Draft
       |
       v
Safety Decision
   +---+---+
   |       |
   v       v
AUTO-   ESCALATE
HANDLE  TO HUMAN

The system does not automatically respond simply because a similar historical message exists.

A historical case must provide sufficiently relevant evidence before the agent considers automatic handling.

Problem

Customer-support conversations contain recurring issues, but not every incoming message should be automatically handled.

The agent is designed to:

Classify an incoming customer message into a small set of support intents.
Retrieve historically similar customer-support cases.
Use relevant historical responses as evidence for drafting a reply.
Decide whether the request can be safely handled automatically.
Escalate uncertain cases to a human with an explicit reason.

The project prioritizes evidence quality and safe escalation over maximum automation rate.

Dataset

The project uses the Twitter Customer Support Conversations dataset.

For this Apple Support agent, conversations involving the AppleSupport account were extracted and paired as:

Customer Message -> Apple Support Reply

The resulting Apple-specific interaction dataset contains:

106,646 customer -> Apple Support pairs

The original full dataset is intentionally excluded from the repository because it is large and is not required for inference.

Intent Taxonomy

The system uses 12 support intents:

Intent	Description
battery_power_issue	Battery drain, charging and power-related problems
ios_update_issue	iOS updates and update-related problems
app_issue	Application crashes, freezing and app-specific problems
device_hardware_issue	Hardware, display, camera and physical-device problems
apple_id_account_issue	Apple ID, login, verification and account access
calls_network_issue	Calls, cellular service, signal and network problems
icloud_issue	iCloud storage, syncing and related problems
keyboard_autocorrect_issue	Keyboard, typing and autocorrect problems
apple_music_media_issue	Apple Music and media-related problems
purchase_billing_issue	Purchases, billing, charges and subscriptions
accessibility_feature_issue	Accessibility features and related functionality
other_support	Issues that do not fit the defined categories

When multiple issues appear in a message, the system attempts to identify the primary support issue.

Technical Approach
1. Intent Classification

The current intent classifier uses:

TF-IDF features
Unigrams + bigrams
Logistic Regression
Balanced class weights

The classifier was trained using high-confidence keyword-based pseudo-labels generated from the development sample.

This is treated as a development baseline rather than a fully human-supervised production classifier.

2. Historical Reply Retrieval

The agent maintains a searchable index of historical Apple Support interactions.

Retrieval uses:

TF-IDF vectorization
Cosine similarity
Historical customer messages
Corresponding Apple Support replies

For each incoming message, the system retrieves the most similar historical customer-support cases.

3. Evidence Selection

Not every retrieved case is accepted as evidence.

The system prefers historical cases that satisfy:

Matching predicted intent
Sufficient similarity
Non-generic historical reply
Actionable troubleshooting guidance

Generic replies such as:

DM us and we can help.

are not considered strong evidence for automated handling.

4. Safety Decision

The agent makes a conservative decision.

AUTO-HANDLE

A request can be automatically handled when:

The customer message is sufficiently specific.
A matching historical case exists.
The similarity is above the safety threshold.
The historical response contains actionable guidance.
ESCALATE

The request is escalated when:

The message is too vague.
No sufficiently similar historical case is available.
Historical evidence is below the similarity threshold.
Useful actionable evidence cannot be established.

This prevents the system from confidently generating unsupported support responses.

Example
Customer Message
My iPhone battery is draining very quickly.
Agent Output
Intent:
battery_power_issue

Decision:
AUTO-HANDLE

Reason:
Specific customer message + sufficiently similar historical
case with matching intent and actionable guidance.

Historical Evidence:
A similar Apple Support interaction was retrieved.

Draft Reply:
Generated using the retrieved historical support guidance.
Evaluation

A manually reviewed golden set was created and subsequently audited and cleaned.

Current evaluation set:

199 examples

The exact evaluation message is excluded from its own retrieval results to reduce direct retrieval leakage.

Final Agent Results
Metric	Result
Golden examples	199
Intent accuracy	19.10%
Historical intent agreement	73.37%
AUTO-HANDLE	44
ESCALATE	155
AUTO-HANDLE rate	22.11%
Escalation rate	77.89%
Average selected similarity	0.367
Decision Distribution
ESCALATE       155  (77.89%)
AUTO-HANDLE     44  (22.11%)
Escalation Reasons
81  No sufficiently similar historical case with
    matching intent and actionable guidance

51  Usable historical evidence exists, but similarity
    is below the safety threshold

23  Customer message is too vague to safely determine
    the issue
Baseline Comparison

The intent classifier was compared with simple baselines on the cleaned 199-example benchmark.

Approach	Accuracy	Macro F1
Majority baseline	45.0%	0.062
Keyword baseline	40.0%	0.227
TF-IDF + Logistic Regression	37.5%	0.090

The current classifier does not outperform the simple baselines on this benchmark.

This result is intentionally reported rather than hidden.

The purpose of the evaluation is to understand model behavior, failure modes and safety—not to optimize a single headline metric.

What Is Misleading About My Headline Number?

The 19.10% intent accuracy should not be interpreted as a clean estimate of real-world intent understanding.

Important limitations include:

The benchmark contains only 199 examples.
The classes are highly imbalanced.
Some examples remain ambiguous even after auditing and cleaning.
The classifier was trained using weak keyword-based labels rather than a large independently hand-labelled training set.
Intent classification is only one component of the overall agent.
The agent separately evaluates historical evidence and can escalate uncertain cases.

Therefore, the 19.10% accuracy is best treated as a diagnostic benchmark, not as a standalone production-quality metric.

The more operationally important question is:

Does the agent have sufficiently strong historical evidence to safely handle the request?

Failure Analysis
1. Battery vs iOS Update Confusion

Some messages mention an iOS update together with battery, heat or device symptoms.

Example:

Since I installed the new IOS my phone is crashing everyday,
stopped working and start getting hot.

The classifier can over-weight update-related vocabulary.

Hypothesis

The classifier needs better contextual representations and improved multi-issue handling.

2. App / Hardware / Billing Boundary Confusion

Short messages can be difficult to distinguish between:

application failure
hardware failure
billing or purchase context
Hypothesis

A hierarchical taxonomy and clearer annotation guidelines could reduce these boundary errors.

3. Keyboard / Autocorrect / iOS Overlap

Keyboard issues frequently occur after system updates.

This creates overlap between:

keyboard_autocorrect_issue

and:

ios_update_issue
Hypothesis

The classifier should prioritize the concrete customer symptom over the temporal context.

4. Short and Context-Dependent Messages

Messages such as:

Can you please help

contain insufficient information to identify the underlying issue.

The system therefore escalates instead of guessing.

Hypothesis

A clarification-question strategy could improve handling of these messages.

5. Retrieval Similarity Does Not Guarantee Resolution Relevance

High cosine similarity does not necessarily mean that a historical response is appropriate for the new customer.

Therefore, retrieval is combined with:

Intent compatibility
Actionable guidance checks
Generic-response filtering
Similarity threshold
Message-specificity checks
Hypothesis

Sentence embeddings and a stronger reranker could improve evidence quality.

Safety Design

The agent intentionally favors precision over automation volume.

The decision process is:

Specific Message
      +
Matching Intent
      +
Relevant Historical Case
      +
Actionable Guidance
      +
Similarity Above Threshold
      |
      v
AUTO-HANDLE

Otherwise:

ESCALATE
    |
    v
Human Support

This makes uncertainty explicit rather than hiding it behind a confident-looking generated response.

LLM-as-Judge

The project includes an LLM-as-judge evaluation harness:

llm_judge.py

The evaluation is designed to assess:

Decision correctness
Reply quality
Evidence grounding
Safety of automation

The harness supports structured JSON responses, retries and resumable execution.

During development, the available free-tier model API was rate-limited before the full 199-example judge run could be completed.

Therefore, an unsupported full-dataset LLM-judge score is not reported.

This limitation is intentionally disclosed.

Human Evaluation

A stronger evaluation setup would independently review a subset of agent decisions for:

AUTO-HANDLE / ESCALATE correctness
Reply quality
Evidence relevance
Safety

Human judgments can then be compared with LLM-judge decisions using:

Accuracy
Cohen's Kappa
Per-dimension agreement
Streamlit Demo

The project includes an interactive Streamlit application.

Run:

streamlit run app.py

The interface displays:

Customer message
Predicted intent
AUTO-HANDLE / ESCALATE decision
Decision reason
Similarity score
Draft reply
Selected historical evidence
Top retrieved historical cases
CLI Usage

Run:

python agent.py

The CLI provides:

1 - Test one customer message
2 - Evaluate golden set
Test a Customer Message

Select:

1

Then enter a support query.

The agent returns:

Predicted intent
Action
Decision reason
Similarity score
Draft reply
Historical evidence
Retrieved cases
Evaluate the Golden Set

Select:

2

The evaluation runs on the 199-example golden set and writes results to:

data/golden_predictions.csv
Installation

Python 3.10+ recommended.

Install dependencies:

pip install pandas numpy scikit-learn joblib streamlit requests

Then run:

python agent.py

or:

streamlit run app.py
Project Structure
hiver-ai-support-agent/
│
├── agent.py
├── app.py
├── llm_judge.py
├── README.md
├── .gitignore
│
├── data/
│   ├── apple_support_conversations.csv
│   ├── golden_cleaned.csv
│   ├── golden_predictions.csv
│   ├── training_data.csv
│   └── ...
│
├── models/
│   ├── intent_classifier.joblib
│   ├── reply_retriever.joblib
│   ├── similarity_classifier.joblib
│   ├── similarity_model.py
│   ├── train_model.py
│   ├── cross_validate.py
│   └── validate_similarity.py
│
└── ...

The original full twcs.csv dataset is excluded from GitHub because of its size.

Reproducibility

The repository contains the trained model artifacts and prepared Apple Support interaction data required to run inference without downloading the original full dataset.

Basic setup:

pip install pandas numpy scikit-learn joblib streamlit requests

Run the CLI:

python agent.py

Run the Streamlit application:

streamlit run app.py

Run the golden evaluation:

python agent.py

Then select:

2
One-Week Improvement Plan

If given another week, I would prioritize:

1. Larger Human-Labeled Dataset

Create a larger independently labelled training set with clearer guidelines for overlapping intents.

2. Replace Weak Supervision

Train the classifier primarily on human-labelled examples instead of keyword-generated pseudo-labels.

3. Better Semantic Retrieval

Replace TF-IDF retrieval with sentence embeddings and evaluate a stronger reranking model.

4. Clarification Questions

Instead of immediately escalating vague requests, ask a targeted clarification question.

Example:

Could you tell me whether the issue is related to your
battery, an app, or the iPhone itself?
5. Stronger Safety Evaluation

Build a dedicated human-reviewed dataset for:

Unsafe AUTO-HANDLE decisions
Irrelevant historical evidence
Incorrect escalation
Poorly grounded replies
Decision Log
1. Apple Support was selected as the brand

It provided a sufficiently large set of historical customer-support interactions.

2. Customer -> brand reply pairs were used

This directly represents the support behavior the agent is expected to learn from.

3. A small intent taxonomy was chosen

A limited taxonomy makes classification and error analysis more interpretable.

4. Primary issue classification was used

A single customer message can contain multiple symptoms, so the system focuses on the primary support issue.

5. TF-IDF was selected for the initial classifier

It is lightweight, fast and interpretable.

6. Keyword pseudo-labeling was used for development

The manually labelled data was limited, so high-confidence keyword rules were used to expand the development training set.

7. Retrieval was separated from classification

Intent classification identifies the issue while retrieval provides historical evidence for the response.

8. Generic historical responses were filtered

Generic responses provide weak evidence for automated support.

9. Actionable guidance was required

Historical evidence should contain useful troubleshooting information.

10. A similarity threshold was added

Weak historical matches should not trigger automatic handling.

11. Vague messages are escalated

The system avoids guessing when the customer provides insufficient information.

12. Self-retrieval leakage was reduced

The exact evaluation message is excluded from its own retrieval results.

13. Simple baselines were reported

The classifier is evaluated against majority and keyword baselines.

14. Limitations are explicitly documented

Weak benchmark performance and incomplete free-tier LLM judging are reported transparently.

Limitations

This project is a research/prototype implementation rather than a production customer-support system.

Current limitations include:

Small manually reviewed evaluation set
Strong class imbalance
Ambiguous examples
Weakly supervised classifier training
TF-IDF-based semantic representation
Incomplete full-dataset LLM-as-judge evaluation
No completed human-vs-LLM agreement study
Historical support responses may contain outdated guidance
Historical similarity does not guarantee correctness for a new context
Conclusion

This project combines:

Intent Classification + Historical Retrieval + Evidence Filtering + Grounded Response Drafting + Safety-First Escalation

The key design decision is not to maximize automation.

Instead, the agent asks:

"Do I have enough relevant historical evidence to safely handle this request?"

If the evidence is insufficient, the system escalates to a human.

This makes the agent more transparent, auditable and conservative than a system that automatically responds to every similar-looking customer message.

Author : Arish Islam

Built as part of the Hiver SDE Intern Take-Home Assignment.
