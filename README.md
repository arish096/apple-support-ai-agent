# Apple Support AI Agent

An evidence-grounded AI customer support agent built for the **Hiver SDE Intern Take-Home Assignment**.

The system takes an incoming Apple Support customer message, predicts its primary intent, retrieves similar historical support interactions, drafts a response using historical evidence, and makes a conservative **AUTO-HANDLE / ESCALATE** decision.

> **Core principle:** When historical evidence is weak, ambiguous, or insufficiently similar, the agent prefers escalation over unsupported automation.

---

## Overview

Customer-support data is noisy, conversational, and often ambiguous. A useful support agent therefore needs more than a text generator.

This project combines:

**Intent Classification → Historical Retrieval → Evidence Filtering → Reply Drafting → Safety Decision**

The agent is designed to answer one key question:

> **"Do I have enough relevant historical evidence to safely handle this request?"**

If the answer is no, the request is escalated to a human.

---

## Key Features

- Intent classification for Apple Support queries
- Historical customer-support case retrieval
- TF-IDF based similarity search
- Evidence filtering before response generation
- Grounded response drafting
- Similarity-based confidence scoring
- Safety-first `AUTO-HANDLE` vs `ESCALATE` decisions
- Explicit escalation reasons
- Leakage-aware golden-set evaluation
- LLM-as-judge evaluation harness
- Interactive Streamlit interface
- CLI-based inference and evaluation

---

## System Architecture

```text
                    Customer Message
                           │
                           ▼
                  Intent Classification
                           │
                           ▼
                 Historical Case Retrieval
                           │
                           ▼
                    Evidence Filtering
                           │
                           ▼
                   Grounded Reply Draft
                           │
                           ▼
                    Safety Decision
                     ┌─────┴─────┐
                     │           │
                     ▼           ▼
                 AUTO-HANDLE  ESCALATE
                               │
                               ▼
                           Human Agent

The system does not automatically respond simply because a similar historical message exists.

Historical evidence must satisfy relevance and actionability requirements before automatic handling is considered.

Problem Statement

The goal is to build an AI support agent that can:

Classify an incoming customer message into a small set of support intents.
Retrieve historically similar Apple Support interactions.
Use relevant historical responses as evidence for drafting a reply.
Decide whether the request can be safely handled automatically.
Escalate uncertain cases to a human with an explicit reason.

The project prioritizes evidence quality and safe escalation over maximum automation rate.

Dataset

The project uses the Twitter Customer Support Conversations (TWCS) dataset.

For this project, conversations involving the AppleSupport account were extracted and reconstructed into:

Customer Message → Apple Support Reply
Dataset Statistics
Metric	Value
Apple Support conversation pairs	106,646
Golden evaluation examples	199
Full dataset	Excluded from repository

The original full dataset is intentionally excluded from GitHub because of its size. The repository contains the prepared Apple Support data and model artifacts required for inference.

Intent Taxonomy

The system currently uses 12 support intents:

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
other_support	Issues outside the defined categories

When multiple issues appear in one message, the system attempts to identify the primary support issue.

Technical Approach
1. Intent Classification

The current classifier uses:

TF-IDF features
Unigrams + bigrams
Logistic Regression
Balanced class weights

Development training data was expanded using high-confidence keyword-based pseudo-labels.

This is explicitly treated as a development baseline, rather than a fully human-supervised production classifier.

2. Historical Reply Retrieval

Historical Apple Support interactions are indexed for retrieval.

The retrieval pipeline uses:

Customer Message
       ↓
TF-IDF Vectorization
       ↓
Cosine Similarity
       ↓
Top Historical Cases

Each retrieved case contains:

Historical customer message
Corresponding Apple Support response
3. Evidence Filtering

A retrieved case is not automatically treated as valid evidence.

The system prefers cases with:

Matching predicted intent
Sufficient similarity
Non-generic historical responses
Actionable troubleshooting guidance

Generic responses such as:

DM us and we can help.

are treated as weak evidence.

4. Safety Decision
AUTO-HANDLE

A request can be automatically handled when:

The customer message is sufficiently specific.
A matching historical case exists.
Similarity exceeds the configured safety threshold.
The historical response contains actionable guidance.
ESCALATE

A request is escalated when:

The message is too vague.
No sufficiently similar historical case exists.
Historical evidence is below the safety threshold.
Useful actionable evidence cannot be established.

This prevents the system from confidently generating unsupported responses.

Example
Customer Message
My iPhone battery is draining very quickly.
Agent
Intent:
battery_power_issue

Decision:
AUTO-HANDLE

Reason:
Specific customer message with sufficiently similar
historical evidence containing actionable guidance.

The response is drafted using the retrieved historical support guidance.

Evaluation

A manually reviewed and audited golden evaluation set was created from real customer-support examples.

Golden Set

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
Reason	Count
No sufficiently similar historical case with matching intent and actionable guidance	81
Usable evidence exists but similarity is below safety threshold	51
Customer message is too vague	23
Baseline Comparison

The classifier was evaluated against simple baselines on the cleaned 199-example benchmark.

Approach	Accuracy	Macro F1
Majority baseline	45.0%	0.062
Keyword baseline	40.0%	0.227
TF-IDF + Logistic Regression	37.5%	0.090
Interpretation

The current classifier does not outperform the simple baselines on this benchmark.

This result is intentionally reported rather than hidden.

The purpose of the evaluation is to understand:

Model behavior
Failure modes
Evidence quality
Safety decisions

rather than optimizing a single headline number.

What Is Misleading About My Headline Number?

The 19.10% intent accuracy should not be interpreted as a clean estimate of real-world intent understanding.

Important limitations include:

The benchmark contains only 199 examples.
Classes are highly imbalanced.
Some examples remain ambiguous.
The classifier uses weak keyword-based training labels.
Intent classification is only one component of the agent.
The agent separately evaluates historical evidence.
The agent can escalate uncertain requests.

Therefore, the 19.10% accuracy is best treated as a diagnostic benchmark, not a production-quality metric.

The more operationally important question is:

Does the agent have sufficiently strong historical evidence to safely handle the request?

Failure Analysis
1. Battery vs iOS Update Confusion

Messages can mention an iOS update together with battery, heat, or device symptoms.

Example:

Since I installed the new IOS my phone is crashing everyday,
stopped working and start getting hot.
Hypothesis

The classifier overweights update-related vocabulary.

Improvement: Use stronger contextual representations and improved multi-issue handling.

2. App / Hardware / Billing Boundary Confusion

Short messages can be difficult to distinguish between:

Application failures
Hardware failures
Billing/purchase issues
Hypothesis

A hierarchical taxonomy and clearer annotation guidelines could reduce boundary errors.

3. Keyboard / Autocorrect / iOS Overlap

Keyboard problems frequently occur after system updates.

This creates overlap between:

keyboard_autocorrect_issue

and:

ios_update_issue
Hypothesis

The classifier should prioritize the concrete customer symptom over temporal context.

4. Short and Context-Dependent Messages

Messages such as:

Can you please help

do not contain enough information to identify the underlying issue.

The system therefore escalates instead of guessing.

Improvement

Introduce a clarification-question strategy.

5. Retrieval Similarity ≠ Resolution Relevance

High cosine similarity does not guarantee that the retrieved historical response is appropriate for the new customer.

The system therefore combines:

Intent compatibility
Actionable guidance checks
Generic-response filtering
Similarity threshold
Message-specificity checks
Improvement

Use sentence embeddings and a stronger reranker.

Safety Design

The system intentionally favors precision over automation volume.

Specific Message
       +
Matching Intent
       +
Relevant Historical Case
       +
Actionable Guidance
       +
Similarity Above Threshold
       │
       ▼
   AUTO-HANDLE

Otherwise:

ESCALATE
   │
   ▼
Human Support

This makes uncertainty explicit rather than hiding it behind a confident-looking generated response.

LLM-as-Judge

The repository includes an LLM-as-judge evaluation harness:

llm_judge.py

It is designed to evaluate:

Decision correctness
Reply quality
Evidence grounding
Safety of automation

The harness supports structured JSON responses, retries, and resumable execution.

During development, the available free-tier model API was rate-limited before the complete 199-example judge run could be completed.

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
Demo

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
CLI

Run:

python agent.py

The CLI provides:

1 - Test one customer message
2 - Evaluate golden set
Test a Customer Message

Select:

1

The agent returns:

Predicted intent
Action
Decision reason
Similarity score
Draft reply
Historical evidence
Retrieved cases
Evaluate Golden Set

Select:

2

Results are written to:

data/golden_predictions.csv
Installation
Requirements
Python 3.10+
pandas
numpy
scikit-learn
joblib
streamlit
requests

Install:

pip install pandas numpy scikit-learn joblib streamlit requests
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

The repository contains prepared Apple Support interaction data and trained model artifacts required for inference.

Install dependencies
pip install pandas numpy scikit-learn joblib streamlit requests
Run CLI
python agent.py
Run Streamlit
streamlit run app.py
Run Evaluation
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

Could you tell me whether the issue is related
to your battery, an app, or the iPhone itself?
5. Stronger Safety Evaluation

Build a dedicated human-reviewed dataset for:

Unsafe AUTO-HANDLE decisions
Irrelevant historical evidence
Incorrect escalation
Poorly grounded replies
Engineering Decision Log

Key non-obvious decisions made during development:

AppleSupport was selected as the target brand because it provides a sufficiently large set of historical support interactions.
Customer → brand reply pairs were reconstructed to represent actual support behavior.
A compact intent taxonomy was used to make classification and error analysis interpretable.
Primary issue classification was selected because one message may contain multiple symptoms.
TF-IDF was selected initially because it is lightweight, fast, and interpretable.
Keyword pseudo-labeling was used for development because the initial manually labelled dataset was limited.
Retrieval was separated from classification so that intent prediction and evidence retrieval remain independently inspectable.
Generic historical replies were filtered because they provide weak evidence for automation.
Actionable guidance was required before historical evidence could support automatic handling.
A similarity threshold was introduced to prevent weak matches from triggering AUTO-HANDLE.
Vague messages are escalated rather than forcing an uncertain prediction.
Self-retrieval leakage was reduced by excluding the exact evaluation message from retrieval.
Simple baselines were reported to provide a meaningful comparison.
Limitations are explicitly documented instead of hiding weak benchmark results.
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

Intent Classification
        +
Historical Retrieval
        +
Evidence Filtering
        +
Grounded Response Drafting
        +
Safety-First Escalation

The key design decision is not to maximize automation.

Instead, the agent asks:

"Do I have enough relevant historical evidence to safely handle this request?"

If the evidence is insufficient, the system escalates to a human.

This makes the agent more transparent, auditable, and conservative than a system that automatically responds to every similar-looking customer message.

Author

Arish Islam

Built as part of the Hiver SDE Intern Take-Home Assignment.
