# 🍎 Apple Support AI Agent

> AI-powered customer support agent built for the **Hiver SDE Intern Take-Home Assignment**.

An intelligent support system that understands incoming Apple Support messages, identifies the customer's intent, retrieves similar historical support cases, drafts an evidence-grounded reply, and decides whether the request should be **AUTO-HANDLED** or **ESCALATED to a human**.

The core principle is simple:

> **When evidence is weak or ambiguous, the system escalates instead of guessing.**

---

## 🚀 Features

- 🧠 **Intent Classification**
  - Classifies customer messages into 12 Apple Support intents.

- 🔎 **Historical Case Retrieval**
  - Finds similar customer-support interactions from historical Apple Support data.

- 💬 **Grounded Reply Generation**
  - Drafts replies using evidence from historically resolved cases.

- 🛡️ **Safety-First Decision Making**
  - Automatically handles only cases with sufficient evidence.
  - Escalates uncertain or low-confidence cases.

- 📊 **Similarity Confidence**
  - Uses similarity scores to measure how closely historical cases match the incoming message.

- 🧪 **Golden-Set Evaluation**
  - Evaluated on 199 manually reviewed examples.

- ⚖️ **Baseline Comparison**
  - Compared against majority-class and keyword-based baselines.

- 🤖 **LLM-as-Judge Harness**
  - Includes structured evaluation for reply quality, grounding, and safety.

- 🌐 **Streamlit Demo**
  - Interactive interface for testing the support agent.

- 💻 **CLI Evaluation**
  - Run predictions and evaluation directly from the terminal.

---

# 🏗️ System Architecture

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
              Safety Decision Layer
                 ┌─────┴─────┐
                 ▼           ▼
            AUTO-HANDLE   ESCALATE🎯 Problem Statement

The goal is to build an AI customer-support agent that can:

1. Understand an incoming customer message.
2. Classify the primary support intent.
3. Find similar historical customer-support interactions.
4. Use historical responses as evidence.
5. Draft a relevant support response.
6. Decide whether the case can be safely handled automatically.
7. Escalate uncertain cases to a human agent.

The system prioritizes evidence quality and safe escalation rather than maximizing automation.
📊 Dataset

The project uses the Customer Support on Twitter dataset.

The original dataset contains millions of tweets and replies from multiple customer-support brands.

For this project, the selected brand is:

🍎 Apple Support

After reconstructing customer → Apple Support interactions:

Apple Support conversations: 106,646
Evaluation examples: 199
Original full dataset is not included in the repository because of its size.
🧩 Intent Taxonomy

The system uses 12 support intents:

#	Intent
1	battery_power_issue
2	ios_update_issue
3	app_issue
4	device_hardware_issue
5	apple_id_account_issue
6	calls_network_issue
7	icloud_issue
8	keyboard_autocorrect_issue
9	apple_music_media_issue
10	purchase_billing_issue
11	accessibility_feature_issue
12	other_support
⚙️ Technical Approach
1. Intent Classification

The intent classifier uses:

TF-IDF vectorization
Unigrams + bigrams
Logistic Regression
Balanced class weights

The initial training labels were created using high-confidence keyword-based pseudo-labels from the development data.

This is treated as a development baseline rather than a fully human-supervised production classifier.

2. Historical Retrieval

The system retrieves similar historical customer messages using:

TF-IDF
Cosine similarity

Each retrieved case contains:

Customer Message
        +
Historical Apple Support Reply

This allows the system to use previous support interactions as evidence.

3. Evidence Filtering

Retrieved cases are filtered using several conditions:

Predicted intent should match the historical case.
Similarity should be sufficiently high.
Response should contain useful information.
Response should provide actionable guidance.
Generic responses are treated as weak evidence.

For example: "DM us and we can help."
is considered weaker evidence than a response containing specific troubleshooting guidance.
🛡️ Safety Decision Logic

The system follows a conservative decision policy.

AUTO-HANDLE

A case can be automatically handled when:
Specific Customer Message
        +
Matching Intent
        +
Relevant Historical Case
        +
Actionable Guidance
        +
Similarity Above Threshold

ESCALATE

The system escalates when:

The customer message is vague.
No sufficiently similar historical case exists.
Evidence does not match the predicted intent.
Similarity is below the safety threshold.
Retrieved responses are generic.
No actionable guidance is available.

This design intentionally favors precision over automation rate.

💡 Example
Customer Message
My iPhone battery is draining extremely fast after the update.

Agent Output
Intent:
battery_power_issue

Decision:
AUTO-HANDLE

Similarity:
High

Evidence:
Similar historical Apple Support case found.

Reason:
A relevant historical case contains actionable troubleshooting
guidance for a similar battery problem.

📈 Evaluation Results

The system was evaluated on a manually reviewed golden set of:

199 examples
| Metric                      |     Result |
| --------------------------- | ---------: |
| Intent Accuracy             | **19.10%** |
| Historical Intent Agreement | **73.37%** |
| AUTO-HANDLE                 |     **44** |
| ESCALATE                    |    **155** |
| Auto-handle Rate            | **22.11%** |
| Escalation Rate             | **77.89%** |
| Average Selected Similarity |  **0.367** |

The low automation rate is intentional because the system is designed to escalate cases when evidence is insufficient.

🆚 Baseline Comparison

The agent was compared against simple baselines.

| Model                        | Accuracy | Macro F1 |
| ---------------------------- | -------: | -------: |
| Majority Class               |    45.0% |    0.062 |
| Keyword Baseline             |    40.0% |    0.227 |
| TF-IDF + Logistic Regression |    37.5% |    0.090 |

Important observation

The current classifier does not outperform the simple baselines.

This result is intentionally reported rather than hidden because the assignment emphasizes honest evaluation and understanding failure cases.

⚠️ What Is Misleading About My Headline Number?

The 19.10% intent accuracy should not be interpreted as a clean estimate of real-world production performance.

Reasons include:

The evaluation set contains only 199 examples.
The intent distribution is imbalanced.
Some customer messages are inherently ambiguous.
Initial training labels rely partly on weak supervision.
Intent classification is only one component of the complete agent.
Retrieval quality and escalation safety are evaluated separately.

Therefore, the number should be viewed as a diagnostic benchmark, not a production-quality metric.

🔥 Top Failure Modes
1. Battery vs iOS Update Confusion

Example:
Since I installed the new IOS my phone is crashing everyday,
stopped working and start getting hot.

Hypothesis

The classifier may over-weight update-related vocabulary even when the actual problem is battery/device behavior.

Possible improvement

Use contextual representations and multi-issue classification.

2. App / Hardware / Billing Boundaries

Some messages contain overlapping signals.

Possible improvement

Use a hierarchical intent taxonomy and clearer annotation guidelines.

3. Keyboard / Autocorrect / iOS Overlap

Keyboard problems can contain strong iOS/update vocabulary.

Possible improvement

Prioritize the concrete customer symptom over temporal context.

4. Very Short Messages

Example:
Can you please help?

There is not enough information to safely determine the intent.

Current behavior
ESCALATE

Possible improvement

Ask a clarification question before escalating.

5. Similarity ≠ Resolution Relevance

A text can be lexically similar without providing a useful solution.

Possible improvement

Combine:

Intent compatibility
Semantic similarity
Actionability
Response specificity
Learned reranking
🤖 LLM-as-Judge

The repository includes an LLM-based evaluation harness.

The judge evaluates:

Decision correctness
Reply quality
Evidence grounding
Safety

The evaluator produces structured JSON and supports retries/resumable evaluation.

Because the free-tier API used for evaluation was rate-limited before the complete 199-example run, an unsupported full LLM-judge score is not reported.

🧑‍💻 Human Evaluation

A human evaluation framework is included for reviewing:

Decision correctness
Reply quality
Evidence relevance
Safety

Agreement can be measured using:

Accuracy
Cohen's Kappa
Per-dimension agreement
🌐 Streamlit Demo

Run:
streamlit run app.py

The interface displays:

• Customer message
• Predicted intent
• AUTO-HANDLE / ESCALATE decision
• Decision reason
• Similarity score
• Draft reply
• Historical evidence
• Top similar cases
💻 CLI

Run:
python agent.py
