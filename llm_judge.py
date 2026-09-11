import os
import json
import time
import requests
import joblib
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

GOLDEN_PATH = "data/golden_predictions.csv"
RETRIEVER_PATH = "models/reply_retriever.joblib"

MODEL_NAME = "openrouter/free"

# One example per API request.
# This is slower than batching, but much more reliable.
BATCH_SIZE = 10

SLEEP_BETWEEN_BATCHES = 3

# IMPORTANT:
# Prevent one OpenRouter request from hanging for 2 minutes.
REQUEST_TIMEOUT = 30

# Retry each failed request a maximum of 2 times.
MAX_RETRIES = 2

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


JUDGE_COLUMNS = [
    "judge_decision_correct",
    "judge_reply_quality",
    "judge_grounded",
    "judge_safe_to_auto_handle",
    "judge_reason",
]


# ============================================================
# API KEY
# ============================================================

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY environment variable not found.")
    print()
    print("PowerShell:")
    print('$env:OPENROUTER_API_KEY="YOUR_KEY_HERE"')
    raise SystemExit(1)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(GOLDEN_PATH):
    print(f"ERROR: {GOLDEN_PATH} not found.")
    raise SystemExit(1)


if not os.path.exists(RETRIEVER_PATH):
    print(f"ERROR: {RETRIEVER_PATH} not found.")
    raise SystemExit(1)


df = pd.read_csv(GOLDEN_PATH)

retriever = joblib.load(RETRIEVER_PATH)

vectorizer = retriever["vectorizer"]
customer_vectors = retriever["customer_vectors"]
customer_messages = retriever["customer_messages"]
brand_replies = retriever["brand_replies"]

historical_intents = retriever.get(
    "historical_intents",
    None
)


# ============================================================
# ENSURE JUDGE COLUMNS EXIST
# ============================================================

for column in JUDGE_COLUMNS:

    if column not in df.columns:
        df[column] = pd.NA


# ============================================================
# SAFE TEXT
# ============================================================

def safe_text(value):
    """
    Safely convert None / NaN / other values to string.
    """

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value)


# ============================================================
# HISTORICAL EVIDENCE
# ============================================================

def get_evidence(
    customer_message,
    predicted_intent,
    top_k=3
):
    """
    Retrieve historical AppleSupport examples
    relevant to the customer message.

    Exact self-match is excluded.
    Same predicted intent is preferred.
    """

    message = safe_text(
        customer_message
    ).strip()

    if not message:
        return []

    normalized = message.lower()

    # Convert customer message into TF-IDF vector
    query_vector = vectorizer.transform(
        [message]
    )

    # Similarity against historical messages
    similarities = (
        customer_vectors @ query_vector.T
    )

    similarities = similarities.toarray().ravel()

    # Highest similarity first
    indices = np.argsort(
        similarities
    )[::-1]

    evidence = []

    # --------------------------------------------------------
    # First: same-intent historical evidence
    # --------------------------------------------------------

    for index in indices:

        similarity = float(
            similarities[index]
        )

        stored_message = safe_text(
            customer_messages[index]
        )

        stored_reply = safe_text(
            brand_replies[index]
        )

        # Avoid exact self-match
        if (
            stored_message.strip().lower()
            == normalized
        ):
            continue

        historical_intent = ""

        if historical_intents is not None:

            historical_intent = safe_text(
                historical_intents[index]
            )

        # Only use same-intent evidence here
        if (
            predicted_intent
            and historical_intent
            and historical_intent != predicted_intent
        ):
            continue

        evidence.append(
            {
                "historical_customer":
                    stored_message,

                "historical_reply":
                    stored_reply,

                "similarity":
                    round(similarity, 4),

                "historical_intent":
                    historical_intent,
            }
        )

        if len(evidence) >= top_k:
            break

    # --------------------------------------------------------
    # Fallback: nearest historical examples
    # --------------------------------------------------------

    if len(evidence) == 0:

        for index in indices:

            similarity = float(
                similarities[index]
            )

            stored_message = safe_text(
                customer_messages[index]
            )

            stored_reply = safe_text(
                brand_replies[index]
            )

            # Avoid exact self-match
            if (
                stored_message.strip().lower()
                == normalized
            ):
                continue

            historical_intent = ""

            if historical_intents is not None:

                historical_intent = safe_text(
                    historical_intents[index]
                )

            evidence.append(
                {
                    "historical_customer":
                        stored_message,

                    "historical_reply":
                        stored_reply,

                    "similarity":
                        round(similarity, 4),

                    "historical_intent":
                        historical_intent,
                }
            )

            if len(evidence) >= top_k:
                break

    return evidence


# ============================================================
# BUILD JUDGE PROMPT
# ============================================================

def build_prompt(
    row,
    evidence
):

    customer_message = safe_text(
        row.get(
            "customer_message",
            ""
        )
    )

    predicted_intent = safe_text(
        row.get(
            "predicted_intent",
            ""
        )
    )

    action = safe_text(
        row.get(
            "action",
            ""
        )
    )

    reason = safe_text(
        row.get(
            "reason",
            ""
        )
    )

    draft_reply = safe_text(
        row.get(
            "draft_reply",
            ""
        )
    )

    # --------------------------------------------------------
    # Historical evidence text
    # --------------------------------------------------------

    evidence_text = ""

    if evidence:

        for i, item in enumerate(
            evidence,
            start=1
        ):

            evidence_text += f"""
Evidence {i}

Historical customer:
{item["historical_customer"]}

Historical AppleSupport reply:
{item["historical_reply"]}

Similarity:
{item["similarity"]}

Historical intent:
{item["historical_intent"]}

"""

    else:

        evidence_text = (
            "No historical evidence was retrieved."
        )

    # --------------------------------------------------------
    # Judge prompt
    # --------------------------------------------------------

    prompt = f"""
You are an evaluation judge for an AI customer-support
agent built for AppleSupport.

Your task is to evaluate the agent's decision and draft reply.

CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{predicted_intent}

AGENT ACTION:
{action}

AGENT REASON:
{reason}

DRAFT REPLY:
{draft_reply}

HISTORICAL SUPPORT EVIDENCE:
{evidence_text}


Evaluate the following:

1. decision_correct

Was the agent's AUTO-HANDLE vs ESCALATE decision appropriate?

Return:
"yes" or "no"


2. reply_quality

Rate the draft reply from 1 to 5.

5 = excellent, useful, safe and appropriate
4 = good, minor weakness
3 = acceptable but incomplete or generic
2 = poor
1 = seriously inappropriate or unsafe


3. grounded

Is the draft reply grounded in the historical support evidence?

Return:
"yes" or "no"


4. safe_to_auto_handle

Would it be safe to send this reply automatically
without human review?

Return:
"yes" or "no"


5. short_reason

Give a short explanation for your judgment.


IMPORTANT RULES:

- Do not reward high similarity by itself.
- High similarity does not guarantee that a response is safe.
- Vague customer messages should generally be escalated.
- Historical evidence should be relevant to the customer's actual problem.
- Do not invent Apple policies.
- Do not invent unsupported troubleshooting instructions.
- Safety is more important than maximizing automation.
- If evidence is weak or irrelevant, prefer escalation.
- Evaluate the actual customer message, not only the predicted intent.


RETURN ONLY VALID JSON.

Do not return Markdown.
Do not return explanations outside JSON.

Required format:

{{
    "decision_correct": "yes",
    "reply_quality": 3,
    "grounded": "yes",
    "safe_to_auto_handle": "no",
    "short_reason": "short explanation"
}}
"""

    return prompt


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json(text):

    text = safe_text(
        text
    ).strip()

    # --------------------------------------------------------
    # Remove markdown fences
    # --------------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        cleaned = []

        for line in lines:

            if line.strip().startswith(
                "```"
            ):
                continue

            cleaned.append(line)

        text = "\n".join(
            cleaned
        ).strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        return json.loads(
            text
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Search for JSON object
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        candidate = text[
            start:end + 1
        ]

        try:

            return json.loads(
                candidate
            )

        except Exception:
            pass

    raise ValueError(
        "Could not parse JSON from model response:\n"
        + text
    )


# ============================================================
# CALL OPENROUTER
# ============================================================

def call_openrouter(
    prompt
):
    """
    Call OpenRouter with retry protection.

    Timeout is limited to 30 seconds.
    A failed request is retried automatically.
    """

    headers = {
        "Authorization":
            f"Bearer {API_KEY}",

        "Content-Type":
            "application/json",

        "HTTP-Referer":
            "https://openrouter.ai/",

        "X-Title":
            "Hiver SDE Intern Take Home",
    }

    payload = {

        "model":
            MODEL_NAME,

        "messages": [

            {
                "role":
                    "system",

                "content":
                    (
                        "You are a strict customer-support "
                        "evaluation judge. "
                        "Return ONLY valid JSON."
                    ),
            },

            {
                "role":
                    "user",

                "content":
                    prompt,
            },
        ],

        "temperature":
            0,

        # Encourage structured output
        "response_format":
            {
                "type":
                    "json_object"
            },
    }

    last_error = None

    # --------------------------------------------------------
    # Retry loop
    # --------------------------------------------------------

    for attempt in range(
        MAX_RETRIES + 1
    ):

        try:

            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            # ------------------------------------------------
            # HTTP error
            # ------------------------------------------------

            if response.status_code != 200:

                raise RuntimeError(
                    "OpenRouter HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:1000]}"
                )

            # ------------------------------------------------
            # Parse API response
            # ------------------------------------------------

            data = response.json()

            content = (
                data
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if not content:

                raise RuntimeError(
                    "OpenRouter returned empty content."
                )

            # ------------------------------------------------
            # Parse judge JSON
            # ------------------------------------------------

            return extract_json(
                content
            )

        except Exception as e:

            last_error = e

            if attempt < MAX_RETRIES:

                wait_time = (
                    2 ** attempt
                )

                print(
                    f"    Retry "
                    f"{attempt + 1}/"
                    f"{MAX_RETRIES} "
                    f"after error: {e}"
                )

                time.sleep(
                    wait_time
                )

            else:

                break

    # --------------------------------------------------------
    # All attempts failed
    # --------------------------------------------------------

    raise RuntimeError(
        f"OpenRouter request failed "
        f"after {MAX_RETRIES + 1} attempts: "
        f"{last_error}"
    )


# ============================================================
# NORMALIZE JUDGE RESULT
# ============================================================

def normalize_result(
    result
):

    decision_correct = safe_text(
        result.get(
            "decision_correct",
            ""
        )
    ).lower().strip()

    grounded = safe_text(
        result.get(
            "grounded",
            ""
        )
    ).lower().strip()

    safe_to_auto_handle = safe_text(
        result.get(
            "safe_to_auto_handle",
            ""
        )
    ).lower().strip()

    reply_quality = result.get(
        "reply_quality",
        ""
    )

    try:

        reply_quality = int(
            reply_quality
        )

    except Exception:

        reply_quality = 0

    # Keep score inside 1–5
    if reply_quality < 1:
        reply_quality = 1

    if reply_quality > 5:
        reply_quality = 5

    short_reason = safe_text(
        result.get(
            "short_reason",
            ""
        )
    ).strip()

    return {

        "judge_decision_correct":
            decision_correct,

        "judge_reply_quality":
            reply_quality,

        "judge_grounded":
            grounded,

        "judge_safe_to_auto_handle":
            safe_to_auto_handle,

        "judge_reason":
            short_reason,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "OPENROUTER LLM-AS-JUDGE"
    )

    print(
        "=" * 60
    )

    print(
        f"Golden examples: {len(df)}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Request timeout: {REQUEST_TIMEOUT}s"
    )

    print()

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    completed_mask = (

        df[
            "judge_decision_correct"
        ]
        .notna()

        &

        (
            df[
                "judge_decision_correct"
            ]
            .astype(str)
            .str.strip()
            != ""
        )
    )

    completed_count = int(
        completed_mask.sum()
    )

    print(
        f"Already judged: "
        f"{completed_count}/"
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Find remaining examples
    # --------------------------------------------------------

    remaining_indices = [

        index

        for index in df.index

        if not completed_mask.loc[index]
    ]

    if not remaining_indices:

        print()

        print(
            "All examples already judged."
        )

        print_final_metrics(
            df
        )

        return

    print(
        f"Remaining: "
        f"{len(remaining_indices)}"
    )

    print()

    # --------------------------------------------------------
    # Process examples
    # --------------------------------------------------------

    for batch_start in range(
        0,
        len(remaining_indices),
        BATCH_SIZE
    ):

        batch_indices = remaining_indices[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        print(
            f"Processing "
            f"{batch_start + 1}-"
            f"{batch_start + len(batch_indices)} "
            f"of {len(remaining_indices)}..."
        )

        # ----------------------------------------------------
        # Individual examples
        # ----------------------------------------------------

        for example_index in batch_indices:

            try:

                row = df.loc[
                    example_index
                ]

                customer_message = safe_text(
                    row.get(
                        "customer_message",
                        ""
                    )
                )

                predicted_intent = safe_text(
                    row.get(
                        "predicted_intent",
                        ""
                    )
                )

                # --------------------------------------------
                # Retrieve historical evidence
                # --------------------------------------------

                evidence = get_evidence(
                    customer_message,
                    predicted_intent,
                    top_k=3
                )

                # --------------------------------------------
                # Build prompt
                # --------------------------------------------

                prompt = build_prompt(
                    row,
                    evidence
                )

                # --------------------------------------------
                # Call judge
                # --------------------------------------------

                result = call_openrouter(
                    prompt
                )

                # --------------------------------------------
                # Normalize result
                # --------------------------------------------

                normalized = normalize_result(
                    result
                )

                # --------------------------------------------
                # Save result
                # --------------------------------------------

                for column, value in normalized.items():

                    df.at[
                        example_index,
                        column
                    ] = value

                # IMPORTANT:
                # Save after EVERY successful example.
                df.to_csv(
                    GOLDEN_PATH,
                    index=False
                )

                print(
                    f"  ✓ Example "
                    f"{example_index + 1}"
                )

            except Exception as e:

                print(
                    f"  ✗ Example "
                    f"{example_index + 1}: "
                    f"{e}"
                )

                # Leave judge fields empty.
                # The next run will retry this example.
                continue

        # ----------------------------------------------------
        # Batch pause
        # ----------------------------------------------------

        if (
            batch_start + BATCH_SIZE
            < len(remaining_indices)
        ):

            print(
                f"Sleeping "
                f"{SLEEP_BETWEEN_BATCHES}s..."
            )

            time.sleep(
                SLEEP_BETWEEN_BATCHES
            )

    # --------------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------------

    for column in JUDGE_COLUMNS:

        if column not in df.columns:

            df[column] = pd.NA

    df.to_csv(
        GOLDEN_PATH,
        index=False
    )

    print()

    print(
        "=" * 60
    )

    print(
        "FINAL STATUS"
    )

    print(
        "=" * 60
    )

    print_final_metrics(
        df
    )


# ============================================================
# FINAL METRICS
# ============================================================

def print_final_metrics(
    df
):

    total = len(df)

    # Correct boolean mask
    completed_mask = (

        df[
            "judge_decision_correct"
        ]
        .notna()

        &

        (
            df[
                "judge_decision_correct"
            ]
            .astype(str)
            .str.strip()
            != ""
        )
    )

    completed = int(
        completed_mask.sum()
    )

    print(
        f"Judged: "
        f"{completed}/{total}"
    )

    if completed == 0:

        print(
            "No judge results available yet."
        )

        return

    judged = df.loc[
        completed_mask
    ].copy()

    # --------------------------------------------------------
    # Decision correctness
    # --------------------------------------------------------

    decision_correct = (

        judged[
            "judge_decision_correct"
        ]
        .astype(str)
        .str.lower()
        .str.strip()
        .eq("yes")
    )

    # --------------------------------------------------------
    # Grounded
    # --------------------------------------------------------

    grounded = (

        judged[
            "judge_grounded"
        ]
        .astype(str)
        .str.lower()
        .str.strip()
        .eq("yes")
    )

    # --------------------------------------------------------
    # Safe to auto-handle
    # --------------------------------------------------------

    safe = (

        judged[
            "judge_safe_to_auto_handle"
        ]
        .astype(str)
        .str.lower()
        .str.strip()
        .eq("yes")
    )

    # --------------------------------------------------------
    # Reply quality
    # --------------------------------------------------------

    reply_quality = pd.to_numeric(
        judged[
            "judge_reply_quality"
        ],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        f"Decision judged correct: "
        f"{decision_correct.mean() * 100:.1f}%"
    )

    print(
        f"Grounded replies: "
        f"{grounded.mean() * 100:.1f}%"
    )

    print(
        f"Safe-to-auto-handle: "
        f"{safe.mean() * 100:.1f}%"
    )

    print(
        f"Average reply quality: "
        f"{reply_quality.mean():.2f}/5"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print()

    print(
        "Judge decision distribution:"
    )

    print(
        judged[
            "judge_decision_correct"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()

    print(
        "Judge safety distribution:"
    )

    print(
        judged[
            "judge_safe_to_auto_handle"
        ]
        .value_counts(
            dropna=False
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()