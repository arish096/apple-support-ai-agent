import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score


# ==========================================
# APPLE SUPPORT AI AGENT
# ==========================================

INTENT_MODEL_PATH = "models/intent_classifier.joblib"
RETRIEVER_PATH = "models/reply_retriever.joblib"
GOLDEN_PATH = "data/golden_cleaned.csv"

SIMILARITY_THRESHOLD = 0.30


print("\n========================================")
print("        APPLE SUPPORT AI AGENT")
print("========================================")


# ==========================================
# LOAD MODELS
# ==========================================

print("\nLoading intent classifier...")
intent_model = joblib.load(INTENT_MODEL_PATH)

print("Loading historical reply retriever...")
retriever = joblib.load(RETRIEVER_PATH)

vectorizer = retriever["vectorizer"]
customer_vectors = retriever["customer_vectors"]
customer_messages = retriever["customer_messages"]
brand_replies = retriever["brand_replies"]
historical_intents = retriever["historical_intents"]

print("Models loaded successfully.")


# ==========================================
# RETRIEVE HISTORICAL CASES
# ==========================================

def retrieve_cases(message, top_k=5, exclude_message=None):

    query_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        query_vector,
        customer_vectors
    )[0]

    # Prevent self-match during evaluation
    if exclude_message is not None:

        excluded = exclude_message.strip().lower()

        for i, stored_message in enumerate(customer_messages):

            if stored_message.strip().lower() == excluded:
                similarities[i] = -1.0

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        if similarities[index] < 0:
            continue

        results.append({
            "similarity": float(similarities[index]),
            "customer_message": customer_messages[index],
            "brand_reply": brand_replies[index],
            "historical_intent": historical_intents[index]
        })

    return results


# ==========================================
# MESSAGE SPECIFICITY
# ==========================================

def is_specific_message(message):

    text = message.lower().strip()

    words = text.split()

    if len(words) < 4:
        return False

    vague_phrases = [
        "can you please help",
        "please help",
        "help me",
        "fix this",
        "fix it",
        "what's going on",
        "whats going on",
        "is this fixable",
        "this fixable",
        "please fix",
        "help please",
        "do something"
    ]

    for phrase in vague_phrases:

        if phrase in text and len(words) < 10:
            return False

    issue_signals = [
        "battery",
        "drain",
        "charging",
        "charge",
        "update",
        "ios",
        "iphone",
        "ipad",
        "mac",
        "app",
        "apps",
        "crash",
        "freeze",
        "freezing",
        "slow",
        "keyboard",
        "autocorrect",
        "typing",
        "password",
        "account",
        "icloud",
        "apple id",
        "login",
        "sign in",
        "network",
        "signal",
        "service",
        "call",
        "calling",
        "music",
        "itunes",
        "purchase",
        "charged",
        "billing",
        "screen",
        "display",
        "touch",
        "volume",
        "siri",
        "accessibility",
        "assistivetouch",
        "photos",
        "camera",
        "notification",
        "alert"
    ]

    return any(
        signal in text
        for signal in issue_signals
    )


# ==========================================
# GENERIC RESPONSE CHECK
# ==========================================

def is_generic_reply(reply):

    text = reply.lower().strip()

    generic_patterns = [
        "thanks for reaching out",
        "we're here to help",
        "we are here to help",
        "dm us",
        "send us a dm",
        "direct message us",
        "let's look into this",
        "we'd like to help",
        "we are happy to help",
        "we'd be happy to help",
        "we would love to help",
        "received your dm",
        "we'll follow up with you there"
    ]

    generic_count = sum(
        1
        for pattern in generic_patterns
        if pattern in text
    )

    if len(text.split()) < 12:
        return True

    if generic_count >= 2:
        return True

    if "dm us" in text and len(text.split()) < 30:
        return True

    if "received your dm" in text:
        return True

    return False


# ==========================================
# ACTIONABLE GUIDANCE
# ==========================================

def has_actionable_guidance(reply):

    text = reply.lower()

    action_words = [
        "follow",
        "steps",
        "settings",
        "update",
        "restart",
        "check",
        "turn on",
        "turn off",
        "install",
        "remove",
        "contact",
        "reset",
        "try",
        "go to",
        "tap",
        "click",
        "back up",
        "backup",
        "workaround",
        "article",
        "instructions"
    ]

    return any(
        word in text
        for word in action_words
    )


# ==========================================
# SELECT BEST USABLE EVIDENCE
# ==========================================

def select_best_evidence(
    predicted_intent,
    results
):

    candidates = []

    for result in results:

        same_intent = (
            result["historical_intent"]
            == predicted_intent
        )

        generic = is_generic_reply(
            result["brand_reply"]
        )

        actionable = has_actionable_guidance(
            result["brand_reply"]
        )

        if same_intent and not generic and actionable:

            candidates.append(result)

    if not candidates:
        return None

    # Prefer similarity while requiring useful evidence
    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return candidates[0]


# ==========================================
# DRAFT REPLY
# ==========================================

def draft_reply(
    message,
    predicted_intent,
    results
):

    best_evidence = select_best_evidence(
        predicted_intent,
        results
    )

    if best_evidence is not None:

        return best_evidence["brand_reply"].strip()

    return (
        "Thanks for reaching out. "
        "We found similar historical cases, but the available "
        "resolution does not provide enough reliable guidance "
        "to safely handle this automatically. "
        "We'll escalate this to a support specialist."
    )


# ==========================================
# DECISION LOGIC
# ==========================================

def decide_action(
    message,
    predicted_intent,
    results
):

    # --------------------------------------
    # 1. Specificity check
    # --------------------------------------

    if not is_specific_message(message):

        return (
            "ESCALATE",
            "Customer message is too vague to safely "
            "determine the issue.",
            None
        )

    # --------------------------------------
    # 2. Find usable evidence
    # --------------------------------------

    best_evidence = select_best_evidence(
        predicted_intent,
        results
    )

    # --------------------------------------
    # 3. No usable evidence
    # --------------------------------------

    if best_evidence is None:

        return (
            "ESCALATE",
            "No sufficiently similar historical case with "
            "matching intent and actionable guidance was found.",
            None
        )

    # --------------------------------------
    # 4. Similarity threshold
    # --------------------------------------

    if best_evidence["similarity"] < SIMILARITY_THRESHOLD:

        return (
            "ESCALATE",
            "Usable historical evidence exists, but similarity "
            "is below the safety threshold.",
            best_evidence
        )

    # --------------------------------------
    # 5. Strong evidence
    # --------------------------------------

    return (
        "AUTO-HANDLE",
        "Customer message is specific and a sufficiently "
        "similar historical case has matching intent and "
        "actionable guidance.",
        best_evidence
    )


# ==========================================
# RUN ONE MESSAGE
# ==========================================

def run_agent(
    message,
    exclude_message=None
):

    message = str(message).strip()

    # Predict intent
    predicted_intent = intent_model.predict(
        [message]
    )[0]

    # Retrieve more candidates
    results = retrieve_cases(
        message,
        top_k=5,
        exclude_message=exclude_message
    )

    # Decide
    action, reason, selected_evidence = decide_action(
        message,
        predicted_intent,
        results
    )

    # Draft
    draft = draft_reply(
        message,
        predicted_intent,
        results
    )

    # --------------------------------------
    # Similarity for reporting
    # --------------------------------------

    if selected_evidence is not None:

        similarity = selected_evidence["similarity"]
        historical_intent = (
            selected_evidence["historical_intent"]
        )

    elif results:

        similarity = results[0]["similarity"]
        historical_intent = (
            results[0]["historical_intent"]
        )

    else:

        similarity = 0.0
        historical_intent = None

    return {
        "intent": predicted_intent,
        "action": action,
        "reason": reason,
        "similarity": similarity,
        "historical_intent": historical_intent,
        "draft_reply": draft,
        "evidence": selected_evidence,
        "all_results": results
    }


# ==========================================
# INTERACTIVE TEST
# ==========================================

def interactive_test():

    print("\n========================================")
    print("       INTERACTIVE AGENT TEST")
    print("========================================")

    message = input(
        "\nEnter customer message:\n> "
    )

    result = run_agent(message)

    print("\n========================================")
    print("              RESULT")
    print("========================================")

    print("\nCustomer message:")
    print(message)

    print("\nPredicted intent:")
    print(result["intent"])

    print("\nHistorical evidence intent:")
    print(result["historical_intent"])

    print("\nDecision:")
    print(result["action"])

    print("\nReason:")
    print(result["reason"])

    print(
        "\nSelected evidence similarity:",
        f"{result['similarity']:.3f}"
    )

    print("\n----------------------------------------")
    print("DRAFT REPLY")
    print("----------------------------------------")

    print(result["draft_reply"])

    if result["evidence"]:

        print("\n----------------------------------------")
        print("SELECTED HISTORICAL EVIDENCE")
        print("----------------------------------------")

        print("\nHistorical customer:")
        print(
            result["evidence"]["customer_message"]
        )

        print("\nHistorical AppleSupport reply:")
        print(
            result["evidence"]["brand_reply"]
        )

        print(
            "\nSimilarity:",
            f"{result['evidence']['similarity']:.3f}"
        )

        print(
            "\nHistorical predicted intent:",
            result["evidence"]["historical_intent"]
        )

    print("\n----------------------------------------")
    print("TOP RETRIEVALS")
    print("----------------------------------------")

    for i, result_item in enumerate(
        result["all_results"],
        start=1
    ):

        print(
            f"\n{i}. Similarity: "
            f"{result_item['similarity']:.3f}"
        )

        print(
            "Customer:",
            result_item["customer_message"]
        )

        print(
            "Intent:",
            result_item["historical_intent"]
        )

    print("\n========================================")


# ==========================================
# GOLDEN SET EVALUATION
# ==========================================

def evaluate_golden_set():

    print("\n========================================")
    print("    LEAKAGE-FREE GOLDEN EVALUATION")
    print("========================================")

    df = pd.read_csv(GOLDEN_PATH)

    df = df[
        df["customer_message"].notna()
        & df["clean_label"].notna()
    ].copy()

    df["customer_message"] = (
        df["customer_message"]
        .astype(str)
        .str.strip()
    )

    print("\nGolden examples:", len(df))

    predicted_intents = []
    decisions = []
    similarities = []
    historical_evidence_intents = []
    reasons = []
    draft_replies = []

    for message in df["customer_message"]:

        result = run_agent(
            message,
            exclude_message=message
        )

        predicted_intents.append(
            result["intent"]
        )

        decisions.append(
            result["action"]
        )

        similarities.append(
            result["similarity"]
        )

        historical_evidence_intents.append(
            result["historical_intent"]
        )

        reasons.append(
            result["reason"]
        )

        draft_replies.append(
            result["draft_reply"]
        )

    df["predicted_intent"] = predicted_intents

    df["historical_evidence_intent"] = (
        historical_evidence_intents
    )

    df["agent_decision"] = decisions

    df["top_similarity"] = similarities

    df["decision_reason"] = reasons

    df["draft_reply"] = draft_replies

    # --------------------------------------
    # METRICS
    # --------------------------------------

    intent_accuracy = accuracy_score(
        df["clean_label"],
        df["predicted_intent"]
    )

    intent_agreement = (
        df["predicted_intent"]
        ==
        df["historical_evidence_intent"]
    ).mean()

    auto_handle_count = (
        df["agent_decision"]
        ==
        "AUTO-HANDLE"
    ).sum()

    escalate_count = (
        df["agent_decision"]
        ==
        "ESCALATE"
    ).sum()

    auto_handle_rate = (
        auto_handle_count / len(df)
    )

    escalation_rate = (
        escalate_count / len(df)
    )

    average_similarity = (
        df["top_similarity"].mean()
    )

    # --------------------------------------
    # RESULTS
    # --------------------------------------

    print("\n========================================")
    print("         FINAL AGENT RESULTS")
    print("========================================")

    print(
        f"\nIntent accuracy: "
        f"{intent_accuracy * 100:.2f}%"
    )

    print(
        f"Historical intent agreement: "
        f"{intent_agreement * 100:.2f}%"
    )

    print(
        f"Auto-handle: "
        f"{auto_handle_count}"
    )

    print(
        f"Escalate: "
        f"{escalate_count}"
    )

    print(
        f"Auto-handle rate: "
        f"{auto_handle_rate * 100:.2f}%"
    )

    print(
        f"Escalation rate: "
        f"{escalation_rate * 100:.2f}%"
    )

    print(
        f"Average selected similarity: "
        f"{average_similarity:.3f}"
    )

    print("\nDecision distribution:")

    print(
        df["agent_decision"].value_counts()
    )

    print("\nEscalation reasons:")

    print(
        df.loc[
            df["agent_decision"] == "ESCALATE",
            "decision_reason"
        ].value_counts()
    )

    # --------------------------------------
    # SAVE
    # --------------------------------------

    output_path = (
        "data/golden_predictions.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved evaluation results to:",
        output_path
    )

    print("\n========================================")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\nChoose mode:")

    print("1 - Test one customer message")
    print("2 - Evaluate golden set")

    choice = input(
        "\nEnter 1 or 2:\n> "
    ).strip()

    if choice == "1":

        interactive_test()

    elif choice == "2":

        evaluate_golden_set()

    else:

        print("\nInvalid choice.")