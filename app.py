import streamlit as st
import agent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Apple Support AI Agent",
    page_icon="🍎",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🍎 Apple Support AI Agent")

st.markdown(
    """
    **AI customer-support agent built from historical AppleSupport conversations.**

    The agent:
    - classifies the customer's issue
    - retrieves similar historical support cases
    - drafts a historically grounded reply
    - decides whether to **AUTO-HANDLE** or **ESCALATE**
    """
)


st.divider()


# ============================================================
# CUSTOMER INPUT
# ============================================================

st.subheader("Customer Message")

message = st.text_area(
    "Enter a customer support message:",
    placeholder=(
        "Example: My iPhone battery is draining very quickly "
        "after the latest iOS update."
    ),
    height=130
)


analyze = st.button(
    "Analyze Message",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYZE
# ============================================================

if analyze:

    if not message.strip():

        st.warning(
            "Please enter a customer message first."
        )

    else:

        with st.spinner(
            "Analyzing customer message..."
        ):

            result = agent.run_agent(
                message.strip()
            )


        st.success(
            "Analysis complete."
        )


        # ====================================================
        # TOP SUMMARY
        # ====================================================

        st.subheader("Agent Decision")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Predicted Intent",
                result["intent"]
            )


        with col2:

            if result["action"] == "AUTO-HANDLE":

                st.success(
                    "AUTO-HANDLE"
                )

            else:

                st.error(
                    "ESCALATE"
                )


        with col3:

            st.metric(
                "Similarity",
                f"{result['similarity']:.3f}"
            )


        # ====================================================
        # DECISION REASON
        # ====================================================

        st.subheader("Decision Reason")

        if result["action"] == "AUTO-HANDLE":

            st.success(
                result["reason"]
            )

        else:

            st.warning(
                result["reason"]
            )


        # ====================================================
        # DRAFT REPLY
        # ====================================================

        st.subheader("Draft Reply")

        st.text_area(
            "Proposed customer response:",
            value=result["draft_reply"],
            height=180,
            key="draft_reply"
        )


        # ====================================================
        # HISTORICAL EVIDENCE
        # ====================================================

        st.subheader(
            "Historical Evidence Used"
        )


        evidence = result["evidence"]


        if evidence:

            st.markdown(
                f"""
                **Historical Intent:**  
                `{evidence["historical_intent"]}`

                **Similarity:**  
                `{evidence["similarity"]:.3f}`
                """
            )


            st.markdown(
                "**Previous Customer Message**"
            )

            st.info(
                evidence["customer_message"]
            )


            st.markdown(
                "**Historical AppleSupport Reply**"
            )

            st.success(
                evidence["brand_reply"]
            )


        else:

            st.info(
                "No sufficiently useful historical evidence "
                "was selected."
            )


        # ====================================================
        # TOP RETRIEVALS
        # ====================================================

        with st.expander(
            "View Top Retrieved Historical Cases"
        ):

            results = result["all_results"]


            if results:

                for i, item in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        f"### Case {i}"
                    )


                    st.write(
                        f"**Similarity:** "
                        f"{item['similarity']:.3f}"
                    )


                    st.write(
                        f"**Historical intent:** "
                        f"{item['historical_intent']}"
                    )


                    st.markdown(
                        "**Customer:**"
                    )

                    st.write(
                        item["customer_message"]
                    )


                    st.markdown(
                        "**AppleSupport reply:**"
                    )

                    st.write(
                        item["brand_reply"]
                    )


                    if i < len(results):

                        st.divider()


        # ====================================================
        # SYSTEM INFORMATION
        # ====================================================

        with st.expander(
            "How the decision was made"
        ):

            st.markdown(
                f"""
                **1. Intent classification**

                Predicted intent:
                `{result["intent"]}`

                **2. Historical retrieval**

                The agent searched historical AppleSupport
                customer-support conversations.

                **3. Evidence filtering**

                Historical cases were filtered for:
                - matching intent
                - non-generic response
                - actionable guidance

                **4. Safety decision**

                Similarity threshold:
                `{agent.SIMILARITY_THRESHOLD:.2f}`

                **5. Final action**

                `{result["action"]}`

                **Reason**

                {result["reason"]}
                """
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "About the Agent"
    )

    st.write(
        """
        This prototype uses historical AppleSupport
        conversations to ground support responses.

        The system deliberately escalates when:
        - the customer message is vague
        - useful historical evidence is unavailable
        - similarity is below the safety threshold
        """
    )


    st.divider()


    st.caption(
        "Hiver SDE Intern Take-Home Assignment"
    )