
import streamlit as st
import pandas as pd
import joblib
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Phishing Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

try:

    model = joblib.load("phishing_model.pkl")

    vectorizer = joblib.load("tfidf_vectorizer.pkl")

except FileNotFoundError:

    st.error(
        """
        Model files were not found.

        Please make sure these files exist in the same
        folder as app.py:

        1. phishing_model.pkl
        2. tfidf_vectorizer.pkl
        """
    )

    st.stop()


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

try:

    background_text = [
        "hello team meeting tomorrow",
        "please find the project report attached",
        "thank you for your email",
        "we will discuss the project next week",
        "your account has been suspended",
        "urgent verify your account immediately",
        "click here to confirm your password",
        "your bank account requires verification"
    ]

    background_vectors = vectorizer.transform(
        background_text
    )

    explainer = shap.LinearExplainer(
        model,
        background_vectors
    )

except Exception as e:

    explainer = None

    shap_error = str(e)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ Explainable AI for Phishing Email Detection"
)

st.write(
    """
    This system uses **TF-IDF + Logistic Regression + SHAP**
    to detect phishing emails and explain the important
    features behind the prediction.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🛡️ Project Information")

    st.write(
        """
        **Machine Learning Model**

        Logistic Regression

        **Feature Extraction**

        TF-IDF

        **Explainable AI**

        SHAP
        """
    )

    st.divider()

    st.subheader("Datasets Used")

    st.write(
        """
        • CEAS_08

        • Enron

        • Ling

        • Nazario

        • Nigerian Fraud

        • Phishing Email

        • SpamAssassin
        """
    )

    st.divider()

    st.subheader("SHAP Meaning")

    st.write(
        """
        🔴 Positive SHAP value

        Pushes the prediction toward phishing.


        🟢 Negative SHAP value

        Pushes the prediction toward legitimate.


        Larger absolute value = stronger influence.
        """
    )


# ============================================================
# EMAIL INPUT
# ============================================================

st.subheader("📧 Enter Email Content")

email = st.text_area(
    "Paste the email or message below:",
    height=250,
    placeholder=(
        "Example:\n\n"
        "URGENT! Your bank account has been suspended.\n\n"
        "Click here immediately to verify your password.\n\n"
        "Failure to verify your account within 24 hours "
        "will result in permanent suspension."
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Email",
    type="primary",
    use_container_width=True
):

    # ========================================================
    # CHECK INPUT
    # ========================================================

    if not email.strip():

        st.warning(
            "⚠️ Please enter an email before analyzing."
        )

        st.stop()


    # ========================================================
    # CONVERT EMAIL TO TF-IDF
    # ========================================================

    try:

        email_vector = vectorizer.transform(
            [email]
        )

    except Exception as e:

        st.error(
            "Error while converting the email to TF-IDF."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    try:

        prediction = model.predict(
            email_vector
        )[0]

        probabilities = model.predict_proba(
            email_vector
        )[0]

    except Exception as e:

        st.error(
            "Error while making the prediction."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # PROBABILITIES
    # ========================================================

    legitimate_probability = (
        probabilities[0] * 100
    )

    phishing_probability = (
        probabilities[1] * 100
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("🎯 Prediction")


    if prediction == 1:

        st.error(
            "🚨 PHISHING EMAIL DETECTED"
        )

    else:

        st.success(
            "✅ LEGITIMATE EMAIL"
        )


    # ========================================================
    # PROBABILITY METRICS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "🟢 Legitimate Probability",
            f"{legitimate_probability:.2f}%"
        )


    with col2:

        st.metric(
            "🔴 Phishing Probability",
            f"{phishing_probability:.2f}%"
        )


    # ========================================================
    # CONFIDENCE BAR
    # ========================================================

    st.subheader("📊 Prediction Confidence")


    if prediction == 1:

        st.progress(
            phishing_probability / 100
        )

        st.write(
            f"Phishing confidence: "
            f"**{phishing_probability:.2f}%**"
        )

    else:

        st.progress(
            legitimate_probability / 100
        )

        st.write(
            f"Legitimate confidence: "
            f"**{legitimate_probability:.2f}%**"
        )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 Why did the model make this prediction?"
    )


    # ========================================================
    # CHECK SHAP EXPLAINER
    # ========================================================

    if explainer is None:

        st.warning(
            "SHAP explanation could not be initialized."
        )

        st.code(
            shap_error
        )

    else:

        try:

            # ------------------------------------------------
            # Calculate SHAP values
            # ------------------------------------------------

            shap_result = explainer(
                email_vector
            )


            # ------------------------------------------------
            # Get feature names
            # ------------------------------------------------

            feature_names = (
                vectorizer
                .get_feature_names_out()
            )


            # ------------------------------------------------
            # Get SHAP values
            # ------------------------------------------------

            shap_values = shap_result.values


            # ------------------------------------------------
            # Handle different SHAP versions
            # ------------------------------------------------

            if len(shap_values.shape) == 3:

                # Shape:
                # samples × features × classes

                shap_values = shap_values[0, :, 0]

            elif len(shap_values.shape) == 2:

                # Shape:
                # samples × features

                shap_values = shap_values[0]

            else:

                shap_values = shap_values.flatten()


            # ------------------------------------------------
            # Find features present in email
            # ------------------------------------------------

            feature_indices = (
                email_vector.nonzero()[1]
            )


            # ------------------------------------------------
            # Create explanation data
            # ------------------------------------------------

            explanation_data = []


            for index in feature_indices:

                explanation_data.append(
                    {
                        "Feature":
                            feature_names[index],

                        "SHAP Value":
                            float(
                                shap_values[index]
                            )
                    }
                )


            # ------------------------------------------------
            # Convert to DataFrame
            # ------------------------------------------------

            explanation_df = pd.DataFrame(
                explanation_data
            )


            # =================================================
            # CHECK FEATURES
            # =================================================

            if explanation_df.empty:

                st.info(
                    """
                    No recognized features from the trained
                    TF-IDF vocabulary were found in this email.
                    """
                )

            else:

                # ---------------------------------------------
                # Calculate absolute importance
                # ---------------------------------------------

                explanation_df[
                    "Importance"
                ] = (
                    explanation_df[
                        "SHAP Value"
                    ].abs()
                )


                # ---------------------------------------------
                # Sort by importance
                # ---------------------------------------------

                explanation_df = (
                    explanation_df
                    .sort_values(
                        "Importance",
                        ascending=False
                    )
                    .head(15)
                )


                # ---------------------------------------------
                # Determine direction
                # ---------------------------------------------

                explanation_df[
                    "Direction"
                ] = explanation_df[
                    "SHAP Value"
                ].apply(
                    lambda value:
                    "🔴 Phishing"
                    if value > 0
                    else "🟢 Legitimate"
                )


                # =================================================
                # EXPLANATION TABLE
                # =================================================

                st.write(
                    """
                    The following features had the strongest
                    influence on the model's decision:
                    """
                )


                display_df = explanation_df[
                    [
                        "Feature",
                        "SHAP Value",
                        "Direction"
                    ]
                ].copy()


                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )


                # =================================================
                # TOP FEATURES
                # =================================================

                st.subheader(
                    "🔎 Top Contributing Features"
                )


                top_features = (
                    explanation_df.head(5)
                )


                for _, row in top_features.iterrows():

                    feature = row["Feature"]

                    value = row["SHAP Value"]


                    if value > 0:

                        st.write(
                            f"🔴 **{feature}** "
                            f"→ pushes toward phishing "
                            f"(**{value:+.4f}**)"
                        )

                    else:

                        st.write(
                            f"🟢 **{feature}** "
                            f"→ pushes toward legitimate "
                            f"(**{value:+.4f}**)"
                        )


                # =================================================
                # INTERPRETATION
                # =================================================

                st.info(
                    """
                    **How to interpret SHAP values**

                    🔴 Positive SHAP value:
                    the feature pushes the prediction toward
                    phishing.

                    🟢 Negative SHAP value:
                    the feature pushes the prediction toward
                    legitimate.

                    The larger the absolute SHAP value, the
                    stronger the feature's contribution.

                    **Note:** SHAP values are feature
                    contributions, not probabilities.
                    """
                )


        except Exception as e:

            st.error(
                "⚠️ SHAP explanation failed."
            )

            st.write(
                "Technical error:"
            )

            st.code(
                str(e)
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Explainable AI for Phishing Email Detection | "
    "TF-IDF + Logistic Regression + SHAP"
)
