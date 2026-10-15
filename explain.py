import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. LOAD TRAINED MODEL
# =========================================================

model = joblib.load("phishing_model.pkl")

vectorizer = joblib.load("tfidf_vectorizer.pkl")


print("Model loaded successfully.")


# =========================================================
# 2. CREATE SHAP EXPLAINER
# =========================================================

# Logistic Regression is a linear model,
# so LinearExplainer is suitable.

explainer = shap.LinearExplainer(
    model,
    vectorizer.transform([
        "sample email used to initialize SHAP"
    ])
)


# =========================================================
# 3. FUNCTION TO EXPLAIN AN EMAIL
# =========================================================

def explain_email(email):

    print("\n")
    print("=" * 60)
    print("EMAIL ANALYSIS")
    print("=" * 60)

    print("\nEmail:")
    print(email)


    # -----------------------------------------------------
    # Convert email to TF-IDF
    # -----------------------------------------------------

    email_vector = vectorizer.transform([email])


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(email_vector)[0]

    probabilities = model.predict_proba(email_vector)[0]


    # -----------------------------------------------------
    # Display prediction
    # -----------------------------------------------------

    print("\nPrediction:")

    if prediction == 1:

        print("🚨 PHISHING")

    else:

        print("✅ LEGITIMATE")


    print(
        f"Legitimate probability: "
        f"{probabilities[0] * 100:.2f}%"
    )

    print(
        f"Phishing probability: "
        f"{probabilities[1] * 100:.2f}%"
    )


    # =====================================================
    # SHAP VALUES
    # =====================================================

    shap_values = explainer(email_vector)


    feature_names = vectorizer.get_feature_names_out()

    values = shap_values.values[0]


    # -----------------------------------------------------
    # Find features actually present in the email
    # -----------------------------------------------------

    present_features = email_vector.nonzero()[1]


    feature_data = []


    for index in present_features:

        feature_data.append({
            "Feature": feature_names[index],
            "SHAP Value": values[index]
        })


    explanation_df = pd.DataFrame(feature_data)


    if explanation_df.empty:

        print("\nNo recognizable features found.")

        return


    # -----------------------------------------------------
    # Sort by absolute importance
    # -----------------------------------------------------

    explanation_df["Importance"] = (
        explanation_df["SHAP Value"].abs()
    )


    explanation_df = explanation_df.sort_values(
        "Importance",
        ascending=False
    )


    # =====================================================
    # DISPLAY IMPORTANT FEATURES
    # =====================================================

    print("\n")
    print("=" * 60)
    print("WHY DID THE MODEL MAKE THIS DECISION?")
    print("=" * 60)


    print(
        "\nPositive SHAP value → pushes toward PHISHING"
    )

    print(
        "Negative SHAP value → pushes toward LEGITIMATE"
    )


    print("\nTop contributing features:\n")


    for _, row in explanation_df.head(15).iterrows():

        print(
            f"{row['Feature']:30}"
            f"{row['SHAP Value']:+.4f}"
        )


    # =====================================================
    # CREATE SHAP BAR CHART
    # =====================================================

    top_features = explanation_df.head(10)

    plt.figure(figsize=(10, 6))


    plt.barh(
        top_features["Feature"],
        top_features["SHAP Value"]
    )


    plt.xlabel("SHAP Value")

    plt.ylabel("Feature")

    plt.title(
        "Why did the model make this prediction?"
    )


    plt.gca().invert_yaxis()

    plt.tight_layout()


    plt.savefig(
        "shap_explanation.png",
        dpi=300
    )


    plt.show()


# =========================================================
# 4. TEST EMAIL
# =========================================================

test_email = """
URGENT! Your bank account has been suspended.

Click here immediately to verify your password.

Failure to verify your account within 24 hours
will result in permanent suspension.
"""


explain_email(test_email)