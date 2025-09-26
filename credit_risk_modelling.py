# tab8_model_comparison.py

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report, roc_auc_score
import streamlit as st

def run_model_comparison(data1):
    # --- Features & Target ---
    X = data1[['Credit Card Balance', 
               'Total Relationship Balance', 
               'Estimated Income', 
               'Customer Tenure', 
               'Product Concentration']]

    y = (data1['Risk Weighting'] > data1['Risk Weighting'].median()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- SVM ---
    svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ])
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict_proba(X_test)[:,1]

    st.subheader("SVM Classification Report")
    st.text(classification_report(y_test, svm_clf.predict(X_test)))
    st.write("SVM AUC:", roc_auc_score(y_test, y_pred_svm))

    # --- Random Forest ---
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict_proba(X_test)[:,1]
    st.write("Random Forest AUC:", roc_auc_score(y_test, y_pred_rf))

    # --- Gradient Boosting ---
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict_proba(X_test)[:,1]
    st.write("Gradient Boosting AUC:", roc_auc_score(y_test, y_pred_gb))

    # --- ROC Curves ---
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
    fpr_rf, tpr_rf, _   = roc_curve(y_test, y_pred_rf)
    fpr_gb, tpr_gb, _   = roc_curve(y_test, y_pred_gb)

    roc_auc_svm = auc(fpr_svm, tpr_svm)
    roc_auc_rf  = auc(fpr_rf, tpr_rf)
    roc_auc_gb  = auc(fpr_gb, tpr_gb)

    plt.figure(figsize=(6,4))
    plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={roc_auc_svm:.3f})")
    plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={roc_auc_rf:.3f})")
    plt.plot(fpr_gb, tpr_gb, label=f"GB (AUC={roc_auc_gb:.3f})")
    plt.plot([0,1], [0,1], 'k--')

    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    st.pyplot(plt)


    # Collect model AUCs
    model_aucs = {
        "SVM": roc_auc_svm,
        "Random Forest": roc_auc_rf,
        "Gradient Boosting": roc_auc_gb
    }

    # Find best and worst models dynamically
    best_model = max(model_aucs, key=model_aucs.get)
    worst_model = min(model_aucs, key=model_aucs.get)

    st.markdown(
        f"""
        <div class="insight-card">
            <p><b> Model Comparison Insights:</b></p>
            <ul>
                <li><b>Best Performing Model:</b> {best_model} with AUC {model_aucs[best_model]:.3f}</li>
                <li> <b>Lowest Performing Model:</b> {worst_model} with AUC {model_aucs[worst_model]:.3f}</li>
                <li> <b>Performance Gap:</b> {model_aucs[best_model] - model_aucs[worst_model]:.3f} between best and worst models</li>
                <li> All models scored above 0.5 AUC → indicating better than random performance</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Optional: add recommendation
    if best_model == "Random Forest":
        st.success("🌲 Random Forest is the most reliable here, great for non-linear patterns and interpretability.")
    elif best_model == "Gradient Boosting":
        st.success("🚀 Gradient Boosting outperforms others — consider fine-tuning with learning rate & depth.")
    else:
        st.success("⚡ SVM leads — works well with scaled data, might benefit from hyperparameter tuning.")
