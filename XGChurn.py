import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap


XGB_model = pickle.load(open('XGB_model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))


FEATURE_MAPPING = {
    "StandardScaler__Balance": "Balance",
    "StandardScaler__EstimatedSalary": "Estimated Salary",
    "StandardScaler__NumOfProducts": "Number of Products",
    "StandardScaler__bal_ratio_sal": "Balance to Salary Ratio",
    "OneHotEncoder__Geography_France": "Geography: France",
    "OneHotEncoder__Geography_Germany": "Geography: Germany",
    "OneHotEncoder__Geography_Spain": "Geography: Spain",
    "OneHotEncoder__Gender_Male": "Gender: Male",
    "OneHotEncoder__Gender_Female": "Gender: Female",
    "OneHotEncoder__HasCrCard_-1": "No Credit Card",
    "OneHotEncoder__HasCrCard_1": "Has Credit Card",
    "OneHotEncoder__IsActiveMember_1": "Active Member",
    "OneHotEncoder__IsActiveMember_-1": "Inactive Member",
    "ordinal__Age_Binned": "Age Group"
}

def preprocess_data(df):
    """
    Preprocess the data by dropping unnecessary columns, performing feature engineering,
    and transforming the data using the preprocessor.
    """
    df.drop(columns=['CustomerId', 'Surname', 'CreditScore', 'Tenure'], inplace=True, errors='ignore')
    df['Age_Binned'] = pd.cut(df['Age'], bins=[0, 20, 40, 60, float('inf')], labels=['0-20', '20-40', '40-60', '60+'])
    df['bal_ratio_sal'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['HasCrCard'] = df['HasCrCard'].replace({0: -1})
    df['IsActiveMember'] = df['IsActiveMember'].replace({0: -1})
    df.drop(columns=['Age'], inplace=True, errors='ignore')
    processed_data = preprocessor.transform(df)
    return processed_data

def get_user_input():
    """
    Gather user input for all features in the dataset via Streamlit widgets.
    """
    st.sidebar.header("Enter Customer Details")
    customer_id = st.sidebar.text_input("Customer ID", "12345")
    surname = st.sidebar.text_input("Surname", "Doe")
    geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 18, 100, 35)
    balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 75000.0)
    num_of_products = st.sidebar.slider("Number of Products", 1, 4, 2)
    has_cr_card = st.sidebar.selectbox("Has Credit Card", [1, 0])
    is_active_member = st.sidebar.selectbox("Is Active Member", [1, 0])
    estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

    user_data = {
        'CustomerId': [customer_id],
        'Surname': [surname],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
    }
    return pd.DataFrame(user_data)

def plot_churn_reasons(processed_data, shap_values, feature_names):
    """
    Plot the top reasons for churn using SHAP values as a vertical bar chart.
    """
    total_shap_values = np.abs(shap_values).sum(axis=0)
    total_impact = total_shap_values.sum()
    shap_percentages = {
        FEATURE_MAPPING.get(feature_names[i], feature_names[i]): (total_shap_values[i] / total_impact) * 100
        for i in range(len(feature_names))
    }

    top_features = sorted(shap_percentages.items(), key=lambda x: x[1], reverse=True)[:3]

    reasons_df = pd.DataFrame(top_features, columns=["Feature", "Percentage"])

    fig, ax = plt.subplots()
    ax.bar(reasons_df["Feature"], reasons_df["Percentage"], color="grey")
    ax.set_ylabel("Impact Percentage")
    ax.set_title("Top Reasons for Churn")
    ax.set_xticklabels(reasons_df["Feature"], rotation=45, ha="right")
    st.pyplot(fig)

def main():
    st.title("Customer Churn Prediction App")
    st.sidebar.header("Choose Input Method")
    option = st.sidebar.radio("Input Method", ("Enter Manually", "Upload CSV"))

    if option == "Enter Manually":
        user_input = get_user_input()
        processed_input = preprocess_data(user_input)


        churn_probability = XGB_model.predict_proba(processed_input)[:, 1][0]
        churn_prediction = XGB_model.predict(processed_input)[0]


        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h2 style='color: {'red' if churn_prediction == 1 else 'green'};'>Churn Probability: {churn_probability:.2%}</h2>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h2 style='color: {'red' if churn_prediction == 1 else 'green'};'>Churn Prediction: {'Churn' if churn_prediction == 1 else 'No Churn'}</h2>", unsafe_allow_html=True)


        fig, ax = plt.subplots()
        labels = ['No Churn', 'Churn']
        sizes = [1 - churn_probability, churn_probability]
        colors = ['#66b3ff', '#ff6666']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, explode=(0, 0.1), colors=colors, shadow=True)
        ax.set_title("Churn Probability")
        st.pyplot(fig)


        if churn_prediction == 1:
            st.subheader("Top Reasons for Churn")
            explainer = shap.TreeExplainer(XGB_model)
            shap_values = explainer.shap_values(processed_input)
            plot_churn_reasons(processed_input, shap_values, preprocessor.get_feature_names_out())

    elif option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            processed_data = preprocess_data(data)

            churn_probabilities = XGB_model.predict_proba(processed_data)[:, 1]
            churn_predictions = XGB_model.predict(processed_data)

            data['Churn Probability'] = churn_probabilities
            data['Churn/No Churn'] = churn_predictions

            st.subheader("Updated Dataset with Churn Predictions")
            st.write(data)

            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Updated Dataset", csv, "updated_dataset.csv", "text/csv")

            total_balance = data['Balance'].sum()
            balance_at_risk = data[data['Churn/No Churn'] == 1]['Balance'].sum()
            balance_at_risk_percentage = (balance_at_risk / total_balance) * 100

            st.subheader("Balance at Risk Analysis")
            st.markdown(f"<h2>Total Balance: ${total_balance:,.2f}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h2>Balance at Risk: ${balance_at_risk:,.2f} ({balance_at_risk_percentage:.2f}%)</h2>", unsafe_allow_html=True)

            fig, ax = plt.subplots()
            ax.pie(
                [balance_at_risk, total_balance - balance_at_risk],
                labels=["At Risk", "Safe"],
                autopct='%1.1f%%',
                colors=['#ff6666', '#66b3ff'],
                startangle=140,
                shadow=True
            )
            ax.set_title("Balance at Risk")
            st.pyplot(fig)
            explainer = shap.TreeExplainer(XGB_model)
            shap_values = explainer.shap_values(processed_data)
            st.subheader("Reasons for Most Churns")
            plot_churn_reasons(processed_data, shap_values, preprocessor.get_feature_names_out())

if __name__ == "__main__":
    main()
