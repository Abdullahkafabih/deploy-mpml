import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load('final_menu_profitability_pipeline.pkl')
le = joblib.load('profitability_label_encoder.pkl')

# Set page config with your name
st.set_page_config(
    page_title="Menu Profitability Predictor by Kafabih",
    layout="wide"
)

# App title
st.title("Menu Profitability Predictor by Kafabih")

# Input form
with st.form("prediction_form"):
    st.header("Enter Menu Item Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        restaurant_id = st.selectbox(
            "Restaurant ID",
            options=["R001", "R002", "R003"]
        )
        menu_category = st.selectbox(
            "Menu Category",
            options=["Appetizers", "Beverages", "Desserts", "Main Course"]
        )
        price = st.number_input(
            "Price ($)",
            min_value=0.0,
            step=0.01,
            format="%.2f"
        )
    
    with col2:
        ingredient_count = st.number_input(
            "Number of Ingredients",
            min_value=0,
            step=1
        )
        menu_item_length = st.number_input(
            "Menu Item Name Length",
            min_value=0,
            step=1
        )
    
    submitted = st.form_submit_button("Predict Profitability")

# Make prediction when form is submitted
if submitted:
    # Prepare input data
    input_data = {
        'RestaurantID': restaurant_id,
        'MenuCategory': menu_category,
        'Price': price,
        'IngredientCount': ingredient_count,
        'MenuItemLength': menu_item_length
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[0]
    
    # Display results
    st.success("Prediction completed successfully!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Result")
        st.metric(
            label="Predicted Profitability",
            value=le.inverse_transform(prediction)[0]
        )
    
    with col2:
        st.subheader("Confidence Scores")
        st.write(f"Low: {probabilities[0]*100:.2f}%")
        st.write(f"Medium: {probabilities[1]*100:.2f}%")
        st.write(f"High: {probabilities[2]*100:.2f}%")
    
    # Show probability chart
    st.subheader("Probability Distribution")
    prob_df = pd.DataFrame({
        'Profitability': le.classes_,
        'Probability': probabilities
    })
    st.bar_chart(prob_df.set_index('Profitability'))