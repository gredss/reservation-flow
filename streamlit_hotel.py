import streamlit as st
import joblib
import numpy as np
import pandas as pd

try:
    model = joblib.load('XGB_trained_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def get_user_input():
    st.sidebar.subheader("Booking Information")

    no_of_adults = st.sidebar.number_input("Number of Adults", min_value=0, step=1)
    no_of_children = st.sidebar.number_input("Number of Children", min_value=0, step=1)
    no_of_weekend_nights = st.sidebar.number_input("Number of Weekend Nights", min_value=0, step=1)
    no_of_week_nights = st.sidebar.number_input("Number of Week Nights", min_value=0, step=1)

    required_car_parking_space = st.sidebar.radio("Car Parking Space Required?", ["No", "Yes"])
    required_car_parking_space = 1 if required_car_parking_space == "Yes" else 0

    lead_time = st.sidebar.number_input("Lead Time (days)", min_value=0, step=1)

    arrival_year = st.sidebar.number_input("Arrival Year", min_value=2000, max_value=2100, step=1)
    arrival_month = st.sidebar.selectbox("Arrival Month", list(range(1, 13)), format_func=lambda x: f"{x:02d}")

    def get_day_options(month):
        if month in [1, 3, 5, 7, 8, 10, 12]:
            return list(range(1, 32))
        elif month in [4, 6, 9, 11]:
            return list(range(1, 31))
        else:
            return list(range(1, 29))

    arrival_date = st.sidebar.selectbox("Arrival Day", get_day_options(arrival_month))

    repeated_guest = st.sidebar.radio("Repeated Guest?", ["No", "Yes"])
    repeated_guest = 1 if repeated_guest == "Yes" else 0

    no_of_previous_cancellations = st.sidebar.number_input("Previous Cancellations", min_value=0, step=1)
    no_of_previous_bookings_not_canceled = st.sidebar.number_input("Previous Bookings (Not Canceled)", min_value=0, step=1)
    avg_price_per_room = st.sidebar.number_input("Average Price per Room", min_value=0.0, step=1.0)
    no_of_special_requests = st.sidebar.number_input("Number of Special Requests", min_value=0, step=1)

    meal_plan = st.sidebar.selectbox(
        "Meal Plan",
        ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "No Need Meal"],
    )

    room_type = st.sidebar.selectbox(
        "Room Type Reserved",
        ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"],
    )

    market_segment = st.sidebar.selectbox(
        "Market Segment Type",
        ["Aviation", "Complementary", "Corporate", "Offline", "Online"],
    )

    return {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'required_car_parking_space': required_car_parking_space,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests,
        'meal_plan': meal_plan,
        'room_type': room_type,
        'market_segment': market_segment
    }

def preprocess_input(user_input):
    features = {
        'no_of_adults': user_input['no_of_adults'],
        'no_of_children': user_input['no_of_children'],
        'no_of_weekend_nights': user_input['no_of_weekend_nights'],
        'no_of_week_nights': user_input['no_of_week_nights'],
        'required_car_parking_space': user_input['required_car_parking_space'],
        'lead_time': user_input['lead_time'],
        'arrival_year': user_input['arrival_year'],
        'arrival_month': user_input['arrival_month'],
        'arrival_date': user_input['arrival_date'],
        'repeated_guest': user_input['repeated_guest'],
        'no_of_previous_cancellations': user_input['no_of_previous_cancellations'],
        'no_of_previous_bookings_not_canceled': user_input['no_of_previous_bookings_not_canceled'],
        'avg_price_per_room': user_input['avg_price_per_room'],
        'no_of_special_requests': user_input['no_of_special_requests'],
    }

    meal_options = [
        'type_of_meal_plan_Meal Plan 1',
        'type_of_meal_plan_Meal Plan 2',
        'type_of_meal_plan_Meal Plan 3',
        'type_of_meal_plan_Not Selected'
    ]
    for option in meal_options:
        features[option] = 1 if (
            option.endswith(user_input['meal_plan']) or
            (user_input['meal_plan'] == "No Need Meal" and option.endswith("Not Selected"))
        ) else 0

    room_options = [
        'room_type_reserved_Room_Type 1',
        'room_type_reserved_Room_Type 2',
        'room_type_reserved_Room_Type 3',
        'room_type_reserved_Room_Type 4',
        'room_type_reserved_Room_Type 5',
        'room_type_reserved_Room_Type 6',
        'room_type_reserved_Room_Type 7',
    ]
    for option in room_options:
        features[option] = 1 if option.endswith(user_input['room_type']) else 0

    market_options = [
        'market_segment_type_Aviation',
        'market_segment_type_Complementary',
        'market_segment_type_Corporate',
        'market_segment_type_Offline',
        'market_segment_type_Online'
    ]
    for option in market_options:
        features[option] = 1 if option.endswith(user_input['market_segment']) else 0

    return pd.DataFrame([features])

def main():
    st.title("Hotel Booking Cancellation Predictor")
    st.sidebar.title("Hotel Booking Cancellation Predictor")
    st.markdown("Will this hotel booking be canceled or not?")
    st.sidebar.markdown("Will this hotel booking be canceled or not?")

    user_input = get_user_input()
    input_df = preprocess_input(user_input)

    if st.button("Predict Booking Status"):
        prediction = model.predict(input_df)[0]
        result = "Canceled" if prediction == 1 else "Not Canceled"
        st.success(f"Prediction: {result}")

if __name__ == "__main__":
    main()