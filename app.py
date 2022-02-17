import streamlit as st
import pickle

pickle_in = open('classifier.pkl', 'rb')
model = pickle.load(pickle_in)


@st.cache()
def prediction(Height, Weight):
    predictions = model.predict([[Height, Weight]])

    if predictions == 0:
        pred = "Female"
    else:
        pred = "Male"
    return pred


def main():
    st.write("""
     ### Gender Prediction from height and weight """)

    Height = st.number_input("Applicants height")
    Weight = st.number_input("Applicants Weight")
    result = " "

    if st.button("Predict"):
        result = prediction(Height, Weight)
        st.success('Your gender is {}'.format(result))


if __name__ == '__main__':
    main()
