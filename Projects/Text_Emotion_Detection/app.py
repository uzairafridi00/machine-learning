# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# loading the trained model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# define the emoji dictionary
emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

# define functions
def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# main function
def main():
    st.title("AI-Powered Emotion Detection 💡")
    st.subheader("Understanding Feelings Through Text")

    with st.form(key="my_form"):
        raw_text = st.text_area("Type Your Text Here")
        submit_text = st.form_submit_button(label="Predict Emotion")

    if submit_text:
        col1, col2 = st.columns(2)
        
        prediction = predict_emotion(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            #st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()