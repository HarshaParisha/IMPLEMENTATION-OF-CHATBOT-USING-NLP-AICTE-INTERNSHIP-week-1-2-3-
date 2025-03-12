import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup SSL for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

# Load intents file
file_path = os.path.abspath("intents.json")
with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Initialize TfidfVectorizer and Logistic Regression
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Prepare data for training
tags = []
patterns = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    # Transform the input text and predict intent tag
    input_text_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_text_vector)[0]

    # Find the response associated with the predicted tag
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return response

    # Fallback response if no tag is found
    return "I'm not sure how to respond to that. Can you please rephrase?"

# Initialize a global counter for user input fields
counter = 0

# Main function to handle Streamlit interface
def main():
    global counter

    # Streamlit app title
    st.title("Chatbot Using NLP")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the Chatbot. Please ask your queries below and press Enter!")

        # Create a log file if it doesn't already exist
        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

        # Increment counter for unique input fields
        counter += 1

        # User input field
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)

            # Display chatbot response
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Log the conversation with a timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            # Exit condition when user says goodbye
            if response.lower() in ["goodbye", "bye", "take care"]:
                st.write("Thank you for chatting with me! Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")

        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.subheader("About This Project")
        st.write(
            """
            This project showcases the implementation of a chatbot using Python and NLP techniques.
            It utilizes Logistic Regression for intent classification and Streamlit for the user interface.
            """
        )

        st.subheader("Project Overview")
        st.write(
            """
            1. The chatbot is trained on labeled intents using NLP techniques and Logistic Regression.
            2. An interactive web interface is built using Streamlit for seamless communication.
            """
        )

        st.subheader("How It Works")
        st.write(
            """
            - The chatbot takes user input and predicts the intent using a trained Logistic Regression model.
            - Based on the predicted intent, it fetches a relevant response from the intents dataset.
            - All conversations are logged in a CSV file for reference.
            """
        )

        st.subheader("Conclusion")
        st.write(
            """
            This chatbot demonstrates the integration of NLP and machine learning for intent recognition.
            Future improvements can include adding more intents, enhancing the dataset, and deploying the chatbot on the cloud for public use.
            """
        )

# Entry point for the Streamlit app
if __name__ == "__main__":
    main()
