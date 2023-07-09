import random
import streamlit as st
from streamlit_chat import message
from app_backend import get_response


def process_input(input):
    response = get_response(input)
    return response


st.header("HR ChatBot")
st.markdown("Ask your HR-related question here.")

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "input_message_key" not in st.session_state:
    st.session_state["input_message_key"] = str(random.random())

chat_container = st.container()
user_input = st.text_input("Type your message and press Enter to Send", key=st.session_state["input_message_key"])

if st.button("Send"):
    response = process_input(user_input)

    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(response)
    st.session_state["input_message_key"] = str(random.random())

    st.experimental_rerun()

if st.session_state["generated"]:
    with chat_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))