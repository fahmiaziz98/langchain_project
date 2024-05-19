__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import streamlit as st
from datetime import datetime

from src.embedding import embedding_store
from src.utils import save_uploadedfile
from src.agent import run_agent


st.set_page_config(page_title="RAG Agent")
st.markdown(
    "<h1 style='text-align: center; color: #007BFF;'>RAG Agent</h1>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        "<h2 style='text-align: center; color: #007BFF;'>Upload PDF</h2>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    with st.expander("Contact Information"):
        st.markdown("For any queries, please feel free to contact:")
        st.markdown(
            "Email: [fahmiazizfadhil999@gmail.com](mailto:fahmiazizfadhil999@gmail.com)"
        )
        st.markdown("GitHub: [github.com/fahmiaziz98](https://github.com/fahmiaziz98)")

    with st.expander("Additional Information"):
        st.info("Get Your API key at https://dashboard.cohere.com/api-keys")
        st.markdown(
            "<h4 style='text-align: center;'>Powered by Cohere</h4>",
            unsafe_allow_html=True,
        )
    

if uploaded_file is not None:
    save_uploadedfile(uploaded_file)
    file_size = os.path.getsize(f"tempfolder/{uploaded_file.name}") / (1024 * 1024)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Uploaded PDF: {file_size} MB")
    vector = embedding_store("tempfolder/" + uploaded_file.name)
    st.markdown(
        "<h3 style='text-align: center;'>Now You Are Chatting With "
        + uploaded_file.name
        + "</h3>",
        unsafe_allow_html=True,
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    if uploaded_file is not None:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            while not full_response:
                with st.spinner("Thinking..."):
                    Output = run_agent(prompt, vector)
                    full_response = Output if Output else "Failed to get the response."
                fr = ""
                full_response = str(full_response)
                for i in full_response:
                    import time

                    time.sleep(0.02)
                    fr += i
                    message_placeholder.write(fr + "▌")
                message_placeholder.write(f"{full_response}")
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.write(
                "Please go ahead and upload the PDF in the sidebar, it would be great to have it there and make sure API key Entered"
            )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Please go ahead and upload the PDF in the sidebar, it would be great to have it there and make sure API key Entered",
            }
        )
