import os
import streamlit as st


if not os.path.exists("./tempfolder"):
    os.makedirs("./tempfolder")

def save_uploadedfile(uploadedfile):
    with open(
        os.path.join("tempfolder", uploadedfile.name),
        "wb",
    ) as f:
        f.write(uploadedfile.getbuffer())
    return st.sidebar.success("Saved File")
