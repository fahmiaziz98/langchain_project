__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import streamlit as st

from src.llm import generate_text
from src.vector import vector_database
from src.utils import save_uploadedfile

# Set the page title
st.set_page_config(page_title="Marketing Tools")

# Page header
st.markdown(
    "<h1 style='text-align: center; color: #007BFF;'>Generate Post Marketing for Your Product</h1>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        "<h2 style='text-align: center; color: #007BFF;'>Upload Data</h2>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    col1_name = st.text_input("Product Name")
    col2_name = st.text_input("Description")
    apply_button = st.button("Apply")

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

# Process and display data if Apply button is clicked
if uploaded_file and col1_name and col2_name:
    save_uploadedfile(uploaded_file)
    db = vector_database("tempfolder/" + uploaded_file.name, col1_name, col2_name)

    with st.form('myform'):
        text = st.text_input('Enter Product Name:', '')
        submitted = st.form_submit_button('Submit')
        if submitted:
            res = generate_text(query=text, db=db)
            st.write(res)
else:
    if uploaded_file is None:
        st.info("Please upload a CSV file.")
    elif not (col1_name and col2_name):
        st.info("Please enter both column names.")





