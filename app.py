import streamlit as st
import io
import base64
import utils.summarize as summarize

@st.cache_data
#function to display the PDF of a given file 
def displayPDF(pdf_bytes):
    # Encoding the BytesIO object to base64
    base64_pdf = base64.b64encode(pdf_bytes.getvalue()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<div style="display: flex; justify-content: center;"><iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf" style="margin: auto;"></iframe></div>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

def summarizePDF(pdf_file):
    return summarize.summarize_pdf(pdf_file)

st.set_page_config(layout="wide")
st.title("Summarization Application")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf is not None:
    col1, col2 = st.columns([1,1])

    # Convert the uploaded file to a BytesIO object
    pdf_bytes = io.BytesIO(uploaded_pdf.read())

    with col1:
        st.success("Saved File")
        # Display the PDF
        displayPDF(pdf_bytes)
    with col2:
        st.success("Your Summary")
        #PDF summarized
        st.markdown(summarizePDF(pdf_bytes))
