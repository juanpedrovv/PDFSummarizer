import subprocess
import utils.summarize as summarize
#pdf_file_path = "sample2.pdf"
#print(summarize.summarize_pdf(pdf_file_path))

if __name__ == "__main__":
    command = "streamlit run app.py"
    subprocess.run(command, shell=True)
