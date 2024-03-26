from langchain import PromptTemplate, hub
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from io import BytesIO
from langchain.document_loaders.parsers.pdf import PyPDFParser
from langchain_community.document_loaders.base import BaseLoader
from typing import Optional, List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.blob_loaders import Blob
from dotenv import load_dotenv

class CustomPDFLoader(BaseLoader):
    def __init__(self, stream: BytesIO, password: Optional[Union[str, bytes]] = None, extract_images: bool = False):
        self.stream = stream
        self.parser = PyPDFParser(password=password, extract_images=extract_images)

    def load(self) -> List[Document]:
        blob = Blob.from_data(self.stream.getvalue())
        return list(self.parser.parse(blob))

# Cargar las variables de entorno necesarias
load_dotenv()

def initialize_llm():
    return ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")

def load_and_split_pdf(pdf_file_path):
    loader = CustomPDFLoader(pdf_file_path)
    return loader.load()

def get_map_chain(llm):
    map_prompt = hub.pull("rlm/map-prompt")
    return LLMChain(llm=llm, prompt=map_prompt)

def get_reduce_chain(llm):
    reduce_template = """
    The following is a set of documents:
    {docs}
    Please process these documents to create a structured summary in Markdown format. The summary should include:
    
    - **Title and Subtitles**: Reflect the main sections and subsections of the paper.
    - **Bullet Points**: Summarize key points, findings, and data under appropriate sections.
    - **Conclusions**: Highlight the main conclusions of the paper.
    
    Aim for clarity and conciseness, ensuring the summary is well-organized and presents the essential information in a structured manner similar to the original paper. Use Markdown for formatting:
    
    # For main titles
    ## For subtitles
    - For bullet points
    
    Result:
    """
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    return LLMChain(llm=llm, prompt=reduce_prompt)

def create_map_reduce_chain(llm, map_chain, reduce_chain):
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )
    
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )
    
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

def split_documents(docs):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=300
    )
    return text_splitter.split_documents(docs)

def summarize_pdf(pdf_file_path):   
    llm = initialize_llm()
    map_chain = get_map_chain(llm)
    reduce_chain = get_reduce_chain(llm)
    map_reduce_chain = create_map_reduce_chain(llm, map_chain, reduce_chain)
    
    docs = load_and_split_pdf(pdf_file_path)
    split_docs = split_documents(docs)
    
    return map_reduce_chain.run(split_docs)