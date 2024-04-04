from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  #, OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.llms import LlamaCpp
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_community.document_loaders import UnstructuredPDFLoader

import io
import requests
from bs4 import BeautifulSoup
from db import DB
import re


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def text_cleaner(txt: str):
    txt = re.sub(r' -\n', '', txt)
    return txt


class Model:

    def __init__(self) -> None:
        self.converse = None
        self.db = DB()

    @staticmethod
    async def get_raw_txt(pdf_docs):
        print("processing pdfs...")
        text = ""
        for pdf in pdf_docs:
            contents = await pdf.read()
            pdf_file = io.BytesIO(contents)
            pdf_reader = PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        print(text)
        return text

    @staticmethod
    def get_txt_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=512,
            chunk_overlap=256,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        print('chunks:', len(chunks))
        return chunks

    @staticmethod
    def get_vectors(text_chunks):

        hf_embeddings_model = HuggingFaceEmbeddings(
            model_name="cointegrated/LaBSE-en-ru", model_kwargs={"device": "cpu"}
        )

        vectors = FAISS.from_texts(texts=text_chunks, embedding=hf_embeddings_model)
        return vectors

    @staticmethod
    def get_conv_chain(vectorstore):

        llm = LlamaCpp(
            model_path="../models/saiga_mistral_7b_q8_0.gguf",
            temperature=0.8,
            max_tokens=512,
            top_p=1,
            callback_manager=None,  # callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
            n_ctx=2048,
            n_gpu_layers=30,
        )

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",  # тип поиска похожих документов
                search_kwargs={'k': 2, 'score_threshold': 1.6}
            ),
            memory=memory
        )
        return conversation_chain

    @staticmethod
    def web_scrap_to_txt(url):
        print("started webscraping...")
        content = []
        response = requests.get(url)

        # Create a BeautifulSoup object to parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract specific elements from the HTML
        title = soup.title.text
        paragraphs = soup.find_all('p')
        content.append(title)

        print("completed capturing title")
        for p in paragraphs:
            content.append(p.text)
        return ' '.join(content)

    async def model_data(self, pdf_docs="", url=""):
        raw_text = ""

        if pdf_docs is not None:
            raw_text = await self.get_raw_txt(pdf_docs)
        if url is not None:
            raw_text += self.web_scrap_to_txt(url)
        print("completed processing")
        db_data = self.db.get_data()
        print("completed getting data")
        db_data = db_data + " ".join(self.get_txt_chunks(raw_text))  # TODO: correct db
        vectordb = self.get_vectors(self.get_txt_chunks(raw_text))
        print("completed vectorization")
        self.converse = self.get_conv_chain(vectordb)
        self.db.store_data(db_data)
        print("completed model creation")
        return self.converse

    def get_model(self):
        return self.converse
