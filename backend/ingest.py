from langchain.document_loaders import PyPDFDirectoryLoader, WebBaseLoader, DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from typing import Literal,Any,Optional
from PyPDF2 import PdfReader

import toml

# Configuration
# Configuration
config = toml.load("config.toml")
api_key = config['API_KEYS']['OPENAI']
claude_key = config['API_KEYS']['CLAUDE']
csv_path= config['LOAD_FILE']['CSV']
index_file_path= config["SAVE_DB"]["DB"]
embedding_mode =config['STORAGE']['Embedded_mode']


class Document:
    def __init__(self, page_content, metadata: Optional[dict] = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

class Ingestdocument():
    def __init__(self,config):
        self.config = config['LOAD_FILE']
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=40, separators=["\n\n","\n"," ",".",",","\u200B", "\uff0c","\u3001","\uff0e",
        "\u3002",
        "", #1024
    ],)

    def WebLoader(self):
        # https://github.com/rh-aiservices-bu/llm-on-openshift/blob/dbbd56ad5a99cd79c5e65fd6da8dc47466c2a3b4/examples/notebooks/langchain/Langchain-PgVector-Ingest.ipynb#L84
        website_loader = WebBaseLoader(self.config['websites_list'])
        website_docs = website_loader.load()
        docs = self.text_splitter.split_documents(website_docs)
        return docs
    def csv_loader(self):
        df = pd.read_csv(self.config['CSV'])
        docs = [Document(text) for text in df['concatenated']]
        return docs
    def load_multiPDF_from_directory(self):
        PDFs_path = "data/Law_data/"
        dir_loader = DirectoryLoader(
                        PDFs_path,
                        glob="./*.pdf",
                        loader_cls=PyPDFLoader,
                        show_progress=True,
                        use_multithreading=True)
        documents_loader= dir_loader.load()
        dir_documents_chunk = self.text_splitter.split_documents(documents_loader)
        return dir_documents_chunk
    def pdf_reader(self):
        loader = PyPDFLoader("D:/chatbot_project/chatbot/data/Law_data/219.pdf")
        data = loader.load()
        docs =self.text_splitter.split_documents(data)

        #pages = loader.load_and_split()
        return docs #loader #pages
        
    def DFloader(self):
        df = pd.read_csv('data/Bank_data/vietcombank_faqs.csv')
        df['context'] = 'Đây là câu hỏi:'+df["question"] +"Kèm câu trả lời" +df['answer']
        loader = DataFrameLoader(df,page_content_column='context')
        data = loader.load()
        docs =self.text_splitter.split_documents(data)
        return docs
    def load_word(self):
        from langchain_community.document_loaders import Docx2txtLoader,UnstructuredWordDocumentLoader
        from langchain.document_loaders import ReadTheDocsLoader
        loader = ReadTheDocsLoader("D:/chatbot_project/chatbot/data/Law_data/219.doc", features='html.parser')
        # docs = loader.load()
        # loader = UnstructuredWordDocumentLoader()
        # data = loader.load()
        return loader
    def get_pdf_text(pdf_docs):
        from PyPDF2 import PdfReader
        # https://github.com/panhdjf/Key-extract-information/blob/51c010af1a008463f4cc0713a161c4756381eb70/main.py
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
        

from langchain.embeddings import HuggingFaceEmbeddings,OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
import torch


def get_response_from_query(query, chunks):
    docs = " ".join([d[0].page_content for d in chunks])
    chain = LLMChain(llm=llm, prompt=prompt)

    output = chain.run({'question': query, 'docs': docs})
    return output

docs = Ingestdocument(config).pdf_reader()#load_multiPDF_from_directory()#load_word()## #DFloader()
# https://github.com/AndrewHaward2310/DENSO_GPT_Expert/blob/db5687b05f9a23eed63338733d6a8e55e7827a88/src/Core/model.py#L20
# embeddings= HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder", model_kwargs={"device":'cpu'})

embeddings= HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder", model_kwargs={"device":'cpu'})

#"keepitreal/vietnamese-sbert" ) "bkai-foundation-models/vietnamese-bi-encoder"
db = FAISS.from_documents(docs, embeddings) 


PROMPT_TEMPLATE = """Instructions: 
1. This is a FAQ chatbot for Vietcombank (VCB). You must respond in Vietnamese.
2. Remember that this is the chatbot for Vietcombank (VCB), so you must answer questions related to Vietcombank (VCB) only. Here is the list of some other banks. If the user asks about other banks, you can say "I'm sorry, I can't answer that question", and point our the reason clearly.
3. Be polite and respectful to the user. Don't use inappropriate language or make inappropriate jokes. Don't ask for personal information from the user. Don't provide false information to the user. Don't answer to questions with political, religious, or sensitive topics. In those cases, you can say "I'm sorry, I can't answer that question", and point our the reason clearly.
4. If the process mentioned in the answer has multiple steps, break it down into smaller steps and explain each step clearly in separate bullets.
5. Be clear, precise and concise in your answers.
6. Focus on the question. Previous questions and answers are shown in the context part. You can decide to use the context or not based on the current question.
You will be provided Task and detail instructions, follow this

Task: Please extract information regarding laws and circulars from the provided document. Identify the relevant provisions associated with each law and any circulars mentioned. Use the circulars as evidence to support answers related to the user's question.

Detailed Instructions:

1. Document Analysis:

+ Extract details about laws and circulars mentioned in the document.
+ Note down the specific provisions associated with each law.

2.Provision Identification:
+ For each law identified, locate the provisions within the document that pertain to it.

3.Circular Utilization:

+Identify any circulars referenced within the document.
+Use the information from these circulars as evidence to support answers for the user's questions.

4. Output Requirements:

+Provide a summary of laws and circulars found in the document.
+State the relevant provisions linked to each law.
+Use the circulars as evidence to answer questions based on the extracted content.

Context:
{CONTEXT}

Question:
Let's think step by step. {QUESTION}

Answer in Vietnamese:

"""
# chunks = db.similarity_search_with_score(query, k=3)
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain import HuggingFacePipeline

# https://github.com/sophiamyang/tutorials-LangChain/blob/main/LangChain_QA.ipynb
# template = """Instructions: 
# 1. This is a FAQ chatbot for Vietcombank (VCB). You must respond in Vietnamese.
# 2. Remember that this is the chatbot for Vietcombank (VCB), so you must answer questions related to Vietcombank (VCB) only. Here is the list of some other banks. If the user asks about other banks, you can say "I'm sorry, I can't answer that question", and point our the reason clearly.
# 3. Be polite and respectful to the user. Don't use inappropriate language or make inappropriate jokes. Don't ask for personal information from the user. Don't provide false information to the user. Don't answer to questions with political, religious, or sensitive topics. In those cases, you can say "I'm sorry, I can't answer that question", and point our the reason clearly.
# 4. If the process mentioned in the answer has multiple steps, break it down into smaller steps and explain each step clearly in separate bullets.
# 5. Be clear, precise and concise in your answers.
# 6. Focus on the question. Previous questions and answers are shown in the context part. You can decide to use the context or not based on the current question.

# Context:

# Question:

# Answer in Vietnamese:
# {context}

# Let's think step by step: {question}
# """
template = """Instructions: 
1. This is a chatbot used to answer question related to VAT(Value Added Tax)law. You must respond in Vietnamese.
2. Remember that this is the chatbot for VAT law, so you must answer questions related to VAT only.If the user asks about other topics, you can say "I'm sorry, I can't answer that question", and point our the reason clearly.
3. Be polite and respectful to the user. Don't use inappropriate language or make inappropriate jokes. Don't ask for personal information from the user. Don't provide false information to the user. Don't answer to questions with political, religious, or sensitive topics. In those cases, you can say "I'm sorry, I can't answer that question", and point our the reason clearly.
4. If the process mentioned in the answer has multiple steps, break it down into smaller steps and explain each step clearly in separate bullets.
5. Be clear, precise and concise in your answers.
6. Focus on the question. Previous questions and answers are shown in the context part. You can decide to use the context or not based on the current question.
You will be provided task and detail instructions, follow this to make decisions

Task: Please extract information regarding laws and circulars from the provided document. Identify the relevant provisions associated with each law and any circulars mentioned. Use the circulars as evidence to support answers related to the user's question.

Detailed Instructions:

1. Document Analysis:

+ Extract details about laws and circulars mentioned in the document.
+ Note down the specific provisions associated with each law.

2.Provision Identification:
+ For each law identified, locate the provisions within the document that pertain to it.

3.Circular Utilization:

+Identify any circulars referenced within the document.
+Use the information from these circulars as evidence to support answers for the user's questions.

4. Output Requirements:

+Provide a summary of laws and circulars found in the document.
+State the relevant provisions linked to each law.
+Use the circulars as evidence to answer questions based on the extracted content.

Context:
{context}
Question: Let's think step by step: {question}

Answer in Vietnamese:
"""
def get_conversation_chain(db):
    prompt = ChatPromptTemplate.from_template(template)

    # OpenAI model 
    # llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1, openai_api_key=api_key)
    llm = ChatOpenAI(model_name='gpt-4-turbo', temperature=0.1, openai_api_key=api_key)

    # from langchain_anthropic import ChatAnthropic
    # llm = ChatAnthropic(model='claude-3-opus-20240229',api_key=claude_key)
    # Custom/Huggingface model
    # cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/tmp'

    # https://github.com/panhdjf/Key-extract-information/blob/51c010af1a008463f4cc0713a161c4756381eb70/main.py
    #from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    #tokenizer = AutoTokenizer.from_pretrained("TheBloke/claude2-alpaca-13B-GPTQ")#'bkai-foundation-models/vietnamese-bi-encoder')
    #model = AutoModelForCausalLM.from_pretrained("TheBloke/claude2-alpaca-13B-GPTQ")#'bkai-foundation-models/vietnamese-bi-encoder') AutoModel.
    # tokenizer = AutoTokenizer.from_pretrained("vilm/vinallama-7b")#, cache_dir=cache_dir) 
    # model = AutoModelForCausalLM.from_pretrained("vilm/vinallama-7b", torch_dtype=torch.float32)#, cache_dir=cache_dir)
    # text_pipeline = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=512,
    #     temperature=0.2,
    #     top_p=0.95,
    #     repetition_penalty=1.15,
    #     # streamer=streamer,
    # )
    # llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.2, "max_length":512})

    retriever = db.as_retriever()
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever=retriever, 
        memory=memory, 
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return qa

def qa_answer(message, history):
    return qa(message)["answer"]

history = []
message= "Dịch vụ tài chính liên quan đến tín dụng có cần đóng thuế giá trị gia tăng không"
#"Những trường hợp không kê khai, nộp thuế giá trị gia tăng"
#"Cá nhân không phải tổ chức chuyên kinh doanh thì khi bán tài sản có phải kê khai thuế giá trị gia tăng không"
#"Những trường hợp không kê khai, nộp thuế giá trị gia tăng"
#"Dịch vụ tài chính liên quan đến tín dụng có cần đóng thuế giá trị gia tăng không"
# qa = get_conversation_chain(db)
# answer = qa_answer(message,history)
# print(answer)
# history.append((answer,history))




# import re
# import unicodedata

# reader = PdfReader("data/Bank_data/22-Mo-tai-khoan-so-ien-thoai.pdf")
# page = reader.pages[0]
# print(page.extract_text())
# Ingest= Ingestdocument(config)
# docs = Ingest.loadPDF_from_directory() #csv_loader()#.WebLoader()
# full_text = []
# for chunk in docs:
#     content = chunk.page_content
#     encode_text =content
#     full_text.append(encode_text)
# print(full_text)


# def normalize_text(text):
#     normalized_text = unicodedata.normalize('NFKD', text)
#     return normalized_text

# def clean_text(text):
#     cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     return cleaned_text
# new_text =[]
# # Remove excess newlines and whitespace
# for t in full_text:
#     text = normalize_text(t)
#     cleaned_text = clean_text(text)
#     new_text.append(cleaned_text)
# print(new_text)


    
