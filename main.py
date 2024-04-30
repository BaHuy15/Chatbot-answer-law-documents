from fastapi import APIRouter,HTTPException, Query, Form,FastAPI,Depends, Form,Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse,HTMLResponse
from backend.prompts import law_prompt as prompts#,prompts
from backend.llm import llm
# from backend.indexing import embedding_model
from backend.models import QAquestion,QARespond
# from backend.rag import search_docs
from backend.ingest import Ingestdocument,embeddings as embedding_model
import uvicorn
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings,OpenAIEmbeddings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
import toml
import os 


config = toml.load("config.toml")
api_key = config['API_KEYS']['OPENAI']
csv_path= config['LOAD_FILE']['CSV']
# index_file_path= config["SAVE_DB"]["DB"]
index_file_path= config["SAVE_DB"]["LawDB"]



# =============================Setup fastapi ============================#
app = FastAPI(title="Chatbot for customers service",
    description="",
    version="4.0.0",
    contact={
        "name": "Nguyễn Bá Huy",
        "url": "https://github.com/apocas/restai",
        "email": "bahuy1521@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },)
router = APIRouter()

# ============================ Frontend part =============================#
# Mount the static directory as a route
app.mount("/static", StaticFiles(directory="static"), name="static")
@router.get("/")
async def read_root():
    # Return the HTML file
    with open("bot.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

#===================================== Backend part =========================#

def measure_time(start_time, operation_name):
    try:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{operation_name}: {elapsed_time:.2f} seconds")
        return end_time
    except Exception as e:
        print(f"Error measuring time for {operation_name}: {str(e)}")
        return start_time

def build_rag_context(docs):
    context = ""
    for doc in docs:
        context += doc.page_content + "\n"
    return context


# Check if the FAISS index file exists
if os.path.exists(index_file_path):
    # Load the existing FAISS index
    new_db = FAISS.load_local(index_file_path, embedding_model, allow_dangerous_deserialization=True)
    
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    # ==================CohereRerank ======================#
    # from langchain_cohere import CohereRerank
    # compressor = CohereRerank(cohere_api_key='LzEgQJ1eQWPCF6Wt1BqpSxPOmBN2LXQz2lWCJz9m')
    # ==================Contextualcompression ======================#
    from langchain.retrievers.document_compressors import LLMChainExtractor
    compressor = LLMChainExtractor.from_llm(llm)
    retriever= ContextualCompressionRetriever(base_compressor=compressor, base_retriever=new_db.as_retriever(search_kwargs={"k": 2}))
else:
    # Load the data and create a new FAISS index
    # df = pd.read_csv(csv_path)
    # Define the Document class for storing the page content and metadata
    # class Document:
    #     def __init__(self, page_content, metadata: Optional[dict] = None):
    #         self.page_content = page_content
    #         self.metadata = metadata if metadata is not None else {}
    documents = Ingestdocument(config).pdf_reader()
    # documents = [Document(text) for text in df['concatenated']]
    vector = FAISS.from_documents(documents, embedding_model)
    vector.save_local(index_file_path)
    new_db = FAISS.load_local(index_file_path,embedding_model, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 2})

def search_docs(original_prompt,input_prompt,retrieval_chain):
    docs = retrieval_chain.invoke({"input":input_prompt})
    content = docs['context']
    context = build_rag_context(content)
    context_prompt = original_prompt.format(context=context, input=input_prompt)
    # print('this is context prompt',context_prompt)
    rag_respond = retrieval_chain.invoke({"input":context_prompt}) 
    return context_prompt,rag_respond

document_chain = create_stuff_documents_chain(llm, prompts)
# retriever = new_db.as_retriever()

start_time = time.time()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
end_time = measure_time(start_time, "Create retrieval_chain operation times")

@router.post("/getChatBotResponse")
async def answer_query(input_prompt:QAquestion)->QARespond:
    try:
        start_time = time.time()
        context_prompt,rag_respond = search_docs(prompts,input_prompt.query,retrieval_chain)
        end_time = measure_time(start_time, "RAG operation times")
        return QARespond(rag_prompt=context_prompt,answer=rag_respond['answer'])

    except Exception as e:
        print(f"Error retrieving answer: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# =================================== Execute code ==========================#
app.include_router(router)
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')


# "Dịch vụ tài chính liên quan đến tín dụng có cần đóng thuế giá trị gia tăng không"
# Những trường hợp không kê khai, nộp thuế giá trị gia tăng
# Cá nhân không phải tổ chức chuyên kinh doanh thì khi bán tài sản có phải kê khai thuế giá trị gia tăng không