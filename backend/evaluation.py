# https://docs.ragas.io/en/stable/howtos/integrations/langchain.html
from ragas.langchain.evalchain import RagasEvaluatorChain
# https://github.com/explodinggradients/ragas/issues/571
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    # context_precision,
    context_recall,
)
import os
import openai
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import toml
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
# Configuration
api_key = "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"
# # Initialize ChatOpenAI
# llm = ChatOpenAI(openai_api_key=api_key) 
embeddings= HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder", model_kwargs={"device":'cpu'})
# embeddings= HuggingFaceEmbeddings(model_name="SeaLLMs/SeaLLM-7B-v2.5", model_kwargs={"device":'cpu'})

#"keepitreal/vietnamese-sbert" ) "bkai-foundation-models"/vietnamese-bi-encoder"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=40, separators=["\n\n","\n"," ",".",",","\u200B", "\uff0c","\u3001","\uff0e",
        "\u3002",
        "",])
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("D:/chatbot_project/chatbot/data/Law_data/219.pdf")
data = loader.load()
docs =text_splitter.split_documents(data)

# loader = TextLoader("./nyc_wikipedia/nyc_text.txt")
# https://github.com/langchain-ai/langchain/issues/2326#issuecomment-1528898402
index = VectorstoreIndexCreator(vectorstore_cls=FAISS,embedding=embeddings).from_documents(docs)#.from_loaders([loader])
llm = ChatOpenAI(model_name='gpt-4-turbo', temperature=0.1, openai_api_key=api_key)
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
template = '''Hướng dẫn sử dụng:

Đây là trợ lý ảo để trả lời các câu hỏi liên quan đến Luật Thuế Giá Trị Gia Tăng (VAT). Bạn vui lòng trả lời bằng tiếng Việt.
+ Ghi nhớ rằng đây là trợ lý ảo về Luật VAT, vì vậy bạn chỉ nên trả lời các câu hỏi liên quan đến VAT. Nếu người dùng hỏi về các chủ đề khác, bạn có thể nói "Xin lỗi, tôi không thể trả lời câu hỏi đó" và giải thích rõ nguyên nhân.
+ Hãy lịch sự và tôn trọng người dùng. Xin đừng sử dụng ngôn ngữ không thích hợp hoặc đưa ra những lời nói đùa không thích hợp. Xin đừng yêu cầu thông tin cá nhân từ người dùng. Xin đừng cung cấp thông tin sai lệch cho người dùng. Xin đừng trả lời các câu hỏi có tính chất chính trị, tôn giáo hoặc nhạy cảm. Trong những trường hợp này, bạn có thể nói "Xin lỗi, tôi không thể trả lời câu hỏi đó" và giải thích rõ nguyên nhân.
Nếu quá trình được đề cập trong câu trả lời có nhiều bước, hãy chia nhỏ thành các bước nhỏ hơn và giải thích từng bước một rõ ràng trong các mục riêng biệt.
+ Trả lời rõ ràng, chính xác và ngắn gọn.
+ Tập trung vào câu hỏi. Các câu hỏi trước và các câu trả lời được hiển thị trong phần ngữ cảnh. Bạn có thể quyết định sử dụng ngữ cảnh hoặc không dựa trên câu hỏi hiện tại. Khi nhận được câu hỏi, bạn có thể tìm kiếm thông tin trong tài liệu.
Ví dụ:
Câu hỏi: Để được hoàn thuế giá trị gia tăng đối với dự án đầu tư, doanh nghiệp cần phải đáp ứng những điều kiện nào?

Trả lời:
Phần 1 - Thông tin từ tài liệu:

Theo khoản 3, Điều 1 của Luật số 106/2016/QH13 ngày 6.4.2016 của Quốc hội về sửa đổi và bổ sung một số điều của Luật Thuế Giá Trị Gia Tăng, Luật Thuế Tiêu Thụ Đặc Biệt và Luật Quản Lý Thuế quy định:
"3. Cơ sở kinh doanh được hoàn thuế giá trị gia tăng nếu đáp ứng điều kiện về dự án đầu tư mới và có số thuế giá trị gia tăng còn lại từ ba trăm triệu đồng trở lên."

Phần 2 - Phân tích và giải thích:
Căn cứ vào các quy định và hướng dẫn trên, doanh nghiệp mới thành lập từ dự án đầu tư và doanh nghiệp đang hoạt động có dự án đầu tư (trừ dự án xây nhà để bán hoặc cho thuê mà không hình thành tài sản cố định), nếu đáp ứng các quy định về dự án đầu tư theo Luật Đầu Tư và có số thuế GTGT hàng hóa, dịch vụ mua vào sử dụng cho đầu tư đủ điều kiện kê khai và khấu trừ thuế theo quy định, và có số thuế còn lại từ ba trăm triệu đồng trở lên, sẽ được hoàn thuế giá trị gia tăng theo quy định.



Câu hỏi:
Hãy suy nghĩ từng bước: {query}

Trả lời bằng tiếng Việt:'''

prompt = ChatPromptTemplate.from_template(template)
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
#======================== Reranker ====================================#
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# compressor = CohereRerank(cohere_api_key='LzEgQJ1eQWPCF6Wt1BqpSxPOmBN2LXQz2lWCJz9m')

#======================== Contextual Compressor ====================================#
from langchain.retrievers.document_compressors import LLMChainExtractor
compressor = LLMChainExtractor.from_llm(llm)

# retriever =index.vectorstore.as_retriever()
#==============================================================#
retriever= ContextualCompressionRetriever(base_compressor=compressor, base_retriever=index.vectorstore.as_retriever(search_kwargs={"k": 2}))
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm, 
#     retriever=retriever, 
#     memory=memory, 
    # combine_docs_chain_kwargs={"prompt": prompt}
# )
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever, #index.vectorstore.as_retriever()
    return_source_documents=True,
)
# combine_docs_chain_kwargs={"prompt": prompt}
# https://arxiv.org/ftp/arxiv/papers/2301/2301.11252.pdf
eval_questions = [
    "Sản phẩm nuôi trồng, thủy sản chưa qua bước chế biến có phải chịu thuế gtgt không?",

    "Đối tượng chịu thuế giá trị gia tăng bao gồm những đối tượng nào",

    "Cá nhân, tổ chức có hoạt động sản xuất tại Việt Nam khi mua dịch vụ của cá nhân, tổ chức nước ngoài không có cư trú tại Việt Nam thì có phải chịu thuế gtgt không?",

    "Hộ kinh doanh có doanh thu dưới 100tr có phải kê khai thuế gtgt không?",

    "Mức thuế suất 0% áp dụng đối với loại hàng hóa nào?",
    "Mức thuế suất 5% áp dụng đối với hàng hóa, dịch vụ nào?",
]

eval_answers = [
    "Theo Điều 4 Luật Thuế giá trị gia tăng và khoản 1 Điều 4 Thông tư 219/2013/TT-BTC (được sửa đổi bởi Thông tư 26/2015/TT-BTC), sản phẩm trồng trọt, chăn nuôi, thủy sản, hải sản nuôi trồng, đánh bắt chưa chế biến thành các sản phẩm khác hoặc chỉ qua sơ chế thông thường được xem là đối tượng không chịu thuế GTGT.Cụ thể, các hình thức sơ chế thông thường được quy định bao gồm: làm sạch, phơi, sấy khô, bóc vỏ, xay, xay bỏ vỏ, xát bỏ vỏ, tách hạt, tách cọng, cắt, ướp muối, bảo quản lạnh, bảo quản bằng khí sunfuro, bảo quản theo phương thức cho hóa chất để tránh thối rữa, ngâm trong dung dịch lưu huỳnh hoặc ngâm trong dung dịch bảo quản khác và các hình thức bảo quản thông thường khác.Vì vậy, sản phẩm nuôi trồng, thủy sản chưa qua chế biến hoặc chỉ qua các hình thức sơ chế thông thường như trên sẽ không phải nộp thuế GTGT.Trường hợp sản phẩm đã được chế biến thành các sản phẩm khác hoặc qua các hình thức sơ chế ngoài những hình thức trên, thì sẽ phải chịu thuế GTGT theo mức thuế suất 5% theo quy định.",

    "1. Tổ chức, cá nhân sản xuất, kinh doanh hàng hóa, dịch vụ chịu thuế GTGT ở Việt Nam, bao gồm cả tổ chức, cá nhân nước ngoài kinh doanh tại Việt Nam.2. Tổ chức, cá nhân nhập khẩu hàng hóa chịu thuế GTGT.3. Tổ chức, cá nhân bán, chuyển nhượng bất động sản, chuyển quyền sử dụng đất.4. Tổ chức, cá nhân kinh doanh dịch vụ như: vận tải, bưu chính, viễn thông, tài chính, ngân hàng, bảo hiểm, kinh doanh chứng khoán, y tế, giáo dục - đào tạo, xây dựng, nhà hàng, khách sạn, du lịch, casino, trò chơi điện tử có thưởng, kinh doanh các dịch vụ khác.5. Tổ chức, cá nhân bán, chuyển nhượng, cho thuê, cho thuê lại tài sản như nhà, đất, máy móc, thiết bị.6. Tổ chức, cá nhân khác có hoạt động sản xuất, kinh doanh, nhập khẩu hàng hóa, dịch vụ chịu thuế GTGT.Các đối tượng này phải kê khai, nộp thuế GTGT theo quy định của pháp luật.Trường hợp đối tượng không thuộc diện chịu thuế GTGT, sẽ được quy định cụ thể tại Điều 5 Luật Thuế giá trị gia tăng.",

    "Theo quy định tại Điều 1 Luật Thuế giá trị gia tăng, đối tượng chịu thuế GTGT bao gồm: 1. Tổ chức, cá nhân sản xuất, kinh doanh hàng hóa, dịch vụ chịu thuế GTGT ở Việt Nam. 2. Tổ chức, cá nhân nhập khẩu hàng hóa chịu thuế GTGT.Theo Điều 3 Luật Thuế giá trị gia tăng, dịch vụ cung ứng tại Việt Nam, bao gồm cả dịch vụ do cá nhân, tổ chức nước ngoài không có cơ sở thường trú tại Việt Nam cung ứng, đều được áp dụng thuế GTGT. Vì vậy, trong trường hợp cá nhân, tổ chức tại Việt Nam mua dịch vụ của cá nhân, tổ chức nước ngoài không có cơ sở thường trú tại Việt Nam, thì đối tượng mua dịch vụ (cá nhân, tổ chức tại Việt Nam) phải kê khai, nộp thuế GTGT theo quy định.Mức thuế suất áp dụng đối với dịch vụ này là 10% theo quy định tại Điều 8 Luật Thuế giá trị gia tăng.",

    "Theo Điều 13 Luật Thuế giá trị gia tăng, các đối tượng sau được miễn nộp thuế GTGT: 1. Hộ gia đình, cá nhân kinh doanh có doanh thu hàng năm dưới mức doanh thu chịu thuế GTGT do Chính phủ quy định.2. Cơ sở kinh doanh mới thành lập trong thời gian được miễn, giảm thuế GTGT theo quy định của pháp luật về ưu đãi, khuyến khích đầu tư.Theo Nghị định 209/2013/NĐ-CP về quản lý thuế, cụ thể tại Điều 7 quy định:- Hộ gia đình, cá nhân kinh doanh có tổng doanh thu hàng năm dưới 100 triệu đồng thì được miễn nộp thuế GTGT.Như vậy, căn cứ các quy định trên, hộ kinh doanh có doanh thu dưới 100 triệu đồng/năm sẽ được miễn nộp thuế GTGT.Tuy nhiên, hộ kinh doanh này vẫn phải thực hiện đăng ký kinh doanh, lập sổ sách kế toán và chịu sự quản lý của cơ quan thuế theo quy định.",

    "Theo quy định của Nghị định 209/2013/NĐ-CP thì mức thuế suất 0% áp dụng đối với hàng hóa, dịch vụ xuất khẩu, vận tải quốc tế, hàng hóa, dịch vụ thuộc diện không chịu thuế giá trị gia tăng quy định tại Điều 5 Luật thuế giá trị gia tăng và Khoản 1 Điều 1 của Luật sửa đổi, bổ sung một số điều của Luật thuế giá trị gia tăng khi xuất khẩu, trừ các hàng hóa, dịch vụ quy định tại Điểm đ Khoản 1 Điều 6 Nghị định này. Hàng hóa, dịch vụ xuất khẩu là hàng hóa, dịch vụ được bán, cung ứng cho tổ chức, cá nhân ở nước ngoài và tiêu dùng ở ngoài Việt Nam, trong khu phi thuế quan; hàng hóa, dịch vụ cung cấp cho khách hàng nước ngoài theo quy định của pháp luật.a) Đối với hàng hóa xuất khẩu bao gồm: Hàng hóa xuất khẩu ra nước ngoài, bán vào khu phi thuế quan; công trình xây dựng, lắp đặt ở nước ngoài, trong khu phi thuế quan; hàng hóa bán mà điểm giao, nhận hàng hóa ở ngoài Việt Nam; phụ tùng, vật tư thay thế để sửa chữa, bảo dưỡng phương tiện, máy móc thiết bị cho bên nước ngoài và tiêu dùng ở ngoài Việt Nam; xuất khẩu tại chỗ và các trường hợp khác được coi là xuất khẩu theo quy định của pháp luật.b) Đối với dịch vụ xuất khẩu bao gồm dịch vụ cung ứng trực tiếp cho tổ chức, cá nhân ở nước ngoài hoặc ở trong khu phi thuế quan và tiêu dùng ở ngoài Việt Nam, tiêu dùng trong khu phi thuế quan.Trường hợp cung cấp dịch vụ mà hoạt động cung cấp vừa diễn ra tại Việt Nam, vừa diễn ra ở ngoài Việt Nam nhưng hợp đồng dịch vụ được ký kết giữa hai người nộp thuế tại Việt Nam hoặc có cơ sở thường trú tại Việt Nam thì thuế suất 0 phần trăm chỉ áp dụng đối với phần giá trị dịch vụ thực hiện ở ngoài Việt Nam, trừ trường hợp cung cấp dịch vụ bảo hiểm cho hàng hóa nhập khẩu được áp dụng thuế suất 0% trên toàn bộ giá trị hợp đồng. Trường hợp, hợp đồng không xác định riêng phần giá trị dịch vụ thực hiện tại Việt Nam thì giá tính thuế được xác định theo tỷ lệ (%) chi phí phát sinh tại Việt Nam trên tổng chi phí.Cá nhân ở nước ngoài là người nước ngoài không cư trú tại Việt Nam, người Việt Nam định cư ở nước ngoài và ở ngoài Việt Nam trong thời gian diễn ra việc cung ứng dịch vụ.Tổ chức, cá nhân trong khu phi thuế quan là tổ chức, cá nhân có đăng ký kinh doanh và các trường hợp khác theo quy định của Thủ tướng Chính phủ.c) Vận tải quốc tế quy định tại khoản này bao gồm vận tải hành khách, hành lý, hàng hóa theo chặng quốc tế từ Việt Nam ra nước ngoài hoặc từ nước ngoài đến Việt Nam, hoặc cả điểm đi và đến ở nước ngoài. Trường hợp, hợp đồng vận tải quốc tế bao gồm cả chặng vận tải nội địa thì vận tải quốc tế bao gồm cả chặng nội địa.d) Hàng hóa, dịch vụ xuất khẩu quy định tại các Điểm a, b Khoản này được áp dụng thuế suất 0% phải đáp ứng đủ điều kiện quy định tại Điểm c Khoản 2 Điều 9 Nghị định 209/2013/NĐ-CP và một số trường hợp hàng hóa, dịch vụ được áp dụng mức thuế suất 0% theo điều kiện do Bộ Tài chính quy định",

    "Mức thuế suất 5% áp dụng đối với hàng hóa, dịch vụ quy định tại Khoản 2 Điều 8 Luật thuế giá trị gia tăng và Khoản 3 Điều 1 Luật sửa đổi, bổ sung một số điều của Luật thuế giá trị gia tăng. Một số trường hợp áp dụng mức thuế suất 5% được quy định cụ thể như sau:a) Nước sạch phục vụ sản xuất và sinh hoạt quy định tại Điểm a Khoản 2 Điều 8 Luật thuế giá trị gia tăng không bao gồm các loại nước uống đóng chai, đóng bình và các loại nước giải khát khác thuộc diện áp dụng mức thuế suất 10%.b) Các sản phẩm quy định tại Điểm b Khoản 2 Điều 8 Luật thuế giá trị gia tăng bao gồm:- Phân bón là các loại phân hữu cơ, phân vô cơ, phân vi sinh và các loại phân bón khác;- Quặng để sản xuất phân bón là các quặng làm nguyên liệu để sản xuất phân bón;- Thuốc phòng trừ sâu bệnh bao gồm thuốc bảo vệ thực vật và các loại thuốc phòng trừ sâu bệnh khác;- Các chất kích thích tăng trưởng vật nuôi, cây trồng.c) Thức ăn gia súc, gia cầm và thức ăn cho vật nuôi khác quy định tại Điểm c Khoản 2 Điều 8 Luật thuế giá trị gia tăng bao gồm các loại sản phẩm đã qua chế biến hoặc chưa qua chế biến như: Cám, bã, khô dầu các loại, bột cá, bột xương.d) Dịch vụ sơ chế, bảo quản sản phẩm quy định tại Điểm d Khoản 2 Điều 8 Luật thuế giá trị gia tăng gồm: Phơi, sấy khô, bóc vỏ, tách hạt, cắt lát, xay sát, bảo quản lạnh, ướp muối và các hình thức bảo quản thông thường khác.đ) Thực phẩm tươi sống quy định tại Điểm g Khoản 2 Điều 8 Luật thuế giá trị gia tăng gồm các loại thực phẩm chưa được làm chín hoặc chế biến thành các sản phẩm khác.Lâm sản chưa qua chế biến quy định tại Điểm g Khoản 2 Điều 8 Luật thuế giá trị gia tăng bao gồm các sản phẩm rừng tự nhiên khai thác thuộc nhóm: Song, mây, tre, nứa, nấm, mộc nhĩ; rễ, lá, hoa, cây làm thuốc, nhựa cây và các loại lâm sản khác.e) Sản phẩm hóa dược, dược liệu là nguyên liệu sản xuất thuốc chữa bệnh, thuốc phòng bệnh quy định tại Điểm 1, Khoản 2 Điều 8 Luật thuế giá trị gia tăng.g) Nhà ở xã hội quy định tại Điểm q Khoản 2 Điều 8 Luật thuế giá trị gia tăng và Khoản 3 Điều 1 Luật sửa đổi, bổ sung một số điều của Luật thuế giá trị gia tăng là nhà ở do Nhà nước hoặc tổ chức, cá nhân thuộc các thành phần kinh tế đầu tư xây dựng và đáp ứng các tiêu chí về nhà ở, về: giá bán nhà, về giá cho thuê, về giá cho thuê mua, về đối tượng, điều kiện được mua, được thuê, được thuê mua nhà ở xã hội theo quy định của pháp luật về nhà ở."


]
# contexts=[]
# for q in eval_questions:
#     contexts.append([docs.page_content for docs in index.vectorstore.as_retriever().get_relevant_documents(q)])
examples = [
   {"query": q, "ground_truth": [eval_answers[i]]} 
   for i, q in enumerate(eval_questions)
   ] 
# question = "How did New York City get its name?"
# result = qa_chain({"question": eval_questions[1],"contexts":contexts})


# create evaluation chains
faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
# context_rel_chain = RagasEvaluatorChain(metric=context_precision)
# context_recall_chain = RagasEvaluatorChain(metric=context_recall)

# run the queries as a batch for efficiency
predictions = qa_chain.batch(examples)
# Example
# print(f"This is example: \n {examples}")
# # Predictions
# print(f"And this is prediction: \n {predictions}")


# evaluate
print("Evaluating Faithfulness")
r = faithfulness_chain.evaluate(examples, predictions)
print(f"faithfulness score:{np.sum([score['faithfulness_score'] for score in r])/len([score['faithfulness_score'] for score in r])}")

print("Evaluating Answer revelance")
a = answer_rel_chain.evaluate(examples, predictions)
print(f"Answer revelance score:{np.sum([score['answer_relevancy_score'] for score in a])/len([score['answer_relevancy_score'] for score in a])}")

# print("Evaluating context_recall")
# cr = context_recall_chain.evaluate(examples, predictions)
# print(f"context_recall score:{cr}")
# print(f"context_recall score:{np.sum([score["context_recall_score"] for score in cr])/len([score["context_recall_score"] for score in cr])}")