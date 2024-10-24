from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import fitz 
import docx 

# 加载 DOCX 文件
def load_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)

# 加载 PDF 文件
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 加载文件
def load_documents(dir_path):
    docs = []
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if filename.endswith('.md'):  # 使用 Markdown 加载器处理 Markdown 文件
            loader = UnstructuredMarkdownLoader(file_path)
            docs.extend(loader.load())
        elif filename.endswith('.txt'):  # 使用 TextLoader 处理文本文件
            loader = TextLoader(file_path)
            docs.extend(loader.load())
        elif filename.endswith('.pdf'):  # 使用 PDF 处理器处理 PDF 文件
            pdf_text = load_pdf(file_path)
            docs.append(Document(page_content=pdf_text))  # 将文本包装为 Document 对象
        elif filename.endswith('.docx'):  # 使用 DOCX 处理器处理 DOCX 文件
            docx_text = load_docx(file_path)
            docs.append(Document(page_content=docx_text))  # 将文本包装为 Document 对象
    return docs

# 生成向量知识库
def create_vector_db(documents):
    # 分块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(documents)

    # 加载嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name='/root/autodl-tmp/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') 

    # 创建 Chroma 向量数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory='/root/autodl-tmp/knowledge_db' 
    )
    
    return vectordb

# 主程序
if __name__ == "__main__":
    dir_path = '/root/autodl-tmp/base_knowledge' 
    docs = load_documents(dir_path)
    vector_db = create_vector_db(docs)
