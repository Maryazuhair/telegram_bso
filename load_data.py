import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import shutil

load_dotenv()

def clean_arabic_text(text):
    """تنظيف النص العربي من الأسطر الفارغة أو غير العربية"""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip() and not all(ord(c) < 128 for c in line.strip() if c != ' '):
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def load_and_split_document(pdf_path):
    print("📄 جاري تحميل ملف PDF...")
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✅ تم تحميل {len(documents)} صفحة")
        for doc in documents:
            doc.page_content = clean_arabic_text(doc.page_content)
            doc.metadata["source"] = f"صفحة {doc.metadata.get('page', 0) + 1}"
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "،", "؛", " "]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ تم تقسيم المستند إلى {len(chunks)} جزء")
        return chunks
    except Exception as e:
        print(f"❌ خطأ في تحميل الملف: {e}")
        return None

def create_vector_store(chunks, persist_directory="./chroma_db"):
    print("🔧 جاري إنشاء قاعدة البيانات المتجهية...")
    try:
        if os.path.exists(persist_directory):
            print(f"🗑️ جاري حذف قاعدة البيانات القديمة...")
            shutil.rmtree(persist_directory)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"✅ تم إنشاء قاعدة البيانات في {persist_directory}")
        return vectorstore
    except Exception as e:
        print(f"❌ خطأ في إنشاء قاعدة البيانات: {e}")
        return None

def main():
    pdf_path = "bankf.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ ملف {pdf_path} غير موجود")
        return
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ مفتاح OPENAI_API_KEY غير موجود")
        return
    chunks = load_and_split_document(pdf_path)
    if chunks:
        create_vector_store(chunks)

if __name__ == "__main__":
    main()
