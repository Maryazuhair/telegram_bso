import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import hashlib

load_dotenv()
logger = logging.getLogger(__name__)

class BankChatbot:
    def __init__(self):
        logger.info("🔄 جاري تهيئة الشات بوت...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ مفتاح OPENAI_API_KEY غير موجود")
        
        # مسار قاعدة المتجهات (نفس مجلد chatbot.py)
        db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"❌ قاعدة البيانات غير موجودة في {db_path}. قم بتشغيل load_data.py أولاً.")
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        logger.info("✅ تم تحميل قاعدة المتجهات")
        
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0,
            max_tokens=1000,
            openai_api_key=api_key,
            streaming=True
        )
        
        self.conversation_histories = {}  # session_id -> list of messages
        self.response_cache = {}           # cache_key -> response
        
        self.chain = self._create_chain()
        logger.info("✅ تم تهيئة الشات بوت بنجاح")
    
    def _create_chain(self):
        system_message = """
        أنت المساعد الرقمي الرسمي لبنك سوريا و المهجر.
        
        ========== تعليمات ==========
        🏦 **الهوية:**
        - اسمك: "مساعد سوريا و المهجر"
        - أنت خبير مصرفي متخصص في البنك
        
        📝 قاعدة أساسية:
        قدم إجابات كاملة وشاملة. .
        التزم بالمعلومات المتوفرة في سياق المحادثة فقط. لا تخترع معلومات.
        
        🔍 كيفية الإجابة:
        1. اقرأ كل المعلومات المقدمة بعناية
        2. اجمع كل التفاصيل المتعلقة بالسؤال
        3. نظم المعلومات في فقرات مترابطة
        4. أضف أمثلة وتفاصيل إضافية إن وجدت
        
        🗣️ سياق المحادثة السابقة:
        {history}
        
        🔗 المعلومات المسترجعة من قاعدة المعرفة:
        {context}
        
        👤 المستخدم: {question}
        
        🤖 المساعد:
        """
        prompt = ChatPromptTemplate.from_template(system_message)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def get_context(input_dict):
            docs = retriever.invoke(input_dict["question"])
            return format_docs(docs)
        
        chain = (
            {
                "context": get_context,
                "question": lambda x: x["question"],
                "history": lambda x: x.get("history", "لا يوجد تاريخ سابق.")
            }
            | prompt
            | self.llm
        )
        return chain
    
    def _get_history(self, session_id: str, max_messages: int = 10) -> str:
        history = self.conversation_histories.get(session_id, [])
        if not history:
            return "لا يوجد تاريخ سابق."
        recent = history[-(max_messages * 2):]
        formatted = ""
        for msg in recent:
            role = "المستخدم" if msg["role"] == "user" else "المساعد"
            formatted += f"{role}: {msg['content']}\n"
        return formatted.strip()
    
    def _add_user_message(self, session_id: str, question: str):
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = []
        self.conversation_histories[session_id].append({"role": "user", "content": question})
    
    def _add_assistant_message(self, session_id: str, response: str):
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = []
        self.conversation_histories[session_id].append({"role": "assistant", "content": response})
        # حد طول التاريخ
        if len(self.conversation_histories[session_id]) > 20:
            self.conversation_histories[session_id] = self.conversation_histories[session_id][-20:]
    
    async def get_response_stream(self, question: str, session_id: str = "default"):
        """توليد رد متدفق مع الاحتفاظ بتاريخ المحادثة"""
        # مخبأ حسب السؤال فقط (بدون سياق)
        cache_key = hashlib.md5(question.strip().lower().encode('utf-8')).hexdigest()
        if cache_key in self.response_cache:
            logger.info(f"✅ استخدام رد مخبأ للسؤال: {question}")
            yield self.response_cache[cache_key]
            self._add_assistant_message(session_id, self.response_cache[cache_key])
            return
        
        # استرجاع التاريخ
        history = self._get_history(session_id)
        
        # إضافة سؤال المستخدم إلى التاريخ
        self._add_user_message(session_id, question)
        
        full_response = ""
        try:
            async for chunk in self.chain.astream({
                "question": question,
                "history": history
            }):
                content = chunk.content
                full_response += content
                yield content
            # تخزين الرد في المخبأ
            self.response_cache[cache_key] = full_response
            self._add_assistant_message(session_id, full_response)
        except Exception as e:
            logger.error(f"خطأ في التدفق: {e}", exc_info=True)
            yield "عذراً، حدث خطأ أثناء توليد الرد."
