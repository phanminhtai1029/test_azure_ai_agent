import azure.functions as func
import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

# Import libraries
from google import genai
from supabase import create_client, Client
from pymongo import MongoClient
import requests

# Initialize FunctionApp
app = func.FunctionApp()

# ==================== CONFIGURATION ====================

# Environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MONGODB_CONNECTION_STRING = os.environ.get("MONGODB_CONNECTION_STRING")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Initialize clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
db = mongo_client.agent_db

# ==================== HELPER FUNCTIONS ====================

def send_telegram_message(text: str, chat_id: str = None) -> bool:
    """Gửi tin nhắn về Telegram"""
    try:
        target_chat_id = chat_id or TELEGRAM_CHAT_ID
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": target_chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")
        return False

def create_embedding(text: str) -> Optional[list]:
    """Tạo embedding bằng Gemini"""
    try:
        response = gemini_client.models.embed_content(
            model="models/text-embedding-004",
            contents=text
        )
        if hasattr(response, 'embeddings') and response.embeddings:
            embedding = response.embeddings[0]
            if hasattr(embedding, 'values'):
                return embedding.values
            else:
                return list(embedding)
        elif hasattr(response, 'embedding'):
            if hasattr(response.embedding, 'values'):
                return response.embedding.values
            else:
                return list(response.embedding)
        return None
    except Exception as e:
        logging.error(f"Error creating embedding: {e}")
        return None

def search_rag_documents(query: str, threshold: float = 0.5, count: int = 3) -> list:
    """Tìm kiếm tài liệu tương tự trong Supabase RAG"""
    try:
        query_embedding = create_embedding(query)
        if not query_embedding:
            return []
        
        result = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": count
            }
        ).execute()
        
        return result.data if result.data else []
    except Exception as e:
        logging.error(f"Error searching RAG: {e}")
        return []

def generate_ai_response(user_message: str, context: str = "") -> str:
    """Tạo phản hồi bằng Gemini AI"""
    try:
        prompt = f"""Bạn là AI Planning Assistant - trợ lý lập kế hoạch thông minh.

Nhiệm vụ: Giúp người dùng với mục tiêu và kế hoạch của họ.

Ngữ cảnh từ tài liệu đã lưu:
{context}

Tin nhắn người dùng: {user_message}

Hãy trả lời ngắn gọn, hữu ích bằng tiếng Việt. Nếu người dùng đưa ra mục tiêu, hãy:
1. Xác nhận mục tiêu
2. Đề xuất các bước thực hiện
3. Khuyến khích họ"""

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        return response.text
    except Exception as e:
        logging.error(f"Error generating AI response: {e}")
        return "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."

def save_user_message(chat_id: str, message: str):
    """Lưu tin nhắn người dùng vào MongoDB"""
    try:
        db.user_messages.insert_one({
            "chat_id": chat_id,
            "message": message,
            "timestamp": datetime.utcnow()
        })
        logging.info(f"Saved message from {chat_id}")
    except Exception as e:
        logging.error(f"Error saving message: {e}")

# ==================== FUNCTION 1: TELEGRAM WEBHOOK ====================

@app.route(route="telegram", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def TelegramWebhook(req: func.HttpRequest) -> func.HttpResponse:
    """
    Function xử lý webhook từ Telegram
    URL: /api/telegram
    Method: POST
    """
    logging.info('🤖 Telegram webhook triggered')

    try:
        # Parse request body
        req_body = req.get_json()
        logging.info(f"Received: {json.dumps(req_body, ensure_ascii=False)}")

        # Kiểm tra có message không
        if 'message' not in req_body:
            logging.info("No message in webhook")
            return func.HttpResponse("OK", status_code=200)

        message = req_body['message']
        chat_id = str(message.get('chat', {}).get('id', ''))
        text = message.get('text', '')
        
        if not text:
            logging.info("No text in message")
            return func.HttpResponse("OK", status_code=200)

        logging.info(f"📩 Chat ID: {chat_id}, Message: {text}")

        # Lưu tin nhắn vào MongoDB
        save_user_message(chat_id, text)

        # Xử lý commands
        if text.startswith('/'):
            if text == '/start':
                response_text = """👋 *Xin chào! Tôi là AI Planning Assistant!*

Tôi có thể giúp bạn:
✅ Tạo kế hoạch tuần tự động
✅ Nhắc nhở các mục tiêu hàng ngày
✅ Tìm kiếm tài liệu và ghi chú
✅ Theo dõi tiến độ công việc

*Cách sử dụng:*
Gửi mục tiêu của bạn cho tôi, ví dụ:
"Tôi muốn học Python trong 2 tuần"

Hoặc dùng các lệnh:
/help - Xem hướng dẫn chi tiết
/plan - Xem kế hoạch hiện tại"""
                
            elif text == '/help':
                response_text = """📖 *Hướng Dẫn Sử Dụng*

*1️⃣ Tạo Kế Hoạch:*
Gửi mục tiêu của bạn, ví dụ:
- "Tôi muốn học Python trong 2 tuần"
- "Giúp tôi tập thể dục đều đặn"

*2️⃣ Tìm Kiếm Tài Liệu:*
Hỏi về bất kỳ chủ đề nào, tôi sẽ tìm trong tài liệu đã lưu.

*3️⃣ Xem Kế Hoạch:*
Gõ /plan để xem kế hoạch hiện tại

*4️⃣ Tự Động Hóa:*
- Kế hoạch tuần mới: Chủ nhật 9h sáng
- Nhắc nhở hàng ngày: 4 lần (6h, 12h, 18h, 21h)
- Databases được làm mới tự động

Hãy bắt đầu bằng cách gửi mục tiêu của bạn! 🚀"""
                
            elif text == '/plan':
                # Lấy kế hoạch từ MongoDB
                try:
                    plans = list(db.approved_plans.find(
                        {"chat_id": chat_id}
                    ).sort("created_at", -1).limit(5))
                    
                    if plans:
                        response_text = "📋 *Kế Hoạch Của Bạn:*\n\n"
                        for i, plan in enumerate(plans, 1):
                            goal = plan.get('goal', 'N/A')
                            status = plan.get('status', 'pending')
                            created = plan.get('created_at', datetime.utcnow())
                            
                            # Format date
                            if isinstance(created, datetime):
                                date_str = created.strftime('%d/%m/%Y')
                            else:
                                date_str = 'N/A'
                            
                            status_emoji = "✅" if status == "completed" else "🔄"
                            response_text += f"{status_emoji} *{i}. {goal}*\n"
                            response_text += f"   📅 {date_str} | Status: {status}\n\n"
                    else:
                        response_text = """📋 *Bạn chưa có kế hoạch nào*

Hãy gửi mục tiêu của bạn để tôi tạo kế hoạch!

Ví dụ:
- "Tôi muốn học lập trình"
- "Giúp tôi giảm cân trong 1 tháng"
- "Làm sao để cải thiện tiếng Anh?\""""
                        
                except Exception as e:
                    logging.error(f"Error fetching plans: {e}")
                    response_text = "Xin lỗi, có lỗi khi lấy kế hoạch. Vui lòng thử lại."
                    
            else:
                response_text = """❓ *Lệnh không hợp lệ*

Các lệnh có sẵn:
/start - Bắt đầu
/help - Hướng dẫn
/plan - Xem kế hoạch

Hoặc gửi tin nhắn bình thường để chat với tôi!"""
        
        else:
            # Tin nhắn thường - Tìm kiếm RAG và tạo phản hồi
            logging.info("Processing regular message with RAG")
            
            # Tìm kiếm tài liệu liên quan
            rag_results = search_rag_documents(text, threshold=0.3, count=2)
            
            context = ""
            if rag_results:
                logging.info(f"Found {len(rag_results)} RAG results")
                context = "\n\n".join([
                    f"- {doc['content']}" for doc in rag_results
                ])
            else:
                logging.info("No RAG results found")
            
            # Tạo phản hồi AI
            response_text = generate_ai_response(text, context)

        # Gửi phản hồi về Telegram
        success = send_telegram_message(response_text, chat_id)
        
        if success:
            logging.info(f"✅ Response sent successfully to {chat_id}")
        else:
            logging.error(f"❌ Failed to send response to {chat_id}")

        return func.HttpResponse("OK", status_code=200)

    except Exception as e:
        logging.error(f"❌ Error in TelegramWebhook: {e}", exc_info=True)
        
        # Cố gắng gửi error message về Telegram
        try:
            if 'chat_id' in locals():
                send_telegram_message(
                    "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau.",
                    chat_id
                )
        except:
            pass
        
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=200  # Trả 200 để Telegram không retry
        )

# ==================== FUNCTION 2: WEEKLY PLANNER ====================

@app.timer_trigger(schedule="0 0 9 * * 0", arg_name="myTimer", run_on_startup=False)
def WeeklyPlanner(myTimer: func.TimerRequest) -> None:
    """
    Tạo kế hoạch tuần mới mỗi Chủ nhật 9h sáng
    Schedule: 0 0 9 * * 0 (Cron format: giây phút giờ ngày tháng thứ)
    Thứ 0 = Chủ nhật
    """
    logging.info('📅 WeeklyPlanner triggered')
    
    try:
        # Lấy user profile từ MongoDB
        users = list(db.user_profile.find({}))
        
        if not users:
            logging.warning("No users found in user_profile")
            return
        
        for user in users:
            chat_id = user.get('chat_id')
            if not chat_id or chat_id == 'temp':
                continue
            
            logging.info(f"Creating weekly plan for user: {chat_id}")
            
            # Lấy pending plans của user
            pending_plans = list(db.pending_plans.find(
                {"chat_id": chat_id, "status": "pending"}
            ).limit(5))
            
            if not pending_plans:
                # Gửi reminder để tạo kế hoạch
                message = """📅 *Kế Hoạch Tuần Mới*

Chào buổi sáng Chủ nhật! 🌅

Bạn chưa có kế hoạch nào cho tuần này. Hãy gửi mục tiêu của bạn để tôi giúp tạo kế hoạch nhé!

Ví dụ:
- "Tôi muốn học Python cơ bản"
- "Giúp tôi tập thể dục đều đặn"
- "Cải thiện kỹ năng giao tiếp"

Hãy bắt đầu tuần mới với mục tiêu rõ ràng! 💪"""
                
                send_telegram_message(message, chat_id)
                continue
            
            # Tạo tổng hợp kế hoạch
            plan_summary = "📋 *Kế Hoạch Tuần Này:*\n\n"
            
            for i, plan in enumerate(pending_plans, 1):
                goal = plan.get('goal', 'N/A')
                plan_summary += f"{i}. {goal}\n"
            
            plan_summary += "\n💡 *Gợi ý:*\n"
            plan_summary += "• Chia nhỏ mục tiêu thành các bước nhỏ\n"
            plan_summary += "• Làm việc đều đặn mỗi ngày\n"
            plan_summary += "• Theo dõi tiến độ và điều chỉnh kịp thời\n\n"
            plan_summary += "Chúc bạn một tuần thành công! 🚀"
            
            send_telegram_message(plan_summary, chat_id)
            logging.info(f"✅ Weekly plan sent to {chat_id}")
        
        logging.info(f"✅ WeeklyPlanner completed for {len(users)} users")
        
    except Exception as e:
        logging.error(f"❌ Error in WeeklyPlanner: {e}", exc_info=True)


# ==================== FUNCTION 3: DAILY REMINDER ====================

@app.timer_trigger(schedule="0 0 6,12,18,21 * * *", arg_name="myTimer", run_on_startup=False)
def DailyReminder(myTimer: func.TimerRequest) -> None:
    """
    Gửi nhắc nhở hàng ngày 4 lần/ngày
    Schedule: 0 0 6,12,18,21 * * * 
    - 6h sáng, 12h trưa, 18h chiều, 21h tối
    """
    logging.info('⏰ DailyReminder triggered')
    
    try:
        current_hour = datetime.utcnow().hour + 7  # UTC+7 for Vietnam
        if current_hour >= 24:
            current_hour -= 24
        
        logging.info(f"Current hour (Vietnam): {current_hour}")
        
        # Lấy users từ MongoDB
        users = list(db.user_profile.find({}))
        
        if not users:
            logging.warning("No users found in user_profile")
            return
        
        for user in users:
            chat_id = user.get('chat_id')
            if not chat_id or chat_id == 'temp':
                continue
            
            # Check xem có reminder_times không
            reminder_times = user.get('reminder_times', ['06:00', '12:00', '18:00', '21:00'])
            
            # Kiểm tra xem giờ hiện tại có trong reminder_times không
            current_time = f"{current_hour:02d}:00"
            if current_time not in reminder_times:
                continue
            
            logging.info(f"Sending reminder to user: {chat_id} at {current_time}")
            
            # Lấy approved plans của user
            plans = list(db.approved_plans.find(
                {"chat_id": chat_id, "status": {"$ne": "completed"}}
            ).limit(3))
            
            if not plans:
                # Không có kế hoạch, gửi reminder tạo kế hoạch
                if current_hour == 6:  # Chỉ gửi vào buổi sáng
                    message = """☀️ *Chào Buổi Sáng!*

Bạn chưa có kế hoạch nào. Hãy bắt đầu ngày mới bằng cách đặt mục tiêu cho mình nhé!

Gửi mục tiêu của bạn để tôi giúp tạo kế hoạch. 💪"""
                    send_telegram_message(message, chat_id)
            else:
                # Có kế hoạch, gửi reminder
                time_messages = {
                    6: "☀️ *Chào Buổi Sáng!*",
                    12: "🌤️ *Nghỉ Trưa Rồi!*",
                    18: "🌆 *Buổi Chiều Vui Vẻ!*",
                    21: "🌙 *Buổi Tối An Lành!*"
                }
                
                greeting = time_messages.get(current_hour, "⏰ *Nhắc Nhở*")
                
                message = f"{greeting}\n\n📋 *Kế Hoạch Hôm Nay:*\n\n"
                
                for i, plan in enumerate(plans, 1):
                    goal = plan.get('goal', 'N/A')
                    message += f"{i}. {goal}\n"
                
                message += "\n💪 Hãy tiếp tục cố gắng nhé!"
                
                send_telegram_message(message, chat_id)
                logging.info(f"✅ Reminder sent to {chat_id}")
        
        logging.info(f"✅ DailyReminder completed for {len(users)} users")
        
    except Exception as e:
        logging.error(f"❌ Error in DailyReminder: {e}", exc_info=True)


# ==================== FUNCTION 4: KEEP ALIVE ====================

@app.timer_trigger(schedule="0 0 0 */5 * *", arg_name="myTimer", run_on_startup=False)
def KeepAlive(myTimer: func.TimerRequest) -> None:
    """
    Ping databases mỗi 5 ngày để tránh sleep
    Schedule: 0 0 0 */5 * * (Mỗi 5 ngày lúc 00:00)
    """
    logging.info('🔄 KeepAlive triggered')
    
    try:
        # Ping MongoDB
        mongo_result = db.command('ping')
        logging.info(f"✅ MongoDB ping: {mongo_result}")
        
        # Ping Supabase bằng cách query documents table
        supabase_result = supabase.table('documents').select('id').limit(1).execute()
        logging.info(f"✅ Supabase ping: {len(supabase_result.data) if supabase_result.data else 0} records")
        
        # Test Gemini API
        test_response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Say 'OK' in one word"
        )
        logging.info(f"✅ Gemini ping: {test_response.text[:20]}")
        
        # Gửi thông báo về Telegram
        message = """🔄 *Hệ Thống Đang Hoạt Động*

✅ MongoDB: Connected
✅ Supabase: Connected
✅ Gemini API: Active

Tất cả dịch vụ đang hoạt động bình thường! 💚"""
        
        send_telegram_message(message)
        
        logging.info("✅ KeepAlive completed successfully")
        
    except Exception as e:
        logging.error(f"❌ Error in KeepAlive: {e}", exc_info=True)
        
        # Gửi error alert
        try:
            error_message = f"""⚠️ *KeepAlive Error*

Có lỗi xảy ra khi ping databases:
{str(e)[:200]}

Vui lòng kiểm tra hệ thống!"""
            send_telegram_message(error_message)
        except:
            pass