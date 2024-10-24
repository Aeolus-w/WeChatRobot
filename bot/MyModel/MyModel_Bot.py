import requests
from bot.bot import Bot
from bot.MyModel.MyModel_session import MyModelSession
from bot.session_manager import SessionManager
from common.log import logger
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from config import conf

class MyModelBot(Bot):
    def __init__(self):
        super().__init__()
        self.sessions = SessionManager(MyModelSession, model=conf().get("model") or "MyModel")
        self.api_url = "http://127.0.0.1:6006/v1/chat/completions"  # API的URL
        self.default_args = {
            "model": conf().get("model") or "chatglm3-6b",  # 模型名称
            "temperature": conf().get("temperature", 0.8),
            "top_p": conf().get("top_p", 0.8),
            "max_tokens": 1500,
        }

    def reply(self, query, context=None):
        if context.type == ContextType.TEXT:
            logger.info("[MyModel] query={}".format(query))
            session_id = context["session_id"]

            # 管理特殊指令，例如清除记忆等
            if query == "#清除记忆":
                self.sessions.clear_session(session_id)
                return Reply(ReplyType.INFO, "记忆已清除")
            
            session = self.sessions.session_query(query, session_id)
            reply_content = self.send_message_to_api(session)
            if reply_content:
                self.sessions.session_reply(reply_content["content"], session_id, reply_content["total_tokens"])
                return Reply(ReplyType.TEXT, reply_content["content"])
            else:
                return Reply(ReplyType.ERROR, "无法获取回复")

        else:
            return Reply(ReplyType.ERROR, "暂不支持{}类型的消息".format(context.type))

    def send_message_to_api(self, session: MyModelSession):
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "messages": session.messages,  # 使用会话中的消息
            "temperature": self.default_args["temperature"],
            "top_p": self.default_args["top_p"],
            "max_tokens": self.default_args["max_tokens"],
            "model": self.default_args["model"],
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()  # 检查请求是否成功
            data = response.json()
            return {
                "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                "content": data["choices"][0]["message"]["content"]
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"请求错误: {e}")
            return None
