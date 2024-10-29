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
        # 将 API URL 和端点统一管理
        self.api_endpoints = {
            "chat": "http://127.0.0.1:6006/v1/chat/completions",
            "upload": "http://127.0.0.1:6006/v1/upload",
        }
        # 模型默认参数
        self.default_args = {
            "model": conf().get("model") or "chatglm3-6b",
            "temperature": conf().get("temperature", 0.8),
            "top_p": conf().get("top_p", 0.8),
            "max_tokens": 1500,
        }

    def reply(self, query, context=None):
        """
        根据消息类型处理文本或文件的回复请求
        """
        if context.type == ContextType.TEXT:
            return self.handle_text_query(query, context)

        elif context.type == ContextType.FILE:
            return self.handle_file_upload(context)

        else:
            return Reply(ReplyType.ERROR, "暂不支持{}类型的消息".format(context.type))

    def handle_text_query(self, query, context):
        """
        处理文本消息查询的逻辑，包括管理会话和发送 API 请求
        """
        logger.info("[MyModel] query={}".format(query))
        session_id = context["session_id"]

        # 处理特殊指令
        if query == "#清除记忆":
            self.sessions.clear_session(session_id)
            return Reply(ReplyType.INFO, "记忆已清除")

        # 创建或更新会话并发送消息
        session = self.sessions.session_query(query, session_id)
        reply_content = self.send_message_to_api(session)
        if reply_content:
            self.sessions.session_reply(reply_content["content"], session_id, reply_content["total_tokens"])
            return Reply(ReplyType.TEXT, reply_content["content"])
        else:
            return Reply(ReplyType.ERROR, "无法获取回复")

    def handle_file_upload(self, context):
        """
        处理文件上传，将文件发送到知识库并通知用户上传状态
        """
        file_path = context["file"]["path"]
        upload_response = self.upload_file(file_path)
        if upload_response:
            return Reply(ReplyType.INFO, "文件已经上传至知识库，您可以向我提问。")
        else:
            return Reply(ReplyType.ERROR, "文件上传失败。")

    def send_message_to_api(self, session: MyModelSession):
        """
        向聊天 API 发送消息
        """
        payload = {
            "messages": session.messages,
            "temperature": self.default_args["temperature"],
            "top_p": self.default_args["top_p"],
            "max_tokens": self.default_args["max_tokens"],
            "model": self.default_args["model"],
        }
        return self._post_request(self.api_endpoints["chat"], json=payload)

    def upload_file(self, file_path):
        """
        上传文件到知识库 API
        """
        files = {'file': open(file_path, 'rb')}
        try:
            return self._post_request(self.api_endpoints["upload"], files=files)
        finally:
            files['file'].close()

    def _post_request(self, url, **kwargs):
        """
        发送 POST 请求并处理通用的错误逻辑
        """
        try:
            response = requests.post(url, **kwargs)
            response.raise_for_status()
            data = response.json()
            if "usage" in data and "choices" in data:
                # 专门用于处理聊天 API 的返回
                return {
                    "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                    "content": data["choices"][0]["message"]["content"]
                }
            return True  # 用于文件上传的成功标志
        except requests.exceptions.RequestException as e:
            logger.error(f"请求错误: {e}")
            return None
