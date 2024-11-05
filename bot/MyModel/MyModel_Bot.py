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
        self.api_endpoints = {
            "chat": "http://127.0.0.1:6006/v1/chat/completions",
            "upload": "http://127.0.0.1:6006/v1/upload",
        }
        self.default_args = {
            "model": conf().get("model") or "chatglm3-6b",
            "temperature": conf().get("temperature", 0.8),
            "top_p": conf().get("top_p", 0.8),
            "max_tokens": 1500,
        }

    def reply(self, query, context=None):
        if context.type == ContextType.TEXT:
            return self.handle_text_query(query, context)

        elif context.type == ContextType.FILE:
            return self.handle_file_upload(context)

        else:
            return Reply(ReplyType.ERROR, "暂不支持{}类型的消息".format(context.type))

    def handle_text_query(self, query, context):
        logger.info("[MyModel] query={}".format(query))
        session_id = context["session_id"]

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

    def handle_file_upload(self, context):
        if "file" not in context or "path" not in context["file"]:
            logger.error("缺少文件路径信息")
            return Reply(ReplyType.ERROR, "文件信息缺失，无法上传")

        file_path = context["file"]["path"]
        session_id = context["session_id"]
        logger.info(f"准备上传文件，路径: {file_path}, 会话ID: {session_id}")

        try:
            # 检查文件是否存在
            with open(file_path, 'rb') as file:
                logger.info("找到文件，开始上传...")
                upload_response = self.upload_file(file, session_id)
                if upload_response:
                    logger.info("文件上传成功。")
                    return Reply(ReplyType.INFO, "文件已经上传至知识库，您可以向我提问。")
                else:
                    logger.error("文件上传失败，未返回有效响应。")
                    return Reply(ReplyType.ERROR, "文件上传失败。")
        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            return Reply(ReplyType.ERROR, "文件未找到，上传失败。")
        except Exception as e:
            logger.error(f"上传过程中发生错误: {e}")
            return Reply(ReplyType.ERROR, "上传过程中发生错误。")

    def upload_file(self, file, session_id):
        """
        上传文件到知识库 API，并附带会话ID
        """
        files = {'file': file}
        data = {'session_id': session_id}
        try:
            response = self._post_request(self.api_endpoints["upload"], files=files, data=data)
            if response:
                return response  # 如果上传成功，返回响应
            else:
                logger.error("上传请求未返回有效响应。")
                return None
        except Exception as e:
            logger.error(f"上传文件时发生错误: {e}")
            return None

    def send_message_to_api(self, session: MyModelSession):
        payload = {
            "messages": session.messages,
            "temperature": self.default_args["temperature"],
            "top_p": self.default_args["top_p"],
            "max_tokens": self.default_args["max_tokens"],
            "model": self.default_args["model"],
        }
        return self._post_request(self.api_endpoints["chat"], json=payload)

    def _post_request(self, url, **kwargs):
        try:
            logger.info(f"向 {url} 发送请求，数据: {kwargs}")
            response = requests.post(url, **kwargs)
            response.raise_for_status()
            data = response.json()
            if "usage" in data and "choices" in data:
                return {
                    "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                    "content": data["choices"][0]["message"]["content"]
                }
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"请求错误: {e}")
            return None
