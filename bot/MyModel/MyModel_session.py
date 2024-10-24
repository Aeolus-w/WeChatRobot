from bot.session_manager import Session
from common.log import logger


class MyModelSession(Session):
    def __init__(self, session_id, system_prompt=None, model="chatglm3-6b"):
        super().__init__(session_id, system_prompt)
        self.model = model
        self.reset()
        if not system_prompt:
            logger.warn("[MyModel] `system_prompt` can not be empty")

    def discard_exceeding(self, max_tokens, cur_tokens=None):
        try:
            cur_tokens = self.calc_tokens()
        except Exception as e:
            logger.debug(f"Exception when counting tokens: {e}")
        while cur_tokens > max_tokens and len(self.messages) > 2:
            self.messages.pop(1)  
            cur_tokens = self.calc_tokens()
        return cur_tokens

    def calc_tokens(self):
        # 简单模拟token的计算，假设消息内容长度近似token数量
        return sum(len(msg["content"]) for msg in self.messages)
