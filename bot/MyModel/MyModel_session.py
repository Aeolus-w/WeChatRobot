from bot.session_manager import Session
from common.log import logger
import os
import json
from datetime import datetime
from pathlib import Path

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
        
        # 如果消息超过了最大token限制
        if cur_tokens > max_tokens:
            # 保存前一半消息
            self.save_history()
            
            # 删除前一半消息，保留后半部分
            half_length = len(self.messages) // 2
            self.messages = self.messages[half_length:]
            cur_tokens = self.calc_tokens()

            # 每次超出时保存历史
            self.save_history()

        return cur_tokens

    def calc_tokens(self):
        """
        简单模拟token的计算，假设消息内容长度近似token数量
        """
        return sum(len(msg["content"]) for msg in self.messages)

    def save_history(self):
        """
        将消息的前半部分保存为 JSONL 文件，并按照时间戳命名。
        记录将经过过滤和合并处理。
        """
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建 history 文件夹及对应 session_id 的子文件夹
        history_dir = Path("history") / self.session_id
        history_dir.mkdir(parents=True, exist_ok=True)

        # 只保存前一半的消息
        half_messages = self.messages[:len(self.messages) // 2]
        
        # 处理消息，过滤和合并
        processed_messages = []
        skip_next = False
        for i in range(len(half_messages) - 1):
            if skip_next:
                skip_next = False
                continue
            if half_messages[i]["role"] == "user" and half_messages[i + 1]["role"] == "assistant":
                processed_messages.append({
                    "text": f"User: {half_messages[i]['content']}\n\nAssistant: {half_messages[i + 1]['content']}"
                })
                skip_next = True

        # 创建文件名，以时间戳命名
        filename = history_dir / f"{timestamp}.jsonl"
        
        # 将消息保存为 JSONL 文件
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for message in processed_messages:
                    f.write(json.dumps(message, ensure_ascii=False) + '\n')
            logger.info(f"Saved history to {filename}")
        except Exception as e:
            logger.error(f"Error saving history to {filename}: {str(e)}")
