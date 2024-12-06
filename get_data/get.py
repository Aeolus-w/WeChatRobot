import requests
import json
import time  # 导入 time 模块

# 配置 API URL 和密钥
API_URL = "https://api.moonshot.cn/v1/chat/completions"
API_KEY = "sk-6AcEcqlR736Y0T4bY42TR8TdaTOUEpGCGc7SdK1jhsGYc9kP"

# HTTP 请求头
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 读取 JSONL 文件中的问题
def read_questions(file_path):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            question = json.loads(line)
            questions.append(question)
    return questions

# 调用 API 获取回答
def ask_moonshot(question_text):
    payload = {
        "model": "moonshot-v1-8k",
        "messages": [
            {"role": "user", "content": question_text}
        ]
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            # 提取返回的 Assistant 内容
            return response.json().get("messages", [{}])[-1].get("content", "没有回答")
        else:
            print(f"Error {response.status_code}: {response.text}")
            return "没有回答"
    except Exception as e:
        print(f"Error: {e}")
        return "没有回答"

# 更新 JSONL 文件并写入答案
def write_answers_to_file(questions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in questions:
            test_content = question["test"]
            # 提取 User 部分内容
            question_text = test_content.split('User:')[1].split('\n\n')[0].strip()

            # 获取回答
            answer = ask_moonshot(question_text)

            # 填充答案到 Assistant 部分
            updated_test_content = test_content.replace('Assistant:', f'Assistant:{answer}')
            result = {"test": updated_test_content}

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Processed question: {question_text}")

            # 每处理一个问题，暂停 60 秒
            time.sleep(60)  # 等待 60 秒（1 分钟）

# 主函数
def main():
    input_file = "D:\\VSCodeWorkPlace\\WechatRobot\\get_data\\question.jsonl"  # 输入问题文件
    output_file = "D:\\VSCodeWorkPlace\\WechatRobot\\get_data\\question_ans.jsonl"  # 输出带有回答的文件
    questions = read_questions(input_file)
    write_answers_to_file(questions, output_file)
    print(f"Answers have been written to {output_file}.")

if __name__ == "__main__":
    main()
