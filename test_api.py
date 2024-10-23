import requests
import json
from sshtunnel import SSHTunnelForwarder

def create_ssh_tunnel(ssh_host, ssh_port, ssh_username, ssh_password, remote_bind_address, remote_bind_port, local_bind_port):
    # 创建 SSH 隧道连接
    tunnel = SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_username,
        ssh_password=ssh_password,
        remote_bind_address=(remote_bind_address, remote_bind_port),
        local_bind_address=('127.0.0.1', local_bind_port)
    )
    tunnel.start()  # 启动隧道
    print(f"SSH隧道已建立,正在将本地 {local_bind_port} 映射到远程 {remote_bind_address}:{remote_bind_port}")
    return tunnel  # 返回隧道对象

# 定义测试函数，发送消息并验证回复
def test_api_reply(api_url, model_name="chatglm3-6b"):
    headers = {
        "Content-Type": "application/json"
    }

    # 模拟会话消息，这里发送一个简单的问题
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "开发RAG系统面临哪些问题呢?"
            }
        ],
        "temperature": 0.8,
        "top_p": 0.8,
        "max_tokens": 1500,
        "model": model_name
    }

    try:
        # 发送POST请求到API端口
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # 如果请求失败，会抛出异常

        # 解析响应内容
        data = response.json()
        reply_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # 输出回复内容
        print(f"API回复: {reply_content}")
        return reply_content

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

if __name__ == "__main__":
    # 定义API的URL
    api_url = "http://127.0.0.1:6006/v1/chat/completions"
    create_ssh_tunnel("connect.cqa1.seetacloud.com",28696,"root","Hemcuc1kuD7/","127.0.0.1",6006,6006)

    # 调用测试函数，测试API的回复功能
    test_reply = test_api_reply(api_url)
