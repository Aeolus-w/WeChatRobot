# encoding:utf-8

import os
import signal
import sys
import time

from channel import channel_factory
from common import const
from config import load_config
from plugins import *
import threading

from sshtunnel import SSHTunnelForwarder

def sigterm_handler_wrap(_signo):
    old_handler = signal.getsignal(_signo)

    def func(_signo, _stack_frame):
        logger.info("signal {} received, exiting...".format(_signo))
        conf().save_user_datas()
        if callable(old_handler):  #  check old_handler
            return old_handler(_signo, _stack_frame)
        sys.exit(0)

    signal.signal(_signo, func)


def start_channel(channel_name: str):
    channel = channel_factory.create_channel(channel_name)
    if channel_name in ["wx", "wxy", "terminal", "wechatmp", "wechatmp_service", "wechatcom_app", "wework",
                        const.FEISHU, const.DINGTALK]:
        PluginManager().load_plugins()

    if conf().get("use_linkai"):
        try:
            from common import linkai_client
            threading.Thread(target=linkai_client.start, args=(channel,)).start()
        except Exception as e:
            pass
    channel.startup()

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

def run():
    try:
        # load config
        load_config()
        # ctrl + c
        sigterm_handler_wrap(signal.SIGINT)
        # kill signal
        sigterm_handler_wrap(signal.SIGTERM)

        # create channel
        channel_name = conf().get("channel_type", "wx")

        if "--cmd" in sys.argv:
            channel_name = "terminal"

        if channel_name == "wxy":
            os.environ["WECHATY_LOG"] = "warn"

        start_channel(channel_name)

        while True:
            time.sleep(1)
    except Exception as e:
        logger.error("App startup failed!")
        logger.exception(e)


if __name__ == "__main__":
    #create_ssh_tunnel("connect.cqa1.seetacloud.com",28696,"root","Hemcuc1kuD7/","127.0.0.1",6006,6006)
    #create_ssh_tunnel("connect.cqa1.seetacloud.com",33388,"root","Hemcuc1kuD7/","127.0.0.1",6006,6006)
    create_ssh_tunnel("connect.cqa1.seetacloud.com",28218,"root","Hemcuc1kuD7/","127.0.0.1",6006,6006)
    run()
