import os
from fastapi import APIRouter, HTTPException, File, Form, UploadFile, status
from typing import Optional
from utils.knowledge import update_knowledge_db

router = APIRouter()

@router.post("/v1/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(...), file_encoding: Optional[str] = Form("utf-8")):
    """
    上传文件并保存至服务器指定的路径
    :param file: 上传的文件
    :param session_id: 会话 ID，用于创建会话文件夹
    :param file_encoding: 可选，指定文件编码，默认 utf-8
    :return: 返回上传结果消息
    """
    # 构造会话文件夹路径
    session_folder = f"/root/autodl-tmp/base_knowledge/{session_id}"
    os.makedirs(session_folder, exist_ok=True)

    # 保存文件至会话文件夹中
    file_location = f"{session_folder}/{file.filename}"
    try:
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # 确认文件是否成功上传并存在
        if os.path.exists(file_location):
            # 更新知识库
            update_knowledge_db(session_id)
            return {"message": f"文件已成功存储并更新知识库"}
        else:
            raise Exception("文件上传失败，目标文件不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")
