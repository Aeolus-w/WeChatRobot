import os
import gc
import json
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessor
from typing import Union, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def process_response(output: str, use_tool: bool = False) -> Union[str, dict]:
    content = ""
    for response in output.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            content = content.replace("[[训练时间]]", "2023年")
        else:
            if use_tool:
                content = "\n".join(content.split("\n")[1:-1])
                def tool_call(**kwargs):
                    return kwargs

                parameters = eval(content)
                content = {
                    "name": metadata.strip(),
                    "arguments": json.dumps(parameters, ensure_ascii=False)
                }
            else:
                content = {
                    "name": metadata.strip(),
                    "content": content
                }
    return content

# 定义 Embeddings
embeddings = HuggingFaceEmbeddings(model_name="/root/autodl-tmp/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 向量数据库持久化路径
persist_directory = '/root/autodl-tmp/knowledge_db'

# 加载向量数据库
vectordb = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings
)

similarity_threshold = 13  # 相似性阈值
irrelevance_threshold = 16 # 无关性阈值

@torch.inference_mode()
def generate_stream_chatglm3(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    messages = params["messages"]
    tools = params["tools"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)
    messages = process_chatglm_messages(messages, tools=tools)
    query, role = messages[-1]["content"], messages[-1]["role"]

    # 1. 从文本向量库中查找最接近的4个结果
    search_results_with_scores = vectordb.similarity_search_with_score(query, k=4)
    
    # 提取前4个结果及其相似性分数
    retrieved_contents = [result.page_content for result, _ in search_results_with_scores]
    relevance_scores = [score for _, score in search_results_with_scores]
    
    print(f"找到的相关信息及其相似度为: {relevance_scores}")
    
    # 假设有一个判断信息是否足够的逻辑，这里简单以相似度阈值判断
    if all(score <= similarity_threshold for score in relevance_scores):
        # 信息足够，生成基于这些信息的回复
        combined_knowledge = "\n".join([f"相关信息{idx + 1}: {content}" for idx, content in enumerate(retrieved_contents)])
        combined_input = f"用户提问: {query}\n{combined_knowledge}\n请根据以上信息生成一个自然的回答。"
        
        inputs = tokenizer.build_chat_input(combined_input, history=messages[:-1], role=role)
        inputs = inputs.to(model.device)
        input_echo_len = len(inputs["input_ids"][0])

        if input_echo_len >= model.config.seq_length:
            print(f"Input length larger than {model.config.seq_length}")
        
        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.get_command("<|user|>"),
            tokenizer.get_command("<|observation|>")
        ]
    
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True if temperature > 1e-5 else False,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "logits_processor": [InvalidScoreLogitsProcessor()],
        }
        if temperature > 1e-5:
            gen_kwargs["temperature"] = temperature
    
        total_len = 0
        for total_ids in model.stream_generate(**inputs, eos_token_id=eos_token_id, **gen_kwargs):
            total_ids = total_ids.tolist()[0]
            total_len = len(total_ids)
            if echo:
                output_ids = total_ids[:-1]
            else:
                output_ids = total_ids[input_echo_len:-1]
    
            response = tokenizer.decode(output_ids)
            if response and response[-1] != "�":
                response, stop_found = apply_stopping_strings(response, ["<|observation|>"])
    
                yield {
                    "text": response,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": total_len - input_echo_len,
                        "total_tokens": total_len,
                    },
                    "finish_reason": "function_call" if stop_found else None,
                }
    
                if stop_found:
                    break
    
        ret = {
            "text": response,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
            "finish_reason": "stop",
        }
        yield ret

    elif all(score >= irrelevance_threshold for score in relevance_scores):
        # 完全无关
        response_prefix = f"不好意思，我目前的知识还没有办法解决这个问题，如果您有相关的资料可以一并发给我。"
        yield {
            "text": response_prefix,
            "usage": {
                "prompt_tokens": len(query),
                "completion_tokens": 0,
                "total_tokens": len(query),
            },
            "finish_reason": "stop",
        }
    else:
        # 信息不足够，按序号返回检索到的信息，并告知用户
        numbered_info = "\n".join([f"资料{idx + 1}: {content}" for idx, content in enumerate(retrieved_contents)])
        response_prefix = f"不好意思，我不确定能否完全回答这个问题，以下是我找到的一些资料：\n{numbered_info}"
        
        yield {
            "text": response_prefix,
            "usage": {
                "prompt_tokens": len(query),
                "completion_tokens": 0,
                "total_tokens": len(query),
            },
            "finish_reason": "stop",
        }
    
    gc.collect()
    torch.cuda.empty_cache()


def process_chatglm_messages(messages, tools=None):
    _messages = messages
    messages = []
    msg_has_sys = False
    if tools:
        messages.append(
            {
                "role": "system",
                "content": "Answer the following questions as best as you can. You have access to the following tools:",
                "tools": tools
            }
        )
        msg_has_sys = True

    for m in _messages:
        role, content, func_call = m.role, m.content, m.function_call
        if role == "function":
            messages.append(
                {
                    "role": "observation",
                    "content": content
                }
            )

        elif role == "assistant" and func_call is not None:
            for response in content.split("<|assistant|>"):
                metadata, sub_content = response.split("\n", maxsplit=1)
                messages.append(
                    {
                        "role": role,
                        "metadata": metadata,
                        "content": sub_content.strip()
                    }
                )
        else:
            if role == "system" and msg_has_sys:
                msg_has_sys = False
                continue
            messages.append({"role": role, "content": content})
    return messages


def generate_chatglm3(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    for response in generate_stream_chatglm3(model, tokenizer, params):
        pass
    return response


def apply_stopping_strings(reply, stop_strings) -> Tuple[str, bool]:
    stop_found = False
    for string in stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou: is completed, trim it
        for string in stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found
