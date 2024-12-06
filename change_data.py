import json

def convert_json_format(input_file, output_file):
    # 读取原始 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    # 存储转换后的数据
    converted_data = []

    # 逐条转换数据格式
    for entry in data:
        instruction = entry.get("instruction", "")
        output = entry.get("output", "")

        # 构建新的格式
        new_entry = {
            "text": f"User: {instruction}\n\nAssistant: {output}"
        }
        converted_data.append(new_entry)

    # 将转换后的数据写入新 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(converted_data, outfile, ensure_ascii=False, indent=2)

    print(f"转换完成！新数据已保存至 {output_file}")

input_file = './data/raw/self_cognition.json'   # 原始 JSON 文件名
output_file = './data/ok/self_cognition.json' # 转换后 JSON 文件名
convert_json_format(input_file, output_file)
