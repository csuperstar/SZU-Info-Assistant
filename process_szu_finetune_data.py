import json

# 读取 JSON 文件并加载数据
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 删除包含任何 value 为 null 的字典项，重新排序 id
def process_data(data_list):
    processed_data = []
    
    # 遍历每一项数据
    for data in data_list:
        # 检查 conversations 中是否有任意一项的 value 为 null
        has_null_value = any(conv["value"] is None for conv in data["conversations"])
        
        # 如果没有 null 值，则保留该数据
        if not has_null_value:
            processed_data.append(data)
    
    return processed_data

# 合并两个 JSON 文件的数据并重新编号 id
def merge_and_renumber(json1_data, json2_data):
    # 合并两个列表
    merged_data = json1_data + json2_data
    
    # 重新编号 id
    for idx, data in enumerate(merged_data):
        data["id"] = f"identity_{idx}"
    
    return merged_data

# 保存处理后的数据到新的 JSON 文件
def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 主函数
def main():
    # 输入的两个 JSON 文件路径
    input_file_1 = 'data/finetune/processed_szu_finetune_data.json'
    input_file_2 = 'data/finetune/szu_finetune_data.json'
    
    # 输出的 JSON 文件路径
    output_file = 'data/finetune/processed_szu_finetune_data.json'
    
    # 读取两个文件的数据
    data1 = load_json(input_file_1)
    data2 = load_json(input_file_2)
    
    # 分别处理两个文件：删除包含 value 为 null 的条目
    processed_data1 = process_data(data1)
    processed_data2 = process_data(data2)
    
    # 合并并重新编号
    merged_data = merge_and_renumber(processed_data1, processed_data2)
    
    # 保存处理后的数据
    save_json(output_file, merged_data)

# 运行主函数
if __name__ == "__main__":
    main()
