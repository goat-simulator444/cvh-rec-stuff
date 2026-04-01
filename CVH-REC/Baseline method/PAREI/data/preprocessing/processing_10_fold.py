# 打开原始文件并读取内容
# import config
with open('data/preprocessing/10_fold_new.txt', 'r') as file:
    lines = file.readlines()

# 处理每一行，将每个数字加940
processed_lines = []
for line in lines:
    # 去掉行首的 '[' 和行尾的 ']\n'
    line = line.strip().strip('[]')
    # 将字符串分割为数字列表
    numbers = [int(num) for num in line.split(', ')]
    # 对每个数字加940
    updated_numbers = [num + 1382 for num in numbers]
    # 将更新后的数字列表重新拼接为字符串，并添加到结果列表中
    processed_lines.append('[' + ', '.join(map(str, updated_numbers)) + ']')

# 将处理后的结果写入到新的文件
with open('data/10_fold_numbered_new.txt', 'w') as file:
    for line in processed_lines:
        file.write(line + '\n')

print("处理完成，结果已保存。")