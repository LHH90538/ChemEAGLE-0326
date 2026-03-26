import json
import os
from main import ChemEagle

image_path = './examples/reaction5.png'
results = ChemEagle(image_path)

print('===============识别结果===============')
print(json.dumps(results, indent=4, ensure_ascii=False))

# 自动用图片名生成 json 文件名
base_name = os.path.splitext(os.path.basename(image_path))[0]
json_path = f'{base_name}_result.json'

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n✅ 结果已保存到：{json_path}")