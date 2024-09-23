import re

def extract_technical_subject(text):
    match = re.search(r'1\.(.*?)，', text)
    if match:
        return match.group(1).strip()
    else:
        return None

# 示例权利要求文本
text = '1.一种基于频数的围标团体提名算法，其特征在于：包括以下步骤：...其他权利要求内容...'


technical_subject = extract_technical_subject(text)
print("技术主题:", technical_subject)

