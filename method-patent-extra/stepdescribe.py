# -*- coding: utf-8 -*-
import re
def extract_step_descriptions(patent_text):
    # 正则表达式匹配 "所述在步骤*中，" 开始的部分
    pattern = r"所述在步骤(\d+)中，([^。]+)"
    pattern2 = r"所述(S\d+)中([^，]+)，"
    matches = re.findall(pattern, patent_text)

    step_descriptions = {}
    for match in matches:
        step_number, description = match
        step_descriptions[int(step_number)] = description.strip()

    return step_descriptions 



# 示例专利权利要求书文本
patent_text = """
"""

step_descriptions = extract_step_descriptions(patent_text)
for step_number, description in step_descriptions.items():
    print(f"步骤{step_number}具体为：{description}")