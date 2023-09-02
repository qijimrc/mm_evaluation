import io
from PIL import Image

PROMPT_EN = "Please choose the correct option for the above question from the following options: "
PROMPT_ZH = "请从以下几个选项中选出上述问题的正确答案："

def generate_prompt_in_multi_choice(choices, question, language="zh"):
    prompt = question + "\n" + (PROMPT_ZH if language == "zh" else PROMPT_EN) + "\n"
    start_op = 'A'
    for item in choices:
        prompt += f'{start_op}: {item}\n'
        start_op = chr(ord(start_op) + 1)
    return prompt

def get_image_bytes(image_path):
    img = Image.open(image_path).convert('RGB')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="jpeg")
    return img_bytes.getvalue()
    