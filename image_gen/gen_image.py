import os
import json
import openai
from typing import List, Dict, Any, Optional
import fal_client
import requests
from openai import OpenAI
import base64
from dotenv import load_dotenv
import httpx
import time

load_dotenv()


def translate2en(recipe_text: str) -> str:
    prompt = f"""
    将以下菜谱描述翻译成英文。注意只返回英文描述，不要包含任何其他文字。

    【菜谱描述】
    "{recipe_text}"
    """
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": prompt}
        ],
        extra_body={"enable_thinking": False},
    )
    en_recipe = response.choices[0].message.content

    en_prompt=f"""
    Based on the following recipe description, generate a high-quality image of the finished dish. The image should be clear, realistic, and accurately reflect the recipe description. Ensure that the image contains no text. Pay special attention to accurately depicting the key ingredients, cooking state (such as color and texture), plating style, and culinary tradition (e.g., Chinese, Japanese, French, Italian, etc.).

    [Recipe Description]
    "{en_recipe}"
    """

    return en_prompt

def recipe_filter(recipe: Dict[str, Any]) -> Dict[str, Any]|str:
    recipe_text=dict()
    recipe_text['菜名']=recipe['title']
    recipe_text['核心食材']=recipe['ingredients']['core_ingredients']
    recipe_text['其它配料']=recipe['ingredients']['additional_seasonings']

    style = "、".join(recipe['style_select']) + "。"  
    recipe_text['菜系&风格']=style+recipe['ingredients']['cuisine_style']

    recipe_text['制作过程']=dict()
    recipe_text['制作过程']['准备工作']=recipe['preparation']
    recipe_text['制作过程']['烹饪过程']=recipe['cooking_steps']
    recipe_text['制作过程']['tips']=recipe['tips']

    return recipe_text


def prompt_optimize(recipe: Dict[str, Any]) -> str:
    recipe_text=recipe_filter(recipe)

    system_prompt = """
    你是一位厨师和AI提示工程师，擅长将包含步骤描述的菜肴制作过程转换为精准描述菜肴成品外观的的提示词。
    """

    prompt = f"""
    核心任务: 你将收到一份结构化的菜谱文本。你的任务是根据菜谱内容推理出菜肴成品的外观，然后用一段话详细精准地描述出这道菜的外观。
 
    转换规则与思维链: 你必须严格按照以下四个步骤思考，并将所有元素融合成一个连贯的描述菜肴样貌的提示词。
    1. 核心主体: 首先根据菜谱的内容对这道菜的主体进行精准的描述，你需要根据菜谱上的主体材料、烹饪方法等信息，推理出这道菜的主体样貌。描述要确切，避免使用笼统的术语。
    2. 关键配料与细节: 列举菜谱上可见的成分、配料，忽略像糖分、盐等对菜肴外观没有显著影响的元素。比如“铺满厚切的培根”、“金黄的鱿鱼圈”或“层叠的五层酪乳松饼”。
    3. 摆盘与呈现: 根据菜谱的步骤，推理出食物的摆放方式和盛放的容器。
    4. 摄影风格:指定为“中焦拍摄”、“写实风格”
 
    注意:
    1.必须完全遵从菜谱的描述，保持"菜谱-提示词输出"的文本一致性。
    2.使用描绘菜品样貌的描述性文本，尽量避免使用描述行为的文本。
 
    示例：
    1.一款培根奶酪汉堡，第一层为圆形的面包胚，上面撒有黑芝麻，第二层是生菜，第三层是厚切的培根，第四层是融化的陈年切达奶酪，第五层是绿色生菜，第六层也是烤过的黑色芝麻圆形面包；摆放在粗糙的木板上；超写实风格，中焦拍摄。
    2.一款巧克力熔岩蛋糕，融化的黑巧克力从中心流出，蛋糕表面中央点缀以三颗覆盆子和一片薄荷叶，表面均匀轻撒金色糖粉；摆放于洁净的白色瓷盘上；超写实风格，中焦拍摄。

    【菜谱文本】
    "{recipe_text}"
    """
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": system_prompt}
        ],
        extra_body={"enable_thinking": False},
    )
    recipe_opz = response.choices[0].message.content
    
    return recipe_opz


class ImageGenerator():
    def __init__(self, model_name: str="fal-ai/qwen-image",translate: bool=False,prompt_opz: bool=False):
        self.model=model_name
        self.translate=translate
        self.prompt_opz=prompt_opz

    def generate(self, recipe_text: Dict[str, Any] ,image_path: Optional[str]=None) :
        print(f"正在用 {self.model} 生成图像...")
        try:
            if self.prompt_opz:
                recipe_text = prompt_optimize(recipe_text)

            if self.translate:
                prompt = translate2en(recipe_text)
                prompt=prompt[:4500]
                negative_prompt = """
                Any text, including recipe-related text, menus, or text on menus.
                """
            else:
                prompt = f"""
                根据以下菜谱描述，生成一张该菜肴的精美成品图。图像应清晰、逼真、符合菜谱描述,注意图片中不要包含任何文字。特别注意准确呈现菜谱中的关键食材、烹饪状态(如颜色、质地)、摆盘风格和菜系风格(如中国菜、日本菜、法餐、意餐等)。

                【菜谱描述】
                "{recipe_text}"
                """
                
                negative_prompt = """
                任何文字，包括菜谱相关的文字、菜单、菜单上的文字
                """

            results = fal_client.subscribe(
                self.model,
                arguments={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    # "guidance_scale": 3.5,
                    "image_size": "landscape_4_3",
                    "output_format": "png",
                },
            )

            image_url = results["images"][0]["url"]
            if image_path:
                response = requests.get(image_url)
                response.raise_for_status() 
                with open(image_path, "wb") as f:
                    f.write(response.content)

            print(f"图像生成完毕")
            return image_url,recipe_text
        except Exception as e:
            print(f"图像生成失败: {e}")
            raise e

if __name__ == "__main__":
    recipes_path="/data/recipe_generation/recipes_v02_251215/recipes_google_gemini-2.5-pro_style_self_select_nutrition_person_combo.json"  ###
    image_path="/data/recipe_generation/recipes_v02_image/recipes_google_gemini-2.5-pro_style_self_select_nutrition_person_combo/seedream4"   ###
    os.makedirs(image_path, exist_ok=True)

    image_generator = ImageGenerator(model_name="fal-ai/bytedance/seedream/v4/text-to-image",translate=False,prompt_opz=True)

    with open(recipes_path, "r") as f:
        data = json.load(f)
    
    for i in range(0,30):
        recipe = data[i]['recipe']
        image_url,recipe_look=image_generator.generate(recipe, os.path.join(image_path, f"{i}.png"))
        
        prompt_path = os.path.join(image_path, f"{i}_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(recipe_look)

        




         

        

