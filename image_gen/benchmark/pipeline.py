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


def prompt_optimize(recipe_text: str) -> str:
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

    def generate(self, recipe_text: str ,image_path: Optional[str]=None) -> str:
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
            return image_url
        except Exception as e:
            print(f"图像生成失败: {e}")
            raise e

        

class ImageEvaluator():
    def __init__(self, api_key: str,base_url: str="https://openrouter.ai/api/v1", model_name: str="qwen3-vl-plus"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(timeout=60, connect=15.0),
            max_retries=5,
        )
        self.model=model_name

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _format_image_input(self, image_path: str) -> dict:
        if image_path.startswith(('http://', 'https://')):
            # 直接URL
            return {
                "type": "image_url",
                "image_url": {"url": image_path}
            }
        else:
            base64_image = self._encode_image(image_path)
            return {
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }

    def evaluate_image(self, recipe_text: str, image_path: str) -> dict:    
        evaluation_prompt = f"""
        你是一位专业美食摄影师兼厨师评审。请根据以下菜谱描述和对应的生成图像，从五个维度进行评估，每个维度满分10分，并给出简要理由。

        【菜谱】
        "{recipe_text}"

        【评估维度】
        1. 感知质量：图像是否清晰、逼真、无明显失真或伪影？
        2. 文本-图像一致性：图像是否准确呈现菜谱中提到的外观描述、关键食材、烹饪状态（如颜色、质地）、菜系风格（如中国菜、日本菜等）？
        3. 菜品完整性：图像是否展示了最终的成品菜肴，而不是烹饪过程中的某个步骤或单一的原材料？
        4. 语义合理性：图像中的菜肴是否符合现实烹饪逻辑？（例如：蒸菜不应有焦痕，炖菜应有汤汁）
        5. 摆盘与食品美学：整体摆盘是否协调、美观？色彩搭配与食物造型是否有食欲，是否符合食品美学原则？



        请按以下 JSON 格式输出，严格遵守格式，不要输出其他内容：
        {{
          "感知质量": {{"得分": X, "理由": "..."}},
          "文本-图像一致性": {{"得分": X, "理由": "..."}},
          "菜品完整性": {{"得分": X, "理由": "..."}},
          "语义合理性": {{"得分": X, "理由": "..."}},
          "摆盘与食品美学": {{"得分": X, "理由": "..."}}
        }}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": evaluation_prompt},
                        self._format_image_input(image_path),
                    ],
                }
            ],
            temperature=0.1, # 降低温度以获得更一致的输出格式
        )

        response_text = response.choices[0].message.content
        result = json.loads(response_text)
        result["综合评价"] = {
            "最终得分": 0.1*result["感知质量"]["得分"] + 0.4*result["文本-图像一致性"]["得分"] + 0.1*result["菜品完整性"]["得分"] + 0.2*result["语义合理性"]["得分"] + 0.2*result["摆盘与食品美学"]["得分"],
        }
        return result


class Pipeline:
    """
    协调图像生成和评估的主流程。
    """
    def __init__(self, generator: ImageGenerator, evaluator: ImageEvaluator):
        self.generator = generator
        self.evaluator = evaluator

    def generate_image(self, recipe_text: str, output_image_path: str) -> str:
        return self.generator.generate(recipe_text, output_image_path)

    def evaluate_image(self, recipe_text: str, image_path: str) -> dict:
        evaluation_result = self.evaluator.evaluate_image(recipe_text, image_path)
        return evaluation_result

    def generate_and_evaluate(self, recipe_text: str, output_image_path: str) -> dict:
        # 步骤 1: 生成图像
        generated_image_url = self.generator.generate(recipe_text, output_image_path)

        # 步骤 2: 评估图像
        evaluation_result = self.evaluator.evaluate_image(recipe_text, output_image_path)

        # 步骤 3: 返回结果
        return evaluation_result

# --- 使用示例 ---
if __name__ == "__main__":
    QWEN_API_KEY = os.getenv("QWEN_API_KEY") 
    recipe_path="/data/recipe_generation/recipes_v01/recipes_openai_gpt-5.json"
    image_dir="/home/zhangxiaobin/text2image/image_results/wan-25-preview"      ###########
    evaluation_dir="/home/zhangxiaobin/text2image/evaluation_results_gpt5/wan-25-preview"      #########
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    generator = ImageGenerator(model_name="fal-ai/bytedance/seedream/v4/text-to-image",translate=False,prompt_opz=False)  #######
    evaluator = ImageEvaluator(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1", model_name="openai/gpt-5")  ###
    pipeline = Pipeline(generator, evaluator)

    #获取菜谱
    with open(recipe_path, "r") as f:
        recipes = json.load(f)

    # for i in range(0,5):
    #     recipe = recipes[i]['recipe']
    #     result = pipeline.generate_and_evaluate(recipe, os.path.join(image_dir, f"image_{i}.jpg"))
    #     complete_result={
    #         "index": recipes[i]["index"],
    #         "evaluation": result,
    #         "recipe": recipe,
    #     }

    #     evaluation_file_path = os.path.join(evaluation_dir, f"evaluation_{i}.json")
    #     with open(evaluation_file_path, "w") as f:
    #         json.dump(complete_result, f, indent=2, ensure_ascii=False)


    for i in range(0,20):
        recipe = recipes[i]['recipe']
        result = pipeline.evaluate_image(recipe, os.path.join(image_dir, f"image_{i}.jpg"))
        complete_result={
            "index": recipes[i]["index"],
            "evaluation": result,
            "recipe": recipe,
        }
        evaluation_file_path = os.path.join(evaluation_dir, f"evaluation_{i}.json")
        with open(evaluation_file_path, "w") as f:
            json.dump(complete_result, f, indent=2, ensure_ascii=False)

