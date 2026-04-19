import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer


# ----------------------------------------- 合成提示词 -----------------------------------------

SYSTEM_PROMPT = """你是一个专业的**JSON修复/修改**训练数据生成器。你的任务是生成用于训练小型语言模型（100M-200M参数）的高质量数据。你生成的每条样本都必须保证**答案唯一**，不能出现多种合理修复方式。

你的输出目标：
- 一次生成 {batch} 条数据
- 直接输出 **JSON数组**
- 不要输出任何额外说明、注释
- 每条数据必须包含且仅包含以下三个字段：
  - "prompt"
  - "thinking"
  - "response"

---

## 一、样本格式要求

每条样本格式如下：

{{
  "prompt": "...",
  "thinking": "...",
  "response": "```json\\n...\\n```"
}}

字段要求：

1. prompt
- 表达用户要你**修复**一段有错误的JSON，或根据明确规则对JSON做**修改**
- 表述必须自然、多样，不要总是重复同一种开头
- prompt中必须包含待修复/待修改的原始JSON文本
- 若任务涉及类型修复、字段删除、字段筛选、值替换、数组处理等，prompt中必须给出**足够明确的规则**，确保最终答案唯一
- 不允许生成规则模糊、会导致多解的prompt

2. thinking
- 用中文，总体上模拟DeepSeekR1的思考风格
- 长度控制在50-150个字符
- 风格为渐进式、自述式思考
- 思维过程按照以下步骤展开：
  1. 先明确用户想要什么，例如"用户想修改JSON中的字段..."、"这是一个关于JSON修复的问题"
  2. 对问题本身进行简要分析，例如当前JSON结构是什么、存在哪些问题等等
  3. 表明自己打算如何按照用户的需求修复或修改JSON内容，例如"我需要删除字段X，保留字段Y"等等
- 不要出现“我不能展示推理过程”之类的话
- 不要枚举很多候选答案
- 不要凭空补充response中没有的信息
- 不要以上帝视角机械列出所有错误，尽量表现为逐步检查和收敛

3. response
- 只能直接给出最终正确结果
- 必须使用如下格式：
  ```json
  {{合法JSON}}
  ````

* response中的JSON必须能被标准JSON解析器正确解析
* 不要附带解释，不要附带多余文本

---

## 二、唯一答案原则

你生成的每条数据都必须满足：**答案唯一**。

必须严格遵守以下规则：

1. 只生成那些在给定约束下**只有一种合理答案**的样本
2. 如果某条样本存在多种合理修复方式，**直接放弃，不要生成**
3. 修复时必须遵循**最小编辑原则**
4. 所谓最小编辑原则，指：

   * 优先保留原有键名
   * 优先保留原有值
   * 优先保留原有字段顺序
   * 优先保留原有整体结构
   * 只做让结果成为合法JSON或满足用户明确修改要求所必需的最小改动

5. 不允许凭语义猜测缺失内容
6. 不允许脑补新的字段、新的值、新的数组项、新的对象结构
7. 不允许因为“看起来更合理”就改动原始值
8. 如果需要做类型转换、字段删除、去重、替换、筛选，prompt里必须明确写出规则
9. 凡是类型修复任务，必须在prompt中明确目标类型或schema约束，否则不要生成
10. 凡是删除/保留类任务，必须在prompt中明确匹配条件，否则不要生成
11. 凡是重复键问题，统一采用固定规则：

* **保留最后一次出现的值**
* 删除前面重复的同名键

12. 如果坏JSON已经严重残缺，无法在不猜测语义的前提下唯一修复，则不要生成该样本

---

## 三、允许生成的任务类型

你只能生成两大类任务：

A. 语法修复类
这类任务只修复JSON语法，不改变原始语义。必须满足唯一可修复。

适合的错误类型包括：

* 键名未加双引号
* 使用单引号而不是双引号
* 布尔值或null写成 True / False / None
* 多余逗号，尤其尾随逗号
* 明确缺少一个逗号，且插入位置唯一
* 明确缺少一个冒号，且插入位置唯一
* 多余注释混入JSON（// 或 /* */）
* 外层缺少一个明显可唯一补全的花括号或方括号
* 局部存在明显且唯一的引号不匹配，可最小修改修复

B. 规则修改类
这类任务不是纯语法修复，而是根据用户明确要求修改JSON。规则必须明确，保证唯一答案。

适合的修改类型包括：

* 删除指定字段
* 仅保留指定字段
* 将指定字段改为指定值
* 将指定字段转为明确类型（如整数、浮点数、布尔值、null、字符串）
* 删除值为null的字段
* 删除值满足明确条件的字段
* 按明确规则去重数组，且说明保持首次出现顺序或末次出现顺序
* 修改某个配置项、状态值、日期字符串、端口号等明确字段

---

## 四、禁止生成的样本类型

以下样本即使看起来像“修复任务”，也不要生成，因为容易多解：

1. 缺失一个值，但prompt没有明确默认值或目标值
2. 缺失整个键值对
3. 截断得很严重的JSON
4. 截断字符串且无法唯一判断内容
5. 括号/方括号缺失后存在多种配对可能
6. 类型“看起来不对”但prompt没有明确schema
7. 同一个错误可以通过“删除”或“补全”两种方式解决
8. 依赖语义猜测才能确定正确答案的样本
9. “帮我修一下这个JSON”但原文有多种合理修法
10. 修改规则描述模糊，如“删掉不合适的字段”“修正一下类型”“清理一下内容”
11. 数值和字符串都合法，但prompt没说明必须是哪种
12. 中文、英文、数字混合内容的匹配规则不明确的删除类任务

---

## 五、内容复杂度要求

1. JSON复杂度适中
2. 不要过多嵌套，最多2层
3. 键值对数量控制在3到8个之间
4. 内容尽量贴近真实场景，例如：

   * 用户资料
   * 商品信息
   * 订单
   * 配置文件
   * 日程
   * 设备信息
   * 接口参数
   * 代码配置
   * 简单日志结构

5. 不要生成超长文本值
6. 不要生成特别冷门或无意义字段名
7. 尽量让不同样本的场景多样

---

## 六、字段与结果约束

1. response中的JSON必须是**标准JSON**
2. JSON中键名必须全部使用双引号
3. 字符串必须使用双引号
4. 不允许出现注释
5. 不允许出现尾随逗号
6. 布尔值必须是 true / false
7. 空值必须是 null
8. 数字必须符合JSON标准
9. response中字段顺序尽量保持与原始JSON一致；若删除字段，则保留剩余字段的原始相对顺序
10. 修改类任务中，除非prompt明确要求，否则不要额外美化、重命名、重排、归一化其他字段

---

## 七、多样性要求

1. prompt开头要多样化，例如但不限于：

   * “把这段JSON修正成合法格式：...”
   * “下面这个JSON有格式问题，帮我改好：...”
   * “请按要求调整这份JSON：...”
   * “这个配置JSON解析失败了，请修复：...”
   * “将下面JSON中的 enabled 改成布尔值，其余不变：...”
   * “只保留 name、price、stock 三个字段：...”

2. 不要反复使用同一类错误
3. 不要让所有样本都来自同一场景
4. 修复类与修改类都要覆盖
5. 尽量让不同样本之间的错误模式、字段名、场景不同

---

## 八、自检要求

每生成一条数据前，你都必须先隐式检查：

1. 这个prompt对应的答案是否唯一？
2. 是否不需要依赖语义猜测？
3. 是否满足最小编辑原则？
4. response是否是合法JSON？
5. response是否完全符合prompt要求？
6. thinking是否简洁、自然、没有发散？
7. 是否没有多余说明？

如果任一项不满足，就放弃该条并重新生成。

---

## 九、输出要求

最终只输出JSON数组本身，不要输出任何解释，不要输出前言，不要输出“下面是结果”，不要使用markdown包裹整个数组。

输出格式示例：

```json
[
    {{
        "prompt": "把这段JSON修正成合法格式：{{name: 'Tom', age: 18, active: True}}",
        "thinking": "用户让我修复原始JSON，让我检查一下它存在的问题。原始JSON存在三处非法：键名未加双引号，字符串值用了单引号，布尔值True首字母大写。因此，我需要将其修正为双引号键名、双引号字符串、小写true。",
        "response": "`json\\n{{\\"name\\": \\"Tom\\", \\"age\\": 18, \\"active\\": true}}\\n`"
    }},
    {{
        "prompt": "请按要求修改这段JSON：删除所有值为null的字段，其余内容保持不变。JSON如下：{{\"title\": \"会议纪要\", \"owner\": null, \"status\": \"done\", \"pages\": 3}}",
        "thinking": "我需要根据用户的要求修改原始JSON。识别字段owner值为null，需删除；保留title、status、pages三个非空字段，我需要输出标准JSON格式，无多余逗号。",
        "response": "`json\\n{{\\"title\\": \\"会议纪要\\", \\"status\\": \\"done\\", \\"pages\\": 3}}\\n`"
    }},
    ...
]
```
"""

GENERAL_SYSTEM_PROMPT = """你是一个专业的**通用指令遵循**训练数据生成器。你的任务是生成用于训练小型语言模型（100M-200M参数）的高质量冷启动对话数据。

你的输出目标：
- 一次生成 {batch} 条数据
- 直接输出 **JSON数组**
- 不要输出任何额外说明、注释
- 每条数据必须包含且仅包含以下三个字段：
  - "prompt"
  - "thinking"
  - "response"

---

## 一、样本格式要求

每条样本格式如下：

{{
  "prompt": "...",
  "thinking": "...",
  "response": "..."
}}

字段要求：

1. prompt
- 模拟真实用户向AI助手提出的日常问题或指令
- 表述自然、口语化，长度适中（一两句话即可）
- 不要每条都用相同句式开头，保持多样性

2. thinking
- 用中文，模拟DeepSeekR1的思考风格
- 长度严格控制在50-150个**字**之间，不要过短
- 思维过程按照以下三步展开：
  1. 先明确用户想要什么，例如"用户想了解..."、"这是一个关于...的问题"
  2. 对问题本身进行简要分析，例如涉及哪些要点、需要从哪个角度切入
  3. 表明自己打算如何组织回答，例如"我可以从...方面来回答"、"我先说...再说..."
- 风格为渐进式、自述式思考，自然衔接，不要机械地标注步骤编号，不要一开始就给出结论或答案
- 不要出现"我不能展示推理过程"之类的话

3. response
- 直接给出对用户问题的回答
- 回答要准确、实用、简洁
- 不要附带多余的客套话或重复用户的问题

**重要：每条数据的 prompt + thinking + response 总长度控制在500个中文字符以内。**

---

## 二、任务类型覆盖

生成的数据应覆盖以下日常对话场景，每批数据尽量多样：

1. **知识问答**：常识、科学、历史、地理、文化等
2. **生活建议**：健康、饮食、运动、旅行、购物等
3. **工作实用**：邮件写作、会议总结、报告润色、时间管理等
4. **解释说明**：解释概念、术语、原理、流程等
5. **创意写作**：取名、写祝福语、编故事开头、写短文案等
6. **逻辑推理**：简单数学、排序、比较、因果推理等
7. **翻译改写**：中英互译、改写句子、换种说法表达等
8. **观点建议**：推荐、对比、选择建议、优缺点分析等

---

## 三、质量要求

1. response必须事实正确，不要编造虚假信息
2. 如果是知识类问题，回答要准确可靠
3. 如果是建议类问题，回答要实用具体
4. 不要生成敏感、违规、有争议的内容
5. 不要生成涉及个人隐私的内容
6. thinking要与response逻辑一致，不要出现thinking说一套、response做另一套的情况

---

## 四、多样性要求

1. 每批数据中，不同任务类型至少覆盖3种以上
2. prompt的句式要多样，不要总是"请问..."、"帮我..."开头
3. 场景要多样：学生、上班族、家长、旅行者、创作者等不同角色视角
4. 难度要有梯度：有简单的也有稍复杂的

---

## 五、输出要求

最终只输出JSON数组本身，不要输出任何解释，不要输出前言，不要输出"下面是结果"，不要使用markdown包裹整个数组。

输出格式示例：

```json
[
    {{
        "prompt": "为什么天空是蓝色的？",
        "thinking": "用户想了解天空为什么是蓝色的，这是一个经典的光学问题。核心原理是瑞利散射，太阳光中波长较短的蓝光更容易被大气分子散射到各个方向。我可以先说原理再用通俗的方式解释现象，让回答简洁易懂。",
        "response": "天空呈蓝色是因为瑞利散射现象。太阳光包含各种颜色，当它穿过大气层时，波长较短的蓝光比其他颜色更容易被空气分子散射到各个方向，所以我们从各个角度看天空都会看到蓝色。"
    }},
    {{
        "prompt": "推荐三个适合周末去的上海景点",
        "thinking": "用户想找适合周末去的上海景点，这是一个本地出游推荐类问题。上海景点很多，我应该选不同类型的来覆盖不同偏好。我打算推荐三个分别偏文化、自然和文艺的景点，并简要说明各自亮点。",
        "response": "1. 外滩——经典的万国建筑群和黄浦江夜景；2. 上海自然博物馆——适合全家，展品丰富互动性强；3. 田子坊——文艺小店和特色美食集中，适合慢逛拍照。"
    }},
    ...
]
```
"""


# ----------------------------------------- 核心逻辑 -----------------------------------------

def load_existing_data(output_path: str) -> int:
    """加载已有数据的条数, 用于断点续跑"""
    path = Path(output_path)
    if not path.exists():
        return 0

    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def extract_json_from_response(response: str) -> str | None:
    """从 response 中提取 ```json ... ``` 包裹的 JSON 字符串"""
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def validate_item(item: dict, task: str = "json_repair") -> bool:
    """校验单条数据是否合法, json_repair 任务额外验证 response 中的 JSON 可解析"""
    if not isinstance(item, dict):
        return False
    for key in ("prompt", "thinking", "response"):
        if key not in item or not isinstance(item[key], str) or not item[key].strip():
            return False

    if task == "json_repair":
        json_str = extract_json_from_response(item["response"])
        if json_str is None:
            return False
        try:
            json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return False

    return True


TASK_PROMPTS = {
    "json_repair": SYSTEM_PROMPT,
    "general": GENERAL_SYSTEM_PROMPT,
}


async def generate_grpo_data(
    output_path: str,
    task: str = "json_repair",
    batch: int = 10,
    num_samples: int = 10000,
    async_num: int = 5,
):
    """
    调用 OpenAI 兼容 API 合成 GRPO 训练数据（JSON修复任务）

    Args:
        output_path (str): 输出的 jsonl 文件路径
        task (str): 任务类型, json_repair 或 general
        batch (int): 每次请求生成的样本数量
        num_samples (int): 总共需要生成的样本数量
        async_num (int): 并发请求数量
    """
    import json_repair

    system_prompt = TASK_PROMPTS[task]

    # 加载环境变量
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_id = os.getenv("OPENAI_MODEL_ID")

    if not api_key or not base_url or not model_id:
        print("Error: OPENAI_API_KEY, OPENAI_BASE_URL, or OPENAI_MODEL_ID is not set")
        print("Please configure them in the .env file")
        return

    # 禁用 SSL 验证
    http_client = httpx.AsyncClient(verify=False)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    # 断点续跑：统计已有数据条数
    current_count = load_existing_data(output_path)

    if current_count >= num_samples:
        print(f"Already have {current_count} samples, target is {num_samples}. Nothing to do.")
        return

    if current_count > 0:
        print(f"Resuming from {current_count} existing samples...")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 定义单次请求, 带重试确保每批严格返回 batch 条有效数据
    max_retries = 5

    async def fetch_batch() -> list[dict]:
        valid_items = []
        retries = 0

        while len(valid_items) < batch and retries < max_retries:
            need = batch - len(valid_items)
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": system_prompt.format(batch=need)}],
                    temperature=1.0,
                    max_tokens=8000,
                )
                content = response.choices[0].message.content

                # 提取JSON数组
                json_match = re.search(r"\[.*\]", content, re.DOTALL)
                if json_match:
                    batch_data = json_repair.loads(json_match.group(0))
                else:
                    batch_data = json_repair.loads(content)

                if isinstance(batch_data, list):
                    for item in batch_data:
                        if validate_item(item, task) and len(valid_items) < batch:
                            valid_items.append(item)

            except Exception as e:
                print(f"\nError in API request: {e}")

            retries += 1

        return valid_items

    # 以追加模式写入, 支持断点续跑
    f = open(output_path, "a", encoding="utf-8")
    total_written = current_count

    try:
        with tqdm(total=num_samples, initial=current_count, desc="Generating GRPO data") as pbar:
            while total_written < num_samples:
                remaining = num_samples - total_written
                n_tasks = min(async_num, (remaining + batch - 1) // batch)
                tasks = [fetch_batch() for _ in range(n_tasks)]

                for coro in asyncio.as_completed(tasks):
                    results = await coro
                    written_this_batch = 0
                    for item in results:
                        if total_written >= num_samples:
                            break
                        item["sample_id"] = total_written
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        total_written += 1
                        written_this_batch += 1
                    pbar.update(written_this_batch)
                    f.flush()
    finally:
        f.close()

    print("-" * 30)
    print(f"GRPO data generation completed!")
    print(f"Total samples: {total_written:,}")
    print(f"Data saved to: {output_path}")
    print("-" * 30)


# ---------------------------------------- 转换为 bin ----------------------------------------

def convert_to_bin(jsonl_path: str, bin_path: str, tokenizer_path: str) -> None:
    """
    将 GRPO jsonl 数据转换为 .bin 文件, 用于 mid training 继续预训练

    拼接方式: prompt + thinking + response + eos_token, 逐条分词后写入二进制文件

    Args:
        jsonl_path (str): 输入的 jsonl 文件路径
        bin_path (str): 输出的 bin 文件路径
        tokenizer_path (str): tokenizer 路径
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    eos_token = tokenizer.eos_token

    Path(bin_path).parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_samples = 0
    buffer: list[int] = []
    buffer_size = 1_000_000

    with open(jsonl_path, "r", encoding="utf-8") as f_in, \
         open(bin_path, "wb") as f_out:

        for line in tqdm(f_in, desc="Converting to bin"):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 拼接: prompt + thinking + response + eos_token
            text = item["prompt"] + item["thinking"] + item["response"] + eos_token
            token_ids = tokenizer.encode(text)
            buffer.extend(token_ids)
            total_tokens += len(token_ids)
            total_samples += 1

            # 缓冲区满则写入
            while len(buffer) >= buffer_size:
                arr = np.array(buffer[:buffer_size], dtype=np.uint16)
                arr.tofile(f_out)
                buffer = buffer[buffer_size:]

        # 写入剩余数据
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(f_out)

    print("-" * 30)
    print(f"Bin conversion completed!")
    print(f"Total samples: {total_samples:,}, Total tokens: {total_tokens:,}")
    print(f"Data saved to: {bin_path}")
    print("-" * 30)


# ---------------------------------------- 合并 jsonl ----------------------------------------

def merge_jsonl(input_files: list[str], output_path: str, shuffle: bool = False) -> None:
    """将多个 jsonl 文件合并为一个, 可选打乱顺序"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for file_path in input_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(line.strip())

    if shuffle:
        np.random.shuffle(lines)

    for i, line in enumerate(lines):
        item = json.loads(line)
        item["sample_id"] = i
        lines[i] = json.dumps(item, ensure_ascii=False)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("-" * 30)
    print(f"Merge completed!")
    print(f"Input files: {len(input_files)}, Total samples: {len(lines):,}")
    print(f"Shuffle: {shuffle}")
    print(f"Data saved to: {output_path}")
    print("-" * 30)


# ----------------------------------------- 入口 -----------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Generate GRPO training data")
    parser.add_argument("--task", type=str, default="json_repair", choices=["json_repair", "general"], help="Task type: json_repair or general")
    parser.add_argument("--output", type=str, default=None, help="Output jsonl file path")
    parser.add_argument("--batch", type=int, default=10, help="Samples per API call")
    parser.add_argument("--num_samples", type=int, default=10000, help="Total samples to generate")
    parser.add_argument("--async_num", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--to_bin", action="store_true", help="Convert jsonl to bin for mid training")
    parser.add_argument("--bin_input", type=str, default=str(root / "data/grpo_data/grpo.jsonl"), help="Input jsonl file for bin conversion")
    parser.add_argument("--bin_output", type=str, default=str(root / "data/grpo_data/grpo.bin"), help="Output bin file path")
    parser.add_argument("--merge", nargs="+", type=str, default=None, help="Merge multiple jsonl files into one, e.g. --merge a.jsonl b.jsonl")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples when merging")
    args = parser.parse_args()

    default_outputs = {
        "json_repair": str(root / "data/grpo_data/grpo.jsonl"),
        "general": str(root / "data/grpo_data/general.jsonl"),
    }

    if args.merge:
        output_path = args.output or str(root / "data/grpo_data/cold_start.jsonl")
        merge_jsonl(args.merge, output_path, shuffle=args.shuffle)
    elif args.to_bin:
        tokenizer_path = str(root / "mini_tokenizer")
        convert_to_bin(args.bin_input, args.bin_output, tokenizer_path)
    else:
        output_path = args.output or default_outputs[args.task]
        asyncio.run(generate_grpo_data(
            output_path=output_path,
            task=args.task,
            batch=args.batch,
            num_samples=args.num_samples,
            async_num=args.async_num,
        ))


if __name__ == "__main__":
    main()
