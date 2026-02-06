from logging import critical

import requests
import asyncio
import httpx
from openai import AsyncOpenAI
import pytest
import uuid
import json
import re
from deepeval.models.gpt_model import GPTModel
from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

requests.packages.urllib3.disable_warnings()

API_BASE_URL = "https://api.samsungiotcloud.cn"
PANDA_URL = "http://120.26.206.98:8001/stream"
LOCATION_ID = "f4b3af92-5826-416e-8e28-8b1c252912f1"
PAT = "c45ce62d-b7e1-41f8-a86f-6d6d23012272"
API_HEADERS = {
    "Authorization": f"Bearer {PAT}",
    "Content-Type": "application/json"
}

# 加载该位置下的设备
devices = []
response = requests.get(
    url=f"{API_BASE_URL}/devices?location={LOCATION_ID}",
    headers=API_HEADERS,
    verify=False
)
if response.status_code == 200:
    devices = response.json().get("items", [])
# 加载该位置下的房间
rooms = {}
response = requests.get(
    url=f"{API_BASE_URL}/locations/{LOCATION_ID}/rooms",
    headers=API_HEADERS,
    verify=False
)
if response.status_code == 200:
    rooms = {room.get("roomId"): room.get("name") for room in response.json().get("items", [])}

for device in devices:
    device_id = device.get("deviceId")
    room_id = device.get("roomId")
    room_name = rooms.get(room_id)
    device["roomName"] = room_name

    # 设备在离线信息查询
    response = requests.get(f"{API_BASE_URL}/devices/{device_id}/health", headers=API_HEADERS, verify=False)
    if response.status_code != 200:
        print(f"Failed to get health for device {device_id}")
    try:
        response_json = response.json()
        device["state"] = response_json.get("state")
    except requests.exceptions.JSONDecodeError:
        print(f"Could not decode health response for device {device_id}")

QWEN_API_KEY = "sk-62c3f30ff4764eb9b3e1dc94bac59530"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CASE_MODEL = "qwen3-max"
JUDGE_MODEL = "qwen-flash"

http_client = httpx.AsyncClient(verify=False)
openai_client = AsyncOpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    http_client=http_client
)


async def get_case_response(prompt: str):
    completion = await openai_client.chat.completions.create(
        model=CASE_MODEL,
        temperature=0.5,
        messages=[{
            "role": "system",
            "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    data = completion.choices[0].message.content
    return data


def get_streaming_response(json_data, ssl_verify=False):
    full_content = []

    with requests.post(PANDA_URL, json=json_data, stream=True, verify=ssl_verify) as r:
        for line in r.iter_lines():
            if line:
                decoded = line.decode('utf-8').strip()
                if decoded.startswith("data:"):
                    content = decoded[5:].strip()
                    if content and not (content.startswith("{") and "error" in content):
                        full_content.append(content)

    text = "".join(full_content)
    return text


async def load_device_control_cases():
    SMART_HOME_TEST_PROMPT = f"""
        # 角色
        智能家居测试用例生成引擎（严格遵循设备元数据）

        # 输入规范
        用户提供设备列表（JSON数组），每个设备必须包含：
        - label: 设备名称（字符串，如"客厅主灯"）
        - status: 状态（"online"/"offline"）
        - capabilities: 能力集合（字符串数组，如["switch", "brightness"]）

        # 处理规则
        1. 筛选 status=="online" 的设备，取前5个（不足则全取）
        2. 对每个设备生成2个用例（先"打开"后"关闭"）
        3. 【关键】capability判定（仅输出"switch"或"windowShade"）：
           a) 检查 capabilities 数组（不区分大小写）：
              • 优先：存在 "switch" → capability="switch"
              • 次选：存在 "windowShade" → capability="windowShade"
        4. command生成：
           - 打开： "打开设备"
           - 关闭： "关闭设备"
           （保留原始中文名称，如"打开阳台电动窗帘"）

        # 输出要求
        - 仅输出标准JSON数组，无任何额外文本
        - 每个对象严格包含：
          {{
            "id": "ai_test_X",       // X从1开始全局递增
            "deviceId":设备id
            "command": "自然语言命令",
            "capability": "switch" 或 "windowShade"
          }}
        - JSON语法校验：双引号、无尾逗号、中文不转义
        - 按设备筛选顺序生成（先设备1的开/关，再设备2的开/关...）

        # 设备列表
        {devices}
        """

    return await get_case_response(SMART_HOME_TEST_PROMPT)


# 1. 使用 asyncio.run 获取结果
# 2. 使用 json.loads 解析字符串
raw_json = asyncio.run(load_device_control_cases())
cases_data = json.loads(raw_json)

# 3. 准备满足 parametrize 格式的数据
# 注意：parametrize 需要一个元组列表 [(id1, devId1, cmd1, cap1), ...]
cases = [
    (c["id"], c["deviceId"], c["command"], c["capability"])
    for c in cases_data
]
ids = [c["id"] for c in cases_data]

custom_model = GPTModel(
    model_name=JUDGE_MODEL,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    http_client=http_client
)

DEVICE_CONTROL_METRIC = GEval(
    name="CheckDeviceControlAgentOutput",
    criteria="""
        判定实际状态（ACTUAL_OUTPUT）与预期目标（EXPECTED_OUTPUT）在物理意义上是否完全一致：
        1. 语义映射：
           - 'on' 或 'open' 对应“打开”、“开启”、“运行中”等描述。
           - 'off' 或 'close' 对应“关闭”、“停止”、“断开”等描述。
        2. 核心逻辑：
           - 忽略句式差异（如“电视已打开” vs “打开电视”）。
           - 只要 ACTUAL_OUTPUT 的机器状态能够满足 EXPECTED_OUTPUT 描述的最终状态，即视为一致（Pass）。
           - 如果状态矛盾（例如 ACTUAL 为 'off' 但 EXPECTED 为 '已打开'），则视为不一致（Fail）。
        """,
    threshold=0.7,
    model=custom_model,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "id,deviceId,command,capability",
    cases,
    ids=ids
)
async def test_device_control(id: str, deviceId: str, command: str, capability: str):
    # 调用线上接口,获取期望值
    params = {"prompt": command, "location_id": LOCATION_ID, "user_token": PAT, "request_id": str(uuid.uuid4())}
    expected_output = get_streaming_response(params)
    # 调用st接口,获取实际值
    # 在这里调用其他接口，例如获取设备状态
    device_status_url = f"{API_BASE_URL}/devices/{deviceId}/status"
    status_response = requests.get(
        url=device_status_url,
        headers=API_HEADERS,
        verify=False
    )
    actual_output = ""
    if status_response.status_code == 200:
        data = status_response.json()
        # 逐层安全获取，给定默认值 {} 防止 NoneType 报错
        main_component = data.get("components", {}).get("main", {})
        cap_data = main_component.get(capability, {}).get(capability, {})

        # 最终取到 value，如果没有则默认为空字符串或 None
        actual_output = cap_data.get("value", "")
    else:
        print(f"Failed to get status for device {deviceId}")

    test_case = LLMTestCase(
        input=command,
        actual_output=actual_output,
        expected_output=expected_output,
        context=[],
        name=id
    )
    assert_test(test_case, metrics=[DEVICE_CONTROL_METRIC])
