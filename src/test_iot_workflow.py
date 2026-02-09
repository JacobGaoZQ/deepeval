from logging import critical

import requests
import asyncio
import httpx
from openai import AsyncOpenAI
import pytest
import uuid
import json
import re
from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from JudgeLlmClient import JudgeLlmClient

requests.packages.urllib3.disable_warnings()

API_BASE_URL = "https://api.samsungiotcloud.cn"
PANDA_URL = "http://120.26.206.98:8001/stream"
LOCATION_ID = "f4b3af92-5826-416e-8e28-8b1c252912f1"
PAT = "330cfc68-a2f2-4bf5-872b-40019fcc7c23"
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


async def load_device_control_cases(prompt: str):
    return await get_case_response(prompt)


# 单设备控制case初始化
DEVICE_TEST_PROMPT = f"""

# 角色
智能家居测试用例生成引擎

# 输入规范
用户提供设备列表（JSON数组），每个设备包含：
- label: 设备名称
- roomName: 房间名称（需整合进命令）
- status: 状态（"online"/"offline"）
- capabilities: 能力集合

# 处理规则
1. **设备筛选**：仅处理 `status == "online"` 的设备，按顺序取前5个。
2. **命令组合逻辑**：
   - 格式：`[动作][房间名][设备名]`
   - 动作：打开 / 关闭
   - 示例：若 roomName="客厅", label="主灯" -> "打开客厅主灯"
3. **能力（Capability）判定逻辑**（严格二选一）：
   - 若 `capabilities` 包含 "switch"（忽略大小写） -> `capability = "switch"`
   - 否则，若包含 "windowShade"（忽略大小写） -> `capability = "windowShade"`
   - 若两者皆无，则跳过该设备。
4. **用例生成数量**：每个入选设备生成 2 个用例（先“打开”，后“关闭”）。

# 输出要求
- **仅输出标准 JSON 数组**，禁止包含任何 Markdown 格式块、解释性文字或引言。
- 字段定义：
  - `id`: 字符串 "ai_test_X"（X 为从 1 开始的全局自增数字）
  - `deviceId`: 保持输入中的原始 deviceId（若输入无此字段则设为 null）
  - `command`: 结合房间名与设备名的自然语言命令
  - `capability`: 仅限 "switch" 或 "windowShade"
- **语法约束**：双引号包裹键值，中文不转义，严禁尾部逗号。

# 设备列表
{devices}
        """
raw_json = asyncio.run(load_device_control_cases(DEVICE_TEST_PROMPT))
cases_data = json.loads(raw_json)

# 3. 准备满足 parametrize 格式的数据
# 注意：parametrize 需要一个元组列表 [(id1, devId1, cmd1, cap1), ...]
cases = [
    (c["id"], c["deviceId"], c["command"], c["capability"])
    for c in cases_data
]
ids = [c["id"] for c in cases_data]

# 多设备控制case初始化
MULTI_DEVICES_TEST_PROMPT = f"""
你是一个智能家居设备测试用例生成器。请根据用户提供的 **设备列表（JSON格式）**，为所有满足条件的在线设备生成**1个**多设备开关控制相关的测试用例，并严格按照以下结构化格式输出。

### 设备筛选与能力判断规则：
- 仅考虑 `state == "online"` 的设备。
- 仅包含 `capabilities` 列表中包含 `"switch"` 或 `"windowShade"` 的设备（这两种能力支持“打开/关闭”语义）。
- 对每个入选设备，提取字段：`deviceId`、`label`、`roomName`（可选）、`capabilities`。

### 能力类型映射：
- 若 `capabilities` 包含 `"switch"` → `capability = "switch"`
- 若 `capabilities` 包含 `"windowShade"` 但不包含 `"switch"` → `capability = "windowShade"`

### 每个设备操作的 `command` 字段生成规则：
- 如果设备有 `roomName` 字段：
  - 打开：`"打开roomName的label"`
  - 关闭：`"关闭roomName的label"`
- 如果没有 `roomName` 字段：
  - 打开：`"打开label"`
  - 关闭：`"关闭label"`

### 测试用例要求：
1. 必须包含 **至少两个** 符合上述条件的设备。
2. 操作可以是同时打开、同时关闭、或混合（部分开、部分关）。
3. 顶层 `command` 字段必须是一个自然语言句子，清晰描述整个多设备操作，使用“并”、“同时”、“先…再…”等连接词。
4. `steps` 是一个列表，每个元素包含：
   - `deviceId`（字符串）
   - `capability`（"switch" 或 "windowShade"）
   - `command`（对应设备的自然语言指令，如上所述）

### 输出格式：
- 输出必须是 **一个 JSON 数组**，仅包含 **1 个** 测试用例对象。
- 不包含任何额外文本、注释或 Markdown。
- 字段顺序不限，但必须包含 `id`、`command`、`steps`。

### 示例输出：
[
  {{
    "id": "ai_test_1",
    "command": "关闭客厅的主灯并打开卧室的窗帘",
    "steps": [
      {{
        "deviceId": "light_01",
        "capability": "switch",
        "command": "关闭客厅的主灯"
      }},
      {{
        "deviceId": "shade_01",
        "capability": "windowShade",
        "command": "打开卧室的窗帘"
      }}
    ]
  }}
]

### 请基于以下设备列表生成 1 个符合上述规范的测试用例：

{devices}
"""

raw_json = asyncio.run(load_device_control_cases(MULTI_DEVICES_TEST_PROMPT))
multi_devices_cases_data = json.loads(raw_json)

# 3. 准备满足 parametrize 格式的数据
# 注意：parametrize 需要一个元组列表 [(id1, devId1, cmd1, cap1), ...]
multi_devices_cases = [
    (c["id"], c["command"], c["steps"])
    for c in multi_devices_cases_data
]
multi_ids = [c["id"] for c in multi_devices_cases_data]

# --- 实例化并使用 ---
judge_model = JudgeLlmClient(
    model_name=JUDGE_MODEL,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    verify_ssl=False  # 这里设置跳过 SSL 验证
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
    model=judge_model,
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
    print(f"command:{command}----expected_output:{expected_output}-----actual_output:{actual_output}")
    assert_test(test_case, metrics=[DEVICE_CONTROL_METRIC])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "id,command,steps",
    multi_devices_cases,
    ids=multi_ids
)
async def test_multi_device_control(id: str, command: str, steps: list):
    # 调用线上接口,获取期望值
    params = {"prompt": command, "location_id": LOCATION_ID, "user_token": PAT, "request_id": str(uuid.uuid4())}
    expected_output = get_streaming_response(params)
    # 调用st接口,获取实际值
    # 在这里调用其他接口，例如获取设备状态
    actual_output = ""
    for item in steps:
        device_status_url = f"{API_BASE_URL}/devices/{item["deviceId"]}/status"
        status_response = requests.get(
            url=device_status_url,
            headers=API_HEADERS,
            verify=False
        )
        if status_response.status_code == 200:
            data = status_response.json()
            # 逐层安全获取，给定默认值 {} 防止 NoneType 报错
            main_component = data.get("components", {}).get("main", {})
            cap_data = main_component.get(item["capability"], {}).get(item["capability"], {})

            # 最终取到 value，如果没有则默认为空字符串或 None
            value = cap_data.get("value", "")
            actual_output = item["deviceId"] + value
        else:
            print(f"Failed to get status for device {deviceId}")

    test_case = LLMTestCase(
        input=command,
        actual_output=actual_output,
        expected_output=expected_output,
        context=[],
        name=id
    )
    print(f"command:{command}----expected_output:{expected_output}-----actual_output:{actual_output}")
    assert_test(test_case, metrics=[DEVICE_CONTROL_METRIC])
