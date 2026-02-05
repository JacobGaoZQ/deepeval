import requests

requests.packages.urllib3.disable_warnings()

API_BASE_URL = "https://api.samsungiotcloud.cn"
LOCATION_ID = "f4b3af92-5826-416e-8e28-8b1c252912f1"
API_HEADERS = {
    "Authorization": "Bearer 2638ea03-761b-4902-9175-64b5c54ea229",
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
    # 在这里调用其他接口，例如获取设备状态
    device_status_url = f"{API_BASE_URL}/devices/{device_id}/status"
    status_response = requests.get(
        url=device_status_url,
        headers=API_HEADERS,
        verify=False
    )
    if status_response.status_code == 200:
        device["components"] = status_response.json().get("components")
    else:
        print(f"Failed to get status for device {device_id}")
    # 设备在离线信息查询
    response = requests.get(f"{API_BASE_URL}/devices/{device_id}/health", headers=API_HEADERS, verify=False)
    if response.status_code != 200:
        print(f"Failed to get health for device {device_id}")
    try:
        response_json = response.json()
        device["state"] = response_json.get("state")
    except requests.exceptions.JSONDecodeError:
        print(f"Could not decode health response for device {device_id}")

    print(device)
# def load_device_control_cases():
