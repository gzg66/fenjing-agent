import base64
import mimetypes
import requests
import json
import time


def image_to_base64_uri(image_path: str) -> str:
    """将本地图片文件转换为 Base64 data URI 格式。"""
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def main():
    print("Hello from RunningHubAPI!")
    API_KEY = "3b462ed8822942528cf06cb1953a6e9a"
    print(f"API_KEY: {API_KEY}")

    url = "https://www.runninghub.cn/openapi/v2/alibaba/wan-2.6/image-to-video"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    # 将本地图片转为 Base64 data URI
    image_path = r"D:\workspace\fenjing-agent\uploads\gen_c1f93ceda1dc.png"
    image_base64 = image_to_base64_uri(image_path)
    print(f"Image loaded: {image_path} ({len(image_base64)} chars)")

    payload = {
    "imageUrl": image_base64,
    "prompt": "【风格】3D 动画风格，色彩明快饱和，光影富有层次感，营造温馨奇幻的节日氛围。，15秒，16:9\n【故事】小书灵逛繁华灯会：一个可爱的小书灵在充满魔法气息的繁华灯会中穿行，探索奇妙的世界。\n\n【时间轴】\n0-3秒：[推镜头] 建立镜头。繁华灯会的全景，街道两旁挂满了各式各样的发光灯笼，色彩斑斓，光影交错。\n3-6秒：[跟镜头] 镜头切至中景。小书灵 悠闲地在人群上方低空飞行，它好奇地左顾右盼，身后留下一串淡金色的墨滴光效。\n6-9秒：[环绕镜头] 小书灵在一个巨大的巨龙灯笼前停下。巨龙灯笼栩栩如生，火红的光芒映照在小书灵充满期待的脸上。\n9-12秒：[特写/移镜头] 小书灵伸出小手，轻轻触碰一个飘过的兔子灯笼，灯笼发出一阵柔和的魔法涟漪。\n12-15秒：[拉镜头/升降镜头] 小书灵加速向上飞向夜空，背景是成千上万升起的孔明灯，画面逐渐拉远，展示出整个灯会如梦似幻的全貌。\n\n【声音设计】\n配乐：欢快且具有奇幻感的民乐（笛子与古筝），伴随轻微的铃铛声。\n音效：热闹的集市背景音，魔法闪烁声，灯笼摇曳的沙沙声。",
    "negativePrompt": "",
    "resolution": "720p",
    "duration": "15",
    "shotType": "multi"
}

    begin = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        task_id = result.get("taskId")
        print(f"Task submitted successfully. Task ID: {task_id}")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return

    query_url = "https://www.runninghub.cn/openapi/v2/query"
    query_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    # Poll for results
    while True:
        query_payload = {"taskId": task_id}
        response = requests.post(query_url, headers=query_headers, data=json.dumps(query_payload))
        if response.status_code == 200:
            result = response.json()
            status = result["status"]

            if status == "SUCCESS":
                end = time.time()
                print(f"Task completed in {end - begin} seconds.")
                if result.get("results") and len(result["results"]) > 0:
                    output_url = result["results"][0]["url"]
                    print(f"Task completed. URL: {output_url}")
                else:
                    print("Task completed but no results found.")
                break
            elif status == "RUNNING" or status == "QUEUED":
                print(f"Task still processing. Status: {status}")
            else:
                # FAILED or other error status
                error_message = result.get("errorMessage", "Unknown error")
                print(f"Task failed: {error_message}")
                break
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break

        time.sleep(5)


if __name__ == "__main__":
    main()