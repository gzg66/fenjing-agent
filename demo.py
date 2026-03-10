import os
from pickle import TRUE
import requests
import json
import time

def main():
    print("Hello from RunningHubAPI!")
    API_KEY = RUNNINGHUB_API_KEY
    print(f"API_KEY: {API_KEY}")

    url = "https://www.runninghub.cn/openapi/v2/kling-v3.0-pro/image-to-video"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
    "prompt": "欧洲别墅户外露台场景，铺着蓝白格纹桌布的餐桌旁，年轻白人女性穿蓝白条纹短袖衬衫、卡其色短裤，系棕色腰带，赤脚坐着，对面是穿白色 T 恤的年轻白人男性，镜头推进，女性晃着玻璃杯里的果汁，目光望向远处的树林，说“These trees will turn yellow in a month, won't they?”，镜头特写男性低着头说，“but they'll be green again next summer.”，然后女性转头，笑着看向对面的男性，说，“Are you always this optimistic? Or just about summer?”，然后男性抬起头，看着女生说，“Only about summers with you。”",
    "negativePrompt": "",
    "firstImageUrl": "https://www.runninghub.cn/view?filename=6ea666eb43d1e867e9d31b26aa16fed41e77a2387fea0d0c330c250df47344ee.jpg&type=input&subfolder=&Rh-Comfy-Auth=eyJ1c2VySWQiOiIzZjY1MTNlNWEwNjY1N2I4OGYyNjU5NTEzYmU3ZDM0YyIsInNpZ25FeHBpcmUiOjE3NzA4Nzk1NjgzNDUsInRzIjoxNzcwMjc0NzY4MzQ1LCJzaWduIjoiNWY0OWUyMzBmNjM0NTViOGZhZWM5NzQzMDYzMjIzOWYifQ==&Rh-Identify=3f6513e5a06657b88f2659513be7d34c&rand=0.7161105046578343",
    "lastImageUrl": "",
    "duration": "15",
    "cfgScale": 0.5,
    "sound": TRUE
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