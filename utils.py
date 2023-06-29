import requests
from PIL import Image
from io import BytesIO


def get_street_view_params(lat, long, heading, api_key, pitch=-40, fov=90):
    params = {
        "location": f"{lat},{long}",
        "scale": 1,
        "pitch": pitch,
        "fov": fov,
        "heading": heading,
        "size": "640x640",
        "format": "png",
        "visual_refresh": True,
        "key": api_key,
    }

    return params


def get_static_map_params(center_lat, center_long, api_key):
    params = {
        "center": f"{center_lat},{center_long}",
        "zoom": 30,
        "scale": 1,
        "size": "640x640",
        "maptype": "satellite",
        "format": "png",
        "visual_refresh": True,
        "key": api_key,
    }

    return params


def get_sat_and_street(center_lat, center_long, api_key, SAT_URL, STREET_URL):
    images = []
    headings = {0, 90, 180, 270}

    for heading in headings:
        street_params = get_street_view_params(
            center_lat, center_long, heading, api_key
        )
        resp = requests.get(STREET_URL, params=street_params)
        img = Image.open(BytesIO(resp.content))
        images.append(img)
        print(heading)

    # sat_params = get_static_map_params(center_lat, center_long, api_key)

    # resp = requests.get(SAT_URL, params=sat_params)
    # img = Image.open(BytesIO(resp.content))
    # images.append(img)

    return images


def single_test(center_lat, center_long, heading, pitch, fov, api_key, STREET_URL):
    street_params = get_street_view_params(center_lat, center_long, heading, api_key)
    resp = requests.get(STREET_URL, params=street_params)
    img = Image.open(BytesIO(resp.content))

    return img


def get_multi_street(
    center_lat, center_long, headings, pitch, fov, api_key, STREET_URL, SAT_URL
):
    images = []

    for heading in headings:
        street_params = get_street_view_params(
            center_lat, center_long, heading, api_key, pitch, fov
        )
        resp = requests.get(STREET_URL, params=street_params)
        img = Image.open(BytesIO(resp.content))
        images.append(img)
        # print(heading)

    # sat_params = get_static_map_params(center_lat, center_long, api_key)

    # resp = requests.get(SAT_URL, params=sat_params)
    # img = Image.open(BytesIO(resp.content))
    # images.append(img)

    return images


import nvidia_smi
import os


def choose_gpu(min_memory=4 * 1024 * 1024):
    nvidia_smi.nvmlInit()

    num_devices = nvidia_smi.nvmlDeviceGetCount()
    best_gpu = ""

    if num_devices > 0:
        free_memory = []

        for device_id in range(num_devices):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            free_memory.append((info.free, device_id))

        free_memory, best_gpu = sorted(free_memory)[::-1][0]

        if free_memory > min_memory:
            best_gpu = str(best_gpu)

    nvidia_smi.nvmlShutdown()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)


choose_gpu()
