from typing import List, Union
from enum import Enum
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from apiconfig import *
from func import *
from PIL import Image
from time import time
import pandas as pd
import json
import torch
import io
import os
import random

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

eng_model = Reader(['en'], gpu = True, detector = False)

kor_model = Reader(['ko'], gpu=True,
            model_storage_directory='./model/easyOCR',
            user_network_directory='./model/easyOCR',
            recog_network='custom')

class ModelID(str, Enum):
    id = "id"
    foreign = "foreign"


@app.get('/', response_class=HTMLResponse) 
def main(request: Request):
    return templates.TemplateResponse('request.html', context = {"request": request})
    
@app.post("/predict/")
async def predict1(files: List[bytes] = File(), form : ModelID = 'id'):
    start_time = time()
    
    image = Image.open(io.BytesIO(files[0]))
    form = form.value
    image_arr = np.array(image)
    
    # img detect
    print('load 시간:', time() - start_time)
    start_time = time()

    detect_result = detect_postprocess(kor_model.detect(image_arr))
    print('detect 시간:', time() - start_time)
    start_time = time()

    # 데이터 분류 , form에 양식 넣기
    bbox_class, model_type = bbox_classification(image_arr, detect_result, form = form)

    # 글씨 분류
    rec_result = data_recognition(image_arr, bbox_class, model_type, kor_model, eng_model)
    print('reconize 시간:', time() - start_time)
    start_time = time()

    # 후처리 - 모델에 따라 customize 필요 
    result, pre_result = recog_postprocess(rec_result, need_address = True, class_result = form)
    print('후처리 시간:', time() - start_time)
    start_time = time()

    blur_img = blur_image(image_arr, detect_result)
    img = Image.fromarray(blur_img)
    img = img.resize((410, 260))
    blur_url = './img/blur'+str(random.random())+'.png'
    img.save(blur_url)
    result['url'] = blur_url[1:]
    result['pre_result'] = pre_result
    result['form'] = form

    return result