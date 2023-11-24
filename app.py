from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, Header, Request
from fastapi.responses import JSONResponse
from fastapi import requests
from fastapi import FastAPI, Response, Form,status
from detect import detect
import traceback
#Fast API Object
app = FastAPI() 


# origins = ["*"]  
# origins = ["https://user-app.click", "https://user-app.click/", "https://voltox.tech/", "https://voltox.tech" ,"https://voltox.global/", "https://voltox.global","https://super-admin.click/", "https://super-admin.click" ,"http://localhost:3000/","http://localhost:3000"]

origins = ['*']  

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

context = ""


@app.post('/object_detection') 
async def transcripe(file: UploadFile): 
    try:

        # MODEL_NAME = "gpt-3.5-turbo"
        # chromadb.Client()
        print("API HIT ************* Detection")
        img_path = file.filename
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        boxes,logits,phrases = detect(img_path)
        print(boxes,logits)
        boxes,logits = boxes.tolist(),logits.tolist()
        if boxes is not None:
            status = True
        else:
            status = False
        if status:
            response = {'status':200, 'boxes':boxes,'logits' : logits,'phrases':phrases}
        else:
            response = {'status':404, 'boxes':boxes,'logits' : logits,'phrases':phrases}
        print(response)
        return response
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        response = {'status':404, 'boxes':None,'logits' : None,'phrases':None}
        return response
    
