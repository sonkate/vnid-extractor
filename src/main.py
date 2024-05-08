import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse

from utils import image_utils
from extractor.extractor import IdExtractor

model = IdExtractor()

app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>"""

app = FastAPI(title="Tensorflow FastAPI Start Pack", description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(f: UploadFile = File(...)):
    response = {}
    if not f.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        response = {"error": "Image must be jpg, jepg or png format!"}
        return response

    image = image_utils.read_image_file(await f.read())
    # response = model.extract(image)
    try:
        print("Extracting...")
        response = model.extract(image)
    except Exception as e:
        response = {"error": str(e)}
        
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
