# main.py
import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

import torch
from PIL import Image
import torchvision.transforms as transforms

# import model classes + loader util (paste/ensure bạn có DeblurGenerator trong same project)
from gan_deblur import DeblurGenerator
from utils import load_generator_checkpoint

app = FastAPI(title="Deblur Generator API")

# Cho phép frontend ở địa chỉ khác gọi (dev). Ở production hãy chỉnh lại origin hợp lệ.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount thư mục static để phục vụ frontend
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# transform (nhớ tương thích với train)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# denormalize function
def tensor_to_pil(img_tensor):
    # img_tensor: CxHxW, giá trị trong [-1,1]
    img = img_tensor.cpu().clamp(-1, 1)
    img = (img + 1) / 2  # map to [0,1]
    img = img.mul(255).byte()
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img)

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = DeblurGenerator().to(device)
CHECKPOINT_PATH = "generator_epoch99.pth"

try:
    load_generator_checkpoint(generator, CHECKPOINT_PATH, device)
    generator.eval()
    print("Loaded generator checkpoint successfully.")
except Exception as e:
    print("Warning: couldn't load checkpoint at startup:", e)

@app.get("/")
async def index():
    # serve a small HTML that we'll put in static/index.html
    html_path = os.path.join("static", "index.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, "r", encoding="utf-8").read())
    return {"msg": "Put frontend at static/index.html"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Nhận 1 file ảnh, trả về ảnh xử lý (PNG).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là image/*")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không mở được ảnh: {e}")

    # transform và batch dim
    input_t = transform(img).unsqueeze(0).to(device)  # 1xCxHxW

    with torch.no_grad():
        generator.eval()
        out_t = generator(input_t)  # output in [-1,1] (the model returns clamp)
    # tách tensor đầu
    out_img = tensor_to_pil(out_t[0])

    # convert to bytes
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# Optionally endpoint health
@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}
