from fastapi import FastAPI

app = FastAPI(title="Yargı Web Interface")

@app.get("/")
async def home():
    return {"message": "Yargı Web Interface Çalışıyor!", "status": "OK"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
