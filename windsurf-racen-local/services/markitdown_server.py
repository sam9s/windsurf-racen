from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from markitdown import MarkItDown

app = FastAPI(title="Local MarkItDown Service")


class ConvertRequest(BaseModel):
    uri: str


class ConvertResponse(BaseModel):
    markdown: str


@app.post("/convert", response_model=ConvertResponse)
def convert(req: ConvertRequest) -> ConvertResponse:
    try:
        md = MarkItDown(enable_plugins=False)
        result = md.convert_uri(req.uri)
        return ConvertResponse(markdown=result.markdown)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
