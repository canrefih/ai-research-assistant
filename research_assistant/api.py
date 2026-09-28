from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from .pipeline import ResearchPipeline


app = FastAPI(
    title="AI Research Assistant",
    version="0.1.0",
)



def get_pipeline() -> ResearchPipeline:
    return ResearchPipeline()



@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


@app.post("/ask")
def ask(
    request: AskRequest,
    pipeline: ResearchPipeline = Depends(get_pipeline),
) -> dict:
    try:
        report = pipeline.ask(request.question)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "answer": report.answer,
        "sources": [source.source for source in report.sources],
    }