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


class SourceResponse(BaseModel):
    source: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@app.post("/ask")
def ask(
    request: AskRequest,
    pipeline: ResearchPipeline = Depends(get_pipeline),
) -> AskResponse:
    try:
        report = pipeline.ask(request.question)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AskResponse(
        answer=report.answer,
        sources=[
            SourceResponse(source=source.source)
            for source in report.sources
        ],
    )