from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .pipeline import ResearchPipeline
from .web_search import TavilySearchProvider
from functools import lru_cache


app = FastAPI(
    title="AI Research Assistant",
    version="0.1.0",
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(
        Path(__file__).parent / "static" / "index.html",
    )



@lru_cache
def get_pipeline() -> ResearchPipeline:
    pipeline = ResearchPipeline(
        web_search_provider=TavilySearchProvider(),
    )
    pipeline.load_index()
    return pipeline



@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    use_web_search: bool = False
    top_k: int = Field(default=5, ge=1)


class SourceResponse(BaseModel):
    source: str
    title: str | None = None
    url: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@app.post("/ask")
def ask(
    request: AskRequest,
    pipeline: ResearchPipeline = Depends(get_pipeline),
) -> AskResponse:
    try:
        report = pipeline.research(
            request.question,
            top_k=request.top_k,
            use_web_search=request.use_web_search,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AskResponse(
        answer=report.answer,
        sources=[
            SourceResponse(
                source=source.source,
                title=source.title,
                url=source.url,
            )
            for source in report.sources
        ],
    )