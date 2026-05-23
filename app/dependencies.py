from functools import lru_cache

from app.pipelines import PipelineRunner


@lru_cache
def get_pipeline_runner() -> PipelineRunner:
    return PipelineRunner()
