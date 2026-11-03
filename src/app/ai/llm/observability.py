from app.observability import span

from .models import LLMClient, Request, Response


class LLMTracer:
    def __init__(self, inner: LLMClient, name: str = "llm.complete"):
        self.inner, self.name = inner, name

    async def generate(self, request: Request, **kw) -> Response:
        async with span(self.name, n_messages=len(request.messages)) as rec:  # pyright: ignore[reportCallIssue]
            resp = await self.inner.generate(request, **kw)
            in_tokens = 0
            out_tokens = 0
            if resp.usage:
                in_tokens = in_tokens
                out_tokens = out_tokens

            rec.update(
                model=resp.model,
                finish=resp.finish_reason,
                in_tokens=in_tokens,
                out_tokens=out_tokens,
            )
            return resp
