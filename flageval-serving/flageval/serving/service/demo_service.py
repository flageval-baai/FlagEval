from . import NLPModelService, NLPEvalRequest, NLPEvalResponse, NLPCompletion


class DemoService(NLPModelService):
    def global_init(self, model_path: str):
        print("Initial model with path", model_path)

    def infer(self, req: NLPEvalRequest) -> NLPEvalResponse:
        print(req)
        return NLPEvalResponse(
            completions=[
                NLPCompletion(
                    text='Hello, world!',
                    tokens='Hello, world!',
                ),
            ]
        )
