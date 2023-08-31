# FlagEval Serving

> Serving Framework of AI Models for Evaluating on FlagEval Platform.


## Installation

``` shell
pip install --upgrade flageval-serving
```

## Usage

1. **Model**: of course we have a model that is ready be evaluated, let's assume it lives in the path: `/path/to/model`;
2. Then we can write our service code, let's put the service code in `service.py` or './tests/service.py' and take a NLP model as the example:


    ``` python
    from flageval.serving.service import NLPModelService, NLPEvalRequest, NLPEvalResponse, NLPCompletion


    class DemoService(NLPModelService):
        def global_init(self, model_path: str):
            print("Initial model with path", model_path)

        def infer(self, req: NLPEvalRequest) -> NLPEvalResponse:
            return NLPEvalResponse(
                completions=[
                    NLPCompletion(
                        text='Hello, world!',
                        tokens='Hello, world!',
                    ),
                ]
            )

    ```

3. Finally, we use the `flageval-serving` command to serve:

    ```shell
    flageval-serving --service service:DemoService dev /path/to/model  # start a development server
    flageval-serving --service service:DemoService run /path/to/model  # start a production server
    ```
## Dockerfile
FlagEval evaluation platform construction image