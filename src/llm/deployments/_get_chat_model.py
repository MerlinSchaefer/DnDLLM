from enum import Enum
from pathlib import Path

from llama_index.llms.llama_cpp import LlamaCPP

LOCAL_MODEL_CACHE: str = "../.model_cache/"


class AvailableChatModels(Enum):
    LLAMA_2_13B_Q4 = "llama-2-13b-chat.Q4_0.gguf"
    # Add other models here as needed


# TODO: refactor
def get_chat_model(
    model_name: AvailableChatModels,
    use_local: bool = True,
    path_or_url: str | Path | None = None,
    temperature: float = 0.1,
    max_new_tokens: int = 256,
    context_window: int = 3900,
    generate_kwargs: dict = {},
    use_gpu: bool = True,
) -> LlamaCPP:
    """
    Get the chat model.

    Args:
        model_name (AvailableChatModels): The name of the model.
        use_local (bool): Whether to use a local path or a URL. Defaults to True.
        path_or_url (Union[str, Path]): The local path or URL to the model. Defaults to None.

    Returns:
        The loaded chat model.
    """

    if use_local:
        if path_or_url is None:
            # raise ValueError("Local path must be provided when use_local is True.")
            model_path = Path(LOCAL_MODEL_CACHE) / model_name.value
        model_path = Path(path_or_url) / model_name.value  # type: ignore
        # Load the model from the local path
        llm = LlamaCPP(
            # You can pass in the URL to a GGML model to download it automatically
            # model_url=model_url,
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=context_window,
            # kwargs to pass to __call__()
            generate_kwargs=generate_kwargs,
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            model_kwargs={"n_gpu_layers": -1} if use_gpu else {"n_gpu_layers": 0},
            verbose=True,
        )
    else:
        if path_or_url is None:
            raise ValueError("URL must be provided when use_local is False.")
        model_url = path_or_url
        # Load the model from the URL
        # Replace the following line with the actual model loading code
        llm = LlamaCPP(
            # You can pass in the URL to a GGML model to download it automatically
            model_url=model_url,
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            # model_path=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=context_window,
            # kwargs to pass to __call__()
            generate_kwargs=generate_kwargs,
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            model_kwargs={"n_gpu_layers": -1} if use_gpu else {"n_gpu_layers": 0},
            verbose=True,
        )

    return llm


# Usage example
model_name = AvailableChatModels.LLAMA_2_13B_Q4
llm = get_chat_model(model_name, use_local=True, path_or_url="path/to/local/models")
print(llm)
