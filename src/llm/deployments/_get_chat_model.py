from enum import Enum
from pathlib import Path

from llama_index.llms.llama_cpp import LlamaCPP

LOCAL_MODEL_CACHE = Path(__file__).resolve().parent.parent / ".model_cache"


class AvailableChatModels(Enum):
    LLAMA_2_13B_Q4 = "llama-2-13b-chat.Q4_0.gguf"
    LLAMA_32_3B_Q8 = "llama-3.2-3b-instruct-q8_0.gguf"
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
                temperature (float): The temperature for the model. Defaults to 0.1.
        max_new_tokens (int): The maximum number of new tokens. Defaults to 256.
        context_window (int): The context window size. Defaults to 3900.
        use_gpu (bool): Whether to use GPU. Defaults to True.
        generate_kwargs (dict): Additional kwargs for model generation.


    Returns:
        The loaded chat model.
    """

    if use_local:
        if path_or_url is None:
            # raise ValueError("Local path must be provided when use_local is True.")
            model_path = LOCAL_MODEL_CACHE / model_name.value
            print(model_path)
        else:
            model_path = Path(path_or_url).resolve() / model_name.value  # type: ignore

        if not model_path.exists():
            raise FileNotFoundError(f"Local model file not found at {model_path}")

        # Load the model from the local path
        llm = LlamaCPP(
            model_path=str(model_path),
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
        llm = LlamaCPP(
            model_url=model_url,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=context_window,
            # kwargs to pass to __call__()
            generate_kwargs=generate_kwargs,
            model_kwargs={"n_gpu_layers": -1} if use_gpu else {"n_gpu_layers": 0},
            verbose=True,
        )

    return llm


if __name__ == "__main__":
    model_name = AvailableChatModels.LLAMA_2_13B_Q4
    llm = get_chat_model(model_name, use_local=True)
    print(llm)
