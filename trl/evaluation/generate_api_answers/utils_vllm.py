import os
import time
import random
import openai
import logging

from openai import APIError, APIConnectionError, RateLimitError
from openai.types import Completion
from openai.types.chat import ChatCompletion


class ClientError(RuntimeError):
    pass


def get_content(query, base_url, model_name, is_chat_model=False):
    API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
    API_REQUEST_TIMEOUT = int(os.getenv('OPENAI_API_REQUEST_TIMEOUT', '99999'))
    import httpx
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=base_url,
        timeout=httpx.Timeout(API_REQUEST_TIMEOUT),
    )

    call_args = dict(
            model=model_name,
            temperature=0.0,
            max_tokens=2048,
            stop=["\n\n"]
    )

    if is_chat_model:
        messages = [{'role': 'user', 'content': query}]
        call_args['messages'] = messages
        call_func = client.chat.completions.create
    else:
        call_args['prompt'] = query
        call_func = client.completions.create
    call_args['timeout'] = API_REQUEST_TIMEOUT

    result = ''
    try:
        completion = call_func(**call_args, )
        if is_chat_model:
            assert isinstance(completion, ChatCompletion)
            result = completion.choices[0].message.content
        else:
            assert isinstance(completion, Completion)
            result = completion.choices[0].text
    except AttributeError as e:
        err_msg = getattr(completion, "message", "")
        if err_msg:
            time.sleep(random.randint(25, 35))
            raise ClientError(err_msg) from e
        raise ClientError(err_msg) from e
    except (APIConnectionError, RateLimitError) as e:
        err_msg = e.message
        time.sleep(random.randint(25, 35))
        raise ClientError(err_msg) from e
    except APIError as e:
        err_msg = e.message
        if "maximum context length" in err_msg:  # or "Expecting value: line 1 column 1 (char 0)" in err_msg:
            logging.warning(f"max length exceeded. Error: {err_msg}")
            return {'gen': "", 'end_reason': "max length exceeded"}
        time.sleep(1)
        raise ClientError(err_msg) from e
    return result

if __name__ == "__main__":
    conversation_history = []
    user_input = "Hello!"
    res = get_content(user_input, "http://10.77.249.36:8030/v1", "Qwen/QwQ")
    print(f"Response: {res}")

    user_input = "How are you?"
    res = get_content(user_input, "http://10.77.249.36:8030/v1", "Qwen/QwQ")
    print(f"Response: {res}")