from vllm import SamplingParams
from e2b_code_interpreter import Sandbox
from pathlib import Path
from typing import Optional, List

default_system_prompt = "You can answer questions and solve problems. If running code helps, write it inside <code> </code>, and you will see the result. Example: To calculate 2 + 2, write <code> print(2 + 2) </code>."

default_user_script_prompt= "This is a user-provided script containing tools that you can use to help complete tasks. It will be added to the sandbox environment so you can call its functions. If you are unsure how to use the available tools, you can use the `help()` function to inspect them."



def get_code(chat: str, tools_script: str = None, parsing_token: str = "<code>") -> str:
    """
    Extracts and optionally prepends a tools script to a code snippet from a chat message.

    Args:
        chat (str): The chat message containing the code snippet.
        tools_script (str, optional): A script to prepend to the extracted code snippet. Defaults to None.
        parsing_token (str, optional): The token used to identify the start of the code snippet in the chat message. Defaults to "<code>".

    Returns:
        str: The extracted code snippet, optionally prepended with the tools script.
    """
    code = chat.split(parsing_token)[-1]
    if tools_script:
        code = f"{tools_script}\n{code}"
    return code

    
def e2b_executer(code: str, api_key: str, dependancies: Optional[List[str]] = None, template: Optional[str] = None) -> str:
    """
    Executes a given code snippet in an e2b sandbox environment, optionally installing dependencies.

    Args:
        code (str): The code snippet to execute.
        api_key (str): Your E2B API Key.
        dependancies (Optional[List[str]]): A list of dependencies to install before executing the code. Defaults to None.
        template (Optional[str]): An optional template for the sandbox environment. Defaults to None.

    Returns:
        str: The response from the sandbox environment after executing the code.
    """
    sbx = Sandbox(api_key=api_key, template=template)
    if dependancies:
        packages = " ".join(dependancies)
        install_command = f"pip install {packages}"
        sbx.commands.run(install_command)
    response = sbx.run_code(code)
    sbx.kill()
    return str(response)

def read_script(user_script_path: str) -> str:
    """
    Reads and returns the content of the provided script.

    Args:
        user_script_path (str): Path to the user-provided script.
    
    Returns:
        str: The content of the script.
    
    Raises:
        RuntimeError: If the script cannot be read.
    """
    try:
        return Path(user_script_path).read_text()
    except Exception as e:
        raise RuntimeError(f"Error reading the user script: {e}")
    

def prepare_data(
    conversations: List[List[dict]], 
    tokenizer,
    system_prompt: str = default_system_prompt, 
    user_script_description: str = default_user_script_prompt,
    user_script_path: str = None, 
) -> list:
    """
    Prepares the conversational dataset by:
      1. Constructing a system prompt.
         - If a system_prompt is provided, use it; otherwise, use the default.
         - If a user_script_path is provided, read the script and append it with a description.
      2. Adding the system prompt as a dict (with role "system") as the first message in each conversation.
      3. Using the tokenizer's apply_chat_template to convert the conversations to strings.

    Args:
        conversations (List[List[dict]]): A list of conversation lists (each conversation is a list of dict messages).
        tokenizer: The tokenizer instance with an "apply_chat_template" method.
        system_prompt (str, optional): Custom system prompt. Defaults to a pre-defined prompt.
        user_script_description (str, optional): Description to be added before the script content. 
        user_script_path (str, optional): Path to a user-provided script to read and add.

    Returns:
        list: List of string prompts after applying the chat template.
    """

    # Only if a user script path is provided, add both description and script to the system_prompt.
    if user_script_path:
        try:
            from pathlib import Path
            user_script = read_script(user_script_path)
        except Exception as e:
            raise RuntimeError(f"Error reading the user script: {e}")
        system_prompt += "\n" + user_script_description + "\n" + user_script


    # Create the system prompt message dict.
    system_message = {"role": "system", "content": system_prompt}

    # Prepend the system message to each conversation.
    for convo in conversations:
        # Avoid duplicating if already added.
        if not convo or convo[0].get("role") != "system":
            convo.insert(0, system_message)

    # Apply the tokenizer's chat template to each conversation.
    # Assuming tokenize=False to return a string output.
    processed_prompts = tokenizer.apply_chat_template(
        conversations, 
        tokenize=False,  
        add_generation_prompt=True
    )
    return processed_prompts


def generate_model_responses(
    dataset: list,
    llm,
    api_key: str,
    tools_script_path: str = None,
    parsing_token: str = "<code>",
    stop_string: str = "</code>",
    temperature: float = 0.9,
    max_tokens: int = 1024,
    dependancies: list = None,
    template: str = None,
) -> list:
    """
    Generates responses for the model with potential code execution.

    This function takes a prepared dataset (list of prompts), uses the provided llm
    to generate responses, and checks if the response indicates that code execution is expected.
    If so, it extracts the code using get_code(), executes it via e2b_executer(), appends the
    execution output to the conversation, and reprocesses it. Once all chats are complete,
    a list of complete conversations is returned.

    Args:
        dataset (list): List of preprocessed prompts (strings).
        llm: The language model to use for generation.
        sampling_params: Sampling parameters for the llm.generate method.
        tools_scrip_patht (str, optional): path to script to prepend to code extracted. Defaults to None.
        parsing_token (str, optional): Token used to locate the code in the conversation. Defaults to "<code>".
        api_key (str): API key for executing code in the sandbox.
        dependancies (list, optional): List of dependencies to install before code execution. Defaults to None.
        template (str, optional): Optional template for the sandbox environment. Defaults to None.

    Returns:
        list: A list of complete conversations (strings).
    """
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,stop=[stop_string])
    tools_script = read_script(tools_script_path) if tools_script_path else None

    completed_chats = []    # Chats that are fully complete.
    current_batch = dataset  # Start with your initial batch of prompts.

    while current_batch:
        # Generate outputs for the current batch.
        outputs = llm.generate(current_batch, sampling_params)
        next_batch = []  # To store chats that still need code execution.

        # Process each output in the batch.
        for output in outputs:
            # Reconstruct the conversation (prompt + generated response).
            conversation = output.prompt + output.outputs[0].text

            # Check if response is waiting for code execution.
            if output.outputs[0].stop_reason == stop_string:
                # Extract the code to execute.
                code_to_execute = get_code(conversation, tools_script=tools_script, parsing_token=parsing_token)
                # Execute the code via the sandbox.
                executed_result = e2b_executer(code_to_execute, api_key, dependancies, template)
                # Append the execution result to the conversation.
                updated_conversation = conversation + f"{stop_string}<output>" + executed_result + "</output>"
                # Add this updated conversation to the next batch.
                next_batch.append(updated_conversation)
            else:
                # If chat is complete, add it.
                completed_chats.append(conversation)
        # Process next batch.
        current_batch = next_batch

    return completed_chats


