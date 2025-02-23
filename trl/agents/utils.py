from ..import_utils import is_agents_available

if is_agents_available():
    from e2b_code_interpreter import Sandbox
    from pathlib import Path
    from typing import Optional, List, Callable
    from inspect import getsource
    from vllm import LLM,SamplingParams
    from langchain_experimental.utilities import PythonREPL
else:
    raise ImportError(
    "Agents utilities are not available. Please install trl with "
    "`pip install trl[agents]` to use utils"
)

default_system_prompt = "You can answer questions and solve problems. If running code helps, write it inside <code> </code>, and you will see the result. Example: To calculate 2 + 2, write <code> print(2 + 2) </code>."

default_environment_prompt = "This is a user-provided script containing tools that you can use to help complete tasks. It will be added to the sandbox environment so you can call its functions. If you are unsure how to use the available tools, you can use the `help()` function to inspect them."



def get_code(chat: str, tools_script: str = None, parsing_string: str = "<code>") -> str:
    """
    Extracts and optionally prepends a tools script to a code snippet from a chat message.
    Args:
        chat (str): The chat message containing the code snippet.
        tools_script (str, optional): A script to prepend to the extracted code snippet. Defaults to None.
        parsing_string (str, optional): The string used to identify the start of the code snippet in the chat message. Defaults to "<code>".
    Returns:
        str: The extracted code snippet, optionally prepended with the tools script.
    """
    code = chat.split(parsing_string)[-1]
    if tools_script:
        code = f"{tools_script}\n{code}"
    return code


class E2BExecutor:
    """
    A class to handle code execution in an e2b sandbox environment.
    """
    def __init__(self, api_key: str, dependencies: Optional[List[str]] = None, template: Optional[str] = None):
        """
        Initialize the E2BExecutor with API key and optional settings.
        
        Args:
            api_key (str): Your E2B API Key.
            dependencies (Optional[List[str]]): A list of dependencies to install. Defaults to None.
            template (Optional[str]): Template for the sandbox environment. Defaults to None.
        """
        self.api_key = api_key
        self.dependencies = dependencies
        self.template = template

    def execute(self, code: str) -> str:
        """
        Executes a given code snippet in an e2b sandbox environment.
        
        Args:
            code (str): The code snippet to execute.
            
        Returns:
            str: The response from the sandbox environment after executing the code.
        """
        sbx = Sandbox(api_key=self.api_key, template=self.template)
        if self.dependencies:
            packages = " ".join(self.dependencies)
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


class LocalExecutor:
    """
    A class to handle local code execution using LangChain's PythonREPL.
    """
    def execute(self, code: str) -> str:
        """
        Executes a given code snippet using PythonREPL.
        
        Args:
            code (str): The code snippet to execute.
            
        Returns:
            str: The output from executing the code.
        """
        try:    
            repl = PythonREPL()
            result = repl.run(code)
            return str(result)
        except Exception as e:
            return f"Error executing code: {str(e)}"

def prepare_data_for_e2b_agent(
    dataset,  
    tokenizer,
    prompt_column: str = "prompt",  
    system_prompt: str = default_system_prompt, 
    environment_prompt: str = default_environment_prompt,
    tools_script_path: str = None,
) -> list:
    """
    Prepares the Hugging Face dataset for the e2b agent by constructing conversations 
    and applying the system prompt.

    Args:
        dataset: A Hugging Face dataset object
        tokenizer: A tokenizer with an `apply_chat_template` method to process conversations
        prompt_column (str): Name of the column containing prompts. Defaults to "prompt"
        system_prompt (str, optional): The base system prompt. Defaults to default_system_prompt
        environment_prompt (str, optional): Description to prepend before the tools script
        tools_script_path (str, optional): File path to a tools script to include in system prompt

    Returns:
        Dataset: Modified dataset with processed prompts in the prompt column
    """
    # Convert dataset prompts to conversational format
    conversations = [[{"role": "user", "content": prompt}] for prompt in dataset[prompt_column]]

    # If a tools script path is provided, read its content and append with the environment prompt
    if tools_script_path:
        try:
            tool_script = read_script(tools_script_path)
            system_prompt += "\n" + environment_prompt + "\n" + tool_script
        except Exception as e:
            raise RuntimeError(f"Error reading the tools script: {e}")

    # Create the system prompt message
    system_message = {"role": "system", "content": system_prompt}

    # Prepend the system message to each conversation
    for convo in conversations:
        if not convo or convo[0].get("role") != "system":
            convo.insert(0, system_message)

    # Apply the tokenizer's chat template
    processed_prompts = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True
    )

    # Create a new dataset with processed prompts
    return dataset.map(
        lambda x, idx: {prompt_column: processed_prompts[idx]},
        with_indices=True
    )


def prepare_data_for_local_agent(
    dataset,  
    tokenizer,
    prompt_column: str = "prompt",  
    system_prompt: str = default_system_prompt,
    tools: List[Callable] = None,
    include_source_code: bool = True,
) -> list:
    """
    Prepares the Hugging Face dataset for the local agent by constructing conversations 
    and applying the system prompt.

    Args:
        dataset: A Hugging Face dataset object
        tokenizer: A tokenizer with an `apply_chat_template` method to process conversations
        prompt_column (str): Name of the column containing prompts. Defaults to "prompt"
        system_prompt (str, optional): The base system prompt. Defaults to default_system_prompt
        tools (List[Callable], optional): List of callable functions to include in system prompt
        include_source_code (bool, optional): If True, includes source code of tools, else includes docstrings

    Returns:
        Dataset: Modified dataset with processed prompts in the prompt column
    """
    # Convert dataset prompts to conversational format
    conversations = [[{"role": "user", "content": prompt}] for prompt in dataset[prompt_column]]

    # Process tools if provided
    if tools:
        tools_documentation = "\nAvailable tools:\n\n"
        for tool in tools:
            tools_documentation += f"Tool: {tool.__name__}\n"
            if include_source_code:
                try:
                    tools_documentation += f"Source code:\n{getsource(tool)}\n\n"
                except Exception as e:
                    tools_documentation += f"Error getting source: {str(e)}\n\n"
            else:
                doc = tool.__doc__ or "No documentation available"
                tools_documentation += f"Documentation:\n{doc}\n\n"
        
        system_prompt += tools_documentation

    # Create the system prompt message
    system_message = {"role": "system", "content": system_prompt}

    # Prepend the system message to each conversation
    for convo in conversations:
        if not convo or convo[0].get("role") != "system":
            convo.insert(0, system_message)

    # Apply the tokenizer's chat template
    processed_prompts = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True
    )

    # Create a new dataset with processed prompts
    return dataset.map(
        lambda x, idx: {prompt_column: processed_prompts[idx]},
        with_indices=True
    )

def generate_agent_responses(
    dataset: list,
    llm: LLM,
    sampling_params: SamplingParams,
    tools_script_path: str = None,
    parsing_string: str = "<code>",
    stop_string: str = "</code>",
    code_executer = LocalExecutor(),
) -> list:
    """
    Generates responses for the agent with potential code execution.
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
        parsing_string (str, optional): String used to locate the code in the conversation. Defaults to "<code>".
        api_key (str): API key for executing code in the sandbox.
        dependancies (list, optional): List of dependencies to install before code execution. Defaults to None.
        template (str, optional): Optional template for the sandbox environment. Defaults to None.
    Returns:
        list: A list of complete conversations (strings).
    """
    # adding stop string to sampling params
    sampling_params.stop = [stop_string]
    # Read the tools script if provided.
    tools_script = read_script(tools_script_path) if tools_script_path else None

    completed_chats = []    # Chats that are fully complete.
    current_batch = dataset  # Start with your initial batch of prompts.

    while current_batch:
        # Generate outputs for the current batch.
        outputs = llm.generate(current_batch, sampling_params,use_tqdm=False)
        next_batch = []  # To store chats that still need code execution.

        # Process each output in the batch.
        for output in outputs:
            # Reconstruct the conversation (prompt + generated response).
            conversation = output.prompt + output.outputs[0].text

            # Check if response is waiting for code execution.
            if output.outputs[0].stop_reason == stop_string:
                # Extract the code to execute.
                code_to_execute = get_code(conversation, tools_script=tools_script, parsing_string=parsing_string)
                # Execute the code via the sandbox.
                executed_result = code_executer.execute(code_to_execute)
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
