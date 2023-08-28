import os
import re
import copy
import time

import gradio as gr
from text_generation import Client
from transformers import load_tool
from share_btn import community_icon_html, loading_icon_html, share_js, share_btn_css


HF_TOKEN = os.environ.get("HF_TOKEN", None)

FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"

FIM_INDICATOR = "<FILL_HERE>"

FORMATS = """## Model Formats

The model is pretrained on code and is formatted with special tokens in addition to the pure code data,\
such as prefixes specifying the source of the file or tokens separating code from a commit message.\
Use these templates to explore the model's capacities:

### 1. Prefixes 🏷️
For pure code files, use any combination of the following prefixes:

```
<reponame>REPONAME<filename>FILENAME<gh_stars>STARS\ncode<|endoftext|>
```
STARS can be one of: 0, 1-10, 10-100, 100-1000, 1000+

### 2. Commits 💾
The commits data is formatted as follows:

```
<commit_before>code<commit_msg>text<commit_after>code<|endoftext|>
```

### 3. Jupyter Notebooks 📓
The model is trained on Jupyter notebooks as Python scripts and structured formats like:

```
<start_jupyter><jupyter_text>text<jupyter_code>code<jupyter_output>output<jupyter_text>
```

### 4. Issues 🐛
We also trained on GitHub issues using the following formatting:
```
<issue_start><issue_comment>text<issue_comment>...<issue_closed>
```

### 5. Fill-in-the-middle 🧩
Fill in the middle requires rearranging the model inputs. The playground handles this for you - all you need is to specify where to fill:
```
code before<FILL_HERE>code after
```
"""

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[
        gr.themes.GoogleFont("Open Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
)

tool = load_tool("vwxyzjn/pyserini-wikipedia-kilt-doc")
tool_fn = lambda x: tool(x).split("\n")[1][:600] # limit the amount if token, system_prompts

clients = {
    "StarCoderBase TriviaQA": [
        Client(
            "https://api-inference.huggingface.co/models/vwxyzjn/starcoderbase-triviaqa",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
        ),
        {"Wiki": tool_fn},
        """\
Answer the following question:

Q: In which branch of the arts is Patricia Neary famous?
A: Ballets
A2: <request><Wiki>Patricia Neary<call>Patricia Neary (born October 27, 1942) is an American ballerina, choreographer and ballet director, who has been particularly active in Switzerland. She has also been a highly successful ambassador for the Balanchine Trust, bringing George Balanchine's ballets to 60 cities around the globe.<response>
Result=Ballets<submit>

Q: Who won Super Bowl XX?
A: Chicago Bears
A2: <request><Wiki>Super Bowl XX<call>Super Bowl XX was an American football game between the National Football Conference (NFC) champion Chicago Bears and the American Football Conference (AFC) champion New England Patriots to decide the National Football League (NFL) champion for the 1985 season. The Bears defeated the Patriots by the score of 46–10, capturing their first NFL championship (and Chicago's first overall sports victory) since 1963, three years prior to the birth of the Super Bowl. Super Bowl XX was played on January 26, 1986 at the Louisiana Superdome in New Orleans.<response>
Result=Chicago Bears<submit>
"""
    ],
}

def parse_tool_call(text, request_token="<request>", call_token="<call>"):
    """
    Parse request string. Expected format: <request><tool_name>query<call>
    """
    result = re.search(f"(?<={request_token}).*?(?={call_token})", text, re.DOTALL)

    # if we can't find a <request>/<call> span we return none
    if result is None:
        return None, None
    else:
        extracted_text = result.group()

    result = re.search(r"<(.*?)>", extracted_text)

    # if we can't find a tool name we return none
    if result is None:
        return None, None
    else:
        tool = result.group(1)

    # split off the tool name
    query = ">".join(extracted_text.split(">")[1:])

    return tool, query



def generate(
    prompt, system_prompt, version, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    client, tools, _ = clients[version]
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)
    fim_mode = False

    # TextEnv tool
    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
        stop_sequences=["<call>"]
    )
    generation_still_running = True
    while generation_still_running:
        try:
            stream = client.generate_stream(system_prompt + prompt, **generate_kwargs)


            # call env phase
            output = system_prompt + prompt
            previous_token = ""
            for response in stream:
                if response.token.text == "<|endoftext|>":
                    return output
                else:
                    output += response.token.text
                previous_token = response.token.text
                # text env logic:
                tool, query = parse_tool_call(output[len(system_prompt + prompt):])
                print("tool", tool, query)
                if tool is not None and query is not None:
                    if tool not in tools:
                        response = f"Unknown tool {tool}."
                    try:
                        response = tools[tool](query)
                        output += response + "<response>"
                    except Exception as error:
                        response = f"Tool error: {str(error)}"
                yield output[len(system_prompt + prompt):]

            call_output = copy.deepcopy(output)
            # response phase
            generate_kwargs["stop_sequences"] = ["<submit>"]
            stream = client.generate_stream(output, **generate_kwargs)
            previous_token = ""
            for response in stream:
                if response.token.text == "<|endoftext|>":
                    return output
                else:
                    output += response.token.text
                previous_token = response.token.text
                yield output[len(system_prompt + prompt):]

            return output
        except Exception as e:
            if "loading" in str(e):
                gr.Warning("waiting for model to load... (this could take up to 20 minutes, after which things are much faster)")
                time.sleep(7)
                continue
            else:
                raise gr.Error(str(e))           


examples = [
    "X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.1)\n\n# Train a logistic regression model, predict the labels on the test set and compute the accuracy score",
    "// Returns every other value in the array as a new array.\nfunction everyOther(arr) {",
    "Poor English: She no went to the market. Corrected English:",
    "def alternating(list1, list2):\n   results = []\n   for i in range(min(len(list1), len(list2))):\n       results.append(list1[i])\n       results.append(list2[i])\n   if len(list1) > len(list2):\n       <FILL_HERE>\n   else:\n       results.extend(list2[i+1:])\n   return results",
]


def process_example(args):
    for x in generate(args):
        pass
    return x


css = ".generating {visibility: hidden}"

monospace_css = """
#q-input textarea {
    font-family: monospace, 'Consolas', Courier, monospace;
}
"""


css += share_btn_css + monospace_css + ".gradio-container {color: black}"


description = """
<div style="text-align: center;">
    <h1> ⭐ StarCoderBase TriviaQA <span style='color: #e6b800;'>Models</span> Playground</h1>
</div>
<div style="text-align: left;">
    <p>This is a demo to generate text and code with the following StarCoderBase TriviaQA models:</p>
    <ul>
        <li><a href="https://huggingface.co/bigcode/starcoderplus" style='color: #e6b800;'>StarCoderPlus</a>: A finetuned version of StarCoderBase on English web data, making it strong in both English text and code generation.</li>
        <li><a href="https://huggingface.co/bigcode/starcoderbase" style='color: #e6b800;'>StarCoderBase</a>: A code generation model trained on 80+ programming languages, providing broad language coverage for code generation tasks.</li>
        <li><a href="https://huggingface.co/bigcode/starcoder" style='color: #e6b800;'>StarCoderBase TriviaQA</a>: A finetuned version of StarCoderBase specifically focused on Python, while also maintaining strong performance on other programming languages.</li>
    </ul>
    <p><b>Please note:</b> These models are not designed for instruction purposes. If you're looking for instruction or want to chat with a fine-tuned model, you can visit the <a href="https://huggingface.co/spaces/HuggingFaceH4/starchat-playground">StarChat Playground</a>.</p>
</div>
"""
disclaimer = """⚠️<b>Any use or sharing of this demo constitues your acceptance of the BigCode [OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) License Agreement and the use restrictions included within.</b>\
 <br>**Intended Use**: this app and its [supporting model](https://huggingface.co/bigcode) are provided for demonstration purposes; not to serve as replacement for human expertise. For more details on the model's limitations in terms of factuality and biases, see the [model card.](hf.co/bigcode)"""

with gr.Blocks(theme=theme, analytics_enabled=False, css=css) as demo:
    with gr.Column():
        gr.Markdown(description)
        with gr.Row():
            version = gr.Dropdown(
                        list(clients.keys()),
                        value=list(clients.keys())[0],
                        label="Model",
                        info="Choose a model from the list",
                        )
            system_prompt = gr.Textbox(
                value=clients[list(clients.keys())[0]][2],
                label="System prompt",
            )
            
        with gr.Row():
            with gr.Column():
                instruction = gr.Textbox(
                    value="Q: In which country is Oberhofen situated?",
                    # placeholder="Enter your question here. E.g., Q: In which country is Oberhofen situated?",
                    lines=5,
                    label="Input",
                    elem_id="q-input",
                )
                submit = gr.Button("Generate", variant="primary")
                output = gr.Code(elem_id="q-output", lines=30, label="Output")
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("Advanced settings", open=False):
                            with gr.Row():
                                column_1, column_2 = gr.Column(), gr.Column()
                                with column_1:
                                    temperature = gr.Slider(
                                        label="Temperature",
                                        value=0.2,
                                        minimum=0.0,
                                        maximum=1.0,
                                        step=0.05,
                                        interactive=True,
                                        info="Higher values produce more diverse outputs",
                                    )
                                    max_new_tokens = gr.Slider(
                                        label="Max new tokens",
                                        value=256,
                                        minimum=0,
                                        maximum=8192,
                                        step=64,
                                        interactive=True,
                                        info="The maximum numbers of new tokens",
                                    )
                                with column_2:
                                    top_p = gr.Slider(
                                        label="Top-p (nucleus sampling)",
                                        value=0.90,
                                        minimum=0.0,
                                        maximum=1,
                                        step=0.05,
                                        interactive=True,
                                        info="Higher values sample more low-probability tokens",
                                    )
                                    repetition_penalty = gr.Slider(
                                        label="Repetition penalty",
                                        value=1.2,
                                        minimum=1.0,
                                        maximum=2.0,
                                        step=0.05,
                                        interactive=True,
                                        info="Penalize repeated tokens",
                                    )
                                    
                gr.Markdown(disclaimer)
                with gr.Group(elem_id="share-btn-container"):
                    community_icon = gr.HTML(community_icon_html, visible=True)
                    loading_icon = gr.HTML(loading_icon_html, visible=True)
                    share_button = gr.Button(
                        "Share to community", elem_id="share-btn", visible=True
                    )
                gr.Examples(
                    examples=examples,
                    inputs=[instruction],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )
                # gr.Markdown(FORMATS)

    submit.click(
        generate,
        inputs=[instruction, system_prompt, version, temperature, max_new_tokens, top_p, repetition_penalty],
        outputs=[output],
    )
    share_button.click(None, [], [], _js=share_js)
demo.queue(concurrency_count=16).launch(share=True)


"""
Answer the following question:

Q: In which branch of the arts is Patricia Neary famous?
A: Ballets
A2: <request><Wiki>Patricia Neary<call>Patricia Neary (born October 27, 1942) is an American ballerina, choreographer and ballet director, who has been particularly active in Switzerland. She has also been a highly successful ambassador for the Balanchine Trust, bringing George Balanchine's ballets to 60 cities around the globe.<response>
Result=Ballets<submit>

Q: Who won Super Bowl XX?
A: Chicago Bears
A2: <request><Wiki>Super Bowl XX<call>Super Bowl XX was an American football game between the National Football Conference (NFC) champion Chicago Bears and the American Football Conference (AFC) champion New England Patriots to decide the National Football League (NFL) champion for the 1985 season. The Bears defeated the Patriots by the score of 46–10, capturing their first NFL championship (and Chicago's first overall sports victory) since 1963, three years prior to the birth of the Super Bowl. Super Bowl XX was played on January 26, 1986 at the Louisiana Superdome in New Orleans.<response>
Result=Chicago Bears<submit>

Q: In what state is Philadelphia located?"""