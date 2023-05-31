from transformers import StoppingCriteria, StoppingCriteriaList


class StringStoppingCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generations in the batch are completed."""

    def __init__(self, start_lengths, stop_strings, tokenizer):
        self.start_lengths = start_lengths
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the stop strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids)
        decoded_generations = [decoded_generation[start_length:] for start_length, decoded_generation in zip(self.start_lengths, decoded_generations)]
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.stop_strings]))
        return all(done)


class TextHistory:
    def __init__(self, text, system=True):
        
        self.system_spans = []
        self.spans = []
        self.text = text
        self.completed = False

        if len(text)>0:
            self.spans.append((0, len(text)))
            self.system_spans.append(system)
        

    def append(self, text, system=True):
        if len(text)==0:
            raise ValueError("Can't append empty text to history.")
        original_text_length = len(self.text)
        self.text += text
        self.spans.append((original_text_length, len(text)))
        self.system_spans.append(system)


    def complete(self):
        self.completed = True

    @property
    def last_text(self):
        start, end = self.spans[-1]
        return self.text[start: end]



class Environment:
    def __init__(self, model, tokenizer, tools, reward, prompt, generation_kwargs=None):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.tools = tools
        self.reward = reward
        self.max_length = 1024
        self.request_token = "<request>"
        self.call_token = "<call>"
        self.response_token = "<response>"

        if generation_kwargs is None:
            self.generation_kwargs = dict()
        else:
            self.generation_kwargs = generation_kwargs            


    def run(self, tasks):
        
        histories = [TextHistory(self.prompt + task, system=True) for task in tasks]

        while any([not history.completed for history in histories]):
            histories = self.generate(histories)
            # TODO: make this parallel rather than for-loop
            for i, history in enumerate(histories):
                new_history, reward, terminated, truncated, info = self.step(history)
                system_generated = [(len(history), len(history)) for history in histories]


    def step(self, history):
        if self.task_ended(history):
            return history, 
        history

    def compute_reward(self, history):
        return self.reward(history.last_text)
    
    def generate(self, history):
        # TODO: implement a batched geneartion function with the custom stopping criteria
        # TODO: exclude completed histories from generating

        self.generation_kwargs["stopping_criteria"] =  StoppingCriteriaList([StringStoppingCriteria(input_lengths, [self.call_token], self.tokenizer)])

        return history

    def task_ended(self, task):
        if len(task)>self.max_length:
            return True, True
        elif self.tokenizer.eos_token in task:
            return True, False
        else:
            return False, False