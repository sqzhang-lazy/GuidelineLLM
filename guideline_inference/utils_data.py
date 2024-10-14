from torch.utils.data import Dataset


def get_llama_prompt(system_prompt, message):
    texts = f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n'
    # texts = f'[INST] '
    # for user_input, response in chat_history:
    #     texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts += f'{message.strip()} [/INST]'
    return texts

class Llama2Dataset(Dataset):
    def __init__(
        self, datas, tokenizer, system_prompts
    ):
        self.tokenizer = tokenizer


        datas = [jailbreak for jailbreak in datas]

        self.inputs = [self.tokenizer(i, return_tensors='pt') for i in datas]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        return self.inputs[index]

class Llama2NoSysDataset(Dataset):
    def __init__(
        self, datas, tokenizer, system_prompt
    ):
        self.tokenizer = tokenizer

        self.system_prompt = system_prompt

        datas = ['[INST] '+ i + ' [/INST]' for i in datas]

        self.inputs = [self.tokenizer(i, return_tensors='pt') for i in datas]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        return self.inputs[index]

class Llama2GuidelineDataset(Dataset):
    def __init__(
        self, datas, tokenizer, system_prompts
    ):
        # assert len(datas) == len(system_prompts)
        self.tokenizer = tokenizer

        datas = [get_llama_prompt(system_prompt, jailbreak) for system_prompt, jailbreak in zip(system_prompts, datas)]
        # datas = [get_llama_prompt(system_prompts, jailbreak) for jailbreak in datas]

        self.inputs = [self.tokenizer(i, return_tensors='pt') for i in datas]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        return self.inputs[index]
    
def get_vicuna_prompt(system_prompt, message):
    texts = f'{system_prompt}\n'
    texts += f'USER: {message.strip()}\nASSISTANT:'.strip()
    return texts

class VicunaDataset(Dataset):
    def __init__(
        self, datas, tokenizer, system_prompt=""
    ):
        self.tokenizer = tokenizer

        self.system_prompt = system_prompt

        datas = [get_vicuna_prompt(self.system_prompt, i) for i in datas]

        self.inputs = [self.tokenizer(i, return_tensors='pt') for i in datas]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        return self.inputs[index]
    
class VicunaGuidelineDataset(Dataset):
    def __init__(
        self, datas, tokenizer, system_prompts
    ):
        assert len(datas) == len(system_prompts)
        self.tokenizer = tokenizer

        datas = [get_vicuna_prompt(system_prompt, jailbreak) for system_prompt, jailbreak in zip(system_prompts, datas)]

        self.inputs = [self.tokenizer(i, return_tensors='pt') for i in datas]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        return self.inputs[index]