from torch.utils.data import Dataset

    
def get_vicuna_prompt(system_prompt, message):
    texts = f'{system_prompt}\n'
    texts += f'USER: {message.strip()}\nASSISTANT:'.strip()
    return texts

    
class VicunaJailbreakDataset(Dataset):
    def __init__(
        self, datas, tokenizer, system_prompt
    ):
        self.tokenizer = tokenizer

        datas = [get_vicuna_prompt(system_prompt, jailbreak) for jailbreak in datas]

        self.inputs = [self.tokenizer(i, return_tensors='pt', max_length=3096, truncation=True) for i in datas]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        return self.inputs[index]