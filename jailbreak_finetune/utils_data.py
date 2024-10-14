from torch.utils.data import Dataset
import torch
import copy

IGNORE_INDEX = -100


def get_llama_prompt(system_prompt, message):
    texts = f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n'
    # for user_input, response in chat_history:
    #     texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts += f'{message.strip()} [/INST]'
    return texts

def get_vicuna_prompt(system_prompt, message):
    texts = f'{system_prompt}\n'
    texts += f'USER: {message.strip()}\nASSISTANT:'.strip()
    return texts

def _tokenize_fn(sources, targets, tokenizer, max_length):
    # source + target
    tokenized_list = [
        tokenizer(
            source+target,
            return_tensors="pt",
            pad_to_max_length=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        for source, target in zip(sources, targets)
    ]

    # source + target ids

    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    
    # source length

    sources_list = [
        tokenizer(
            text,
            return_tensors="pt",
            pad_to_max_length=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        for text in sources
    ]

    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in sources_list
    ]

    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def preprocess(
    sources,
    targets,
    tokenizer,
    max_len,
    split="train"
):
    """Preprocess the data by tokenizing."""

    # sources_tokenized = _tokenize_fn(sources, tokenizer, max_source_len)
    # targets_tokenized = _tokenize_fn(targets, tokenizer, max_target_len)
    sources_targets_tokenized = _tokenize_fn(sources, targets, tokenizer, max_len)

    if split == "train":
        # input_ids = [torch.cat([source, target], dim=-1) for source, target in zip(sources_tokenized["input_ids"], targets_tokenized["input_ids"])]
        input_ids = sources_targets_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_targets_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
    else:
        # input_ids = sources_tokenized["input_ids"]
        # labels = targets_tokenized["input_ids"]
        pass
    
    return dict(input_ids=input_ids, labels=labels)


class Llama2JailbreakDataset(Dataset):
    def __init__(
        self, local_rank, data, tokenizer, source_len=2048, target_len=512, system_prompt="", split="train"
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
        """
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = target_len

        if local_rank == 0:   
            print("source len: ", source_len, "target len: ", target_len)
        
        sources = [get_llama_prompt(system_prompt, example) for example in data[0]]
        # sources = [example for example in data[0]]

        targets = [f"{example}{tokenizer.eos_token}" for example in data[1]]

        if local_rank == 0:
            print(sources[0])
            print(targets[0])

        data_dict = preprocess(sources, targets, tokenizer, source_len+target_len, split)

        self.input_ids = data_dict["input_ids"]
        
        self.labels = data_dict["labels"]
        # if local_rank == 0 :
        #     print("input_ids:", self.input_ids[0].tolist())
        #     print("labels:", self.labels[0].tolist())
        #     print("input_ids decode:", self.tokenizer.decode(self.input_ids[0],  skip_special_tokens=True, clean_up_tokenization_spaces=True))
        self.tokenizer = tokenizer

        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):

        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
        }


class VicunaJailbreakDataset(Dataset):
    def __init__(
        self, local_rank, data, tokenizer, source_len=2048, target_len=512, system_prompt="", split="train"
    ):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = target_len
        
        # sources = [get_vicuna_prompt(system_prompt, example) for example in data[0]]

        # targets = [f"{example}{tokenizer.eos_token}" for example in data[1]]
        if local_rank == 0:
            print("source len: ", source_len, "target len: ", target_len)
        
        sources = [get_vicuna_prompt(system_prompt, example) for example in data[0]]
        # sources = [example for example in data[0]]

        targets = [f"{example}{tokenizer.eos_token}" for example in data[1]]
        if local_rank == 0:
            print(sources[0])
            print(targets[0])

        # data_dict = preprocess(sources, targets, tokenizer, source_len, target_len, split)
        data_dict = preprocess(sources, targets, tokenizer, source_len+target_len, split)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        # if local_rank == 0 :
        #     print("input_ids:", self.input_ids[0].tolist())
        #     print("labels:", self.labels[0].tolist())
        #     print("input_ids decode:", self.tokenizer.decode(self.input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        self.tokenizer = tokenizer

        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):

        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
        }