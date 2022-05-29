import torch
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List

END_TOKEN = 50256


class Generator():
    '''
    A class to represent generator model

    Attributes:
        generator_model
            Pretrained model for text generation. For example transformers.GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer
            Pretrained tokenizer for the generator_model. For example transformers.GPT2Tokenizer.from_pretrained('gpt2')
        vectorizer_model: List[np.array]
            A model for text vectorization. For example SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
            (It is recommended to choose a fast model)
        device: str
            Either 'cuda:0' (if available) or 'cpu'
    '''
    def __init__(self, generator_model, tokenizer,
                 vectorizer_model: SentenceTransformer =
                 SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        self.tokenizer = tokenizer
        self.generator_model = generator_model.to(self.device)
        self.vectorizer_model = vectorizer_model.to(self.device)

    def generate_with_vector(self, prompt: str, base_vector: torch.Tensor,
                             first: bool = False, k: int = 50, j: int = 20,
                             max_len: int = 30, min_length: int = 3,
                             limit_len: int = 100) -> str:
        '''
        Generates according to the given vector (center of a cluster can be viewed as a topic-vector)

        Parameters:
            prompt: str
                Input text
            base_vector: torch.Tensor
                vector to guide the generation process
            first: bool
                Is it a first phrase to generate (or it continues previously generated according to the given structure)
            k: int
                Number of most probable tokens from vocabulary
            j: int
                Number of best tokens to randomly choose from
            max_len: int
                After max_len tokens are generated, model stops generating if END_TOKEN is in best k
            min_length: int
                Minimum number of tokens to generate
            limit_len: int
                If limit_len tokens are generated, generation process stops

        '''
        if first:
            input_length = 0
        else:
            input_length = len(prompt)
        tokenized_prompt = self.tokenizer.encode(prompt)
        cos = torch.nn.CosineSimilarity(dim=-1)
        next_token = -1
        length_generated = 0
        num_generated = 0
        base_vector = base_vector.to(self.device)
        while not (length_generated != 0 and ('.' in prompt[-length_generated:]
                                              or '!' in prompt[-length_generated:]
                                              or '?' in prompt[-length_generated:]) or (next_token == END_TOKEN
                                                                                        and num_generated > min_length)
                   or num_generated > limit_len):
            top_k = \
            torch.topk(self.generator_model(input_ids=torch.tensor(tokenized_prompt).to(self.device)).logits[-1], k)[1]
            if num_generated > max_len and END_TOKEN in top_k or num_generated > limit_len:
                next_token = END_TOKEN
            else:
                texts_to_vectorize = []
                for token in top_k:
                    texts_to_vectorize.append(prompt[input_length:] +
                                              self.tokenizer.decode(token, skip_special_tokens=True,
                                                                    clean_up_tokenization_spaces=True))
                current_vectors = self.vectorizer_model.encode(texts_to_vectorize, convert_to_tensor=True)
                top_k_results = cos(current_vectors, base_vector)
                top_j = top_k[torch.topk(top_k_results, j)[1]]
                next_token = top_j[random.randint(0, j - 1)].item()
                tokenized_prompt.append(next_token)
                generated_token = self.tokenizer.decode(next_token, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)
                length_generated += len(generated_token)
                num_generated += 1
                prompt = prompt + generated_token
        return prompt

    def simple_generate(self, prompt: str, k: int = 50, max_len: int = 30,
                        min_length: int = 3, limit_len: int = 100):
        '''
        Simple generation to compare with the proposed method

        Parameters:
            prompt: str
                Input text
            k: int
                Number of best tokens to randomly choose from
            max_len: int
                After max_len tokens are generated, model stops generating if END_TOKEN is in best k
            min_length: int
                Minimum number of tokens to generate
            limit_len: int
                If limit_len tokens are generated, generation process stops

        '''
        tokenized_prompt = self.tokenizer.encode(prompt)
        next_token = -1
        length_generated = 0
        num_generated = 0
        while not (length_generated != 0 and ('.' in prompt[-length_generated:]
                                              or '!' in prompt[-length_generated:]
                                              or '?' in prompt[-length_generated:]) or (
                           next_token == END_TOKEN and num_generated > min_length) or num_generated > limit_len):
            top_k = \
            torch.topk(self.generator_model(input_ids=torch.tensor(tokenized_prompt).to(self.device)).logits[-1], k)[1]
            if num_generated > max_len and END_TOKEN in top_k or num_generated > limit_len:
                next_token = END_TOKEN
            else:
                next_token = top_k[random.randint(0, k - 1)].item()
                tokenized_prompt.append(next_token)
                generated_token = self.tokenizer.decode(next_token,
                                                        skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)
                length_generated += len(generated_token)
                num_generated += 1
                prompt = prompt + generated_token
        return prompt

    def generate_with_structure(self, prompt: str, structure: List[torch.Tensor]):
        '''
        Generates according to the given structure (vectors of topics)
        '''
        vector_generate_result = []
        for i, topic_vec in enumerate(structure):
            if i == 0:
                vector_generate_result.append(
                    self.generate_with_vector(prompt,
                                              topic_vec.to(self.device),
                                              first=True)
                )
                prompt = vector_generate_result[-1]
            else:
                vector_generate_result.append(
                    self.generate_with_vector(prompt,
                                              topic_vec.to(self.device),
                                              first=False)[len(prompt):]
                )
                prompt = ''.join(vector_generate_result)
        return vector_generate_result

    def simple_generate_n_gram(self, prompt: str, n: int = 3):
        '''
        Simple generation to compare with the proposed method
        '''
        simple_generate_result = []
        for i in range(n):
            if i == 0:
                simple_generate_result.append(self.simple_generate(prompt))
                prompt = simple_generate_result[-1]
            else:
                simple_generate_result.append(self.simple_generate(prompt)[len(prompt):])
                prompt = ''.join(simple_generate_result)
        return simple_generate_result
