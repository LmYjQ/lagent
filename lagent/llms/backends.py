from typing import Any, Dict, List, Optional, Tuple, Union

from lagent.llms.base_llm import BaseLLM
from lagent.schema import ModelStatusCode
from lagent.utils.util import filter_suffix
from jinja2 import Template
from copy import copy, deepcopy

import os
import asyncio
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock

import aiohttp
import requests
class LLMMixin:
    def render_chat(self, template, messages, bos_token="<s>", add_generation_prompt=False):
        """
        渲染聊天消息模板
        
        :param messages: 消息列表,每个消息是一个字典,包含'role'和'content'键
        :param bos_token: 开始标记
        :param add_generation_prompt: 是否添加生成提示
        :return: 渲染后的字符串
        """
        template = Template(template)
        return template.render(
            messages=messages,
            bos_token=bos_token,
            add_generation_prompt=add_generation_prompt
        )
    def update_gen_params(self, **kwargs):
        gen_params = copy(self.gen_params)
        gen_params.update(kwargs)
        return gen_params

    def chat_completion(self,inputs: List[dict],
                    session_id=0,
                    sequence_start: bool = True,
                    sequence_end: bool = True,
                    stream: bool = True,
                    ignore_eos: bool = False,
                    skip_special_tokens: Optional[bool] = False,
                    timeout: int = 30,
                    **kwargs):
        raise NotImplementedError

    def completion(self,inputs: Union[str, List[str]],
                 session_id: int = 2967,
                 sequence_start: bool = True,
                 sequence_end: bool = True,
                 ignore_eos: bool = False,
                 skip_special_tokens: Optional[bool] = False,
                 timeout: int = 30,
                 **kwargs):
        raise NotImplementedError


class BackendConfig:
    pass

def dispatch(backend: Union[str, object], 
            path:str='', 
            model_name: Optional[str] = None,
            server_name: Optional[str] = '0.0.0.0',
            server_port: Optional[int] = 23333,
            url: Optional[str] = None,
            api_keys: Optional[Union[List[str], str]] = None,
            chat_template_config: Optional[dict] = dict(),
            chat_template:str = None,
            engine_config: Optional[dict] = dict(),
            **kwargs):
    if backend=='lmdeploy_server':
        return LMDeployServerBackend(
            path=path,
            model_name=model_name,
            server_name=server_name,
            server_port=server_port,
            serve_cfg = engine_config,
            chat_template_config = chat_template_config,
            chat_template = chat_template,
            **kwargs
        )
    elif backend=='lmdeploy_client':
        return LMDeployClientBackend(
            model_name=kwargs.get('model_name'),
            url=kwargs.get('url', '0.0.0.0:23333'),
                **kwargs
        )
    
    elif backend=='transformer':
        return TransformerBackend(**kwargs)
    elif backend=='api':
        return ApiBackend(**kwargs)
    else:
        raise NotImplementedError(f'backend {backend} not supported')
    
class LMDeployServerBackend(LLMMixin):
    def __init__(self, path: str,
                model_name: Optional[str] = None,
                server_name: str = '0.0.0.0',
                server_port: int = 23333,
                log_level: str = 'WARNING',
                chat_template_config: Optional[dict] = dict(),
                chat_template:str = None,
                serve_cfg=dict(),
                **kwargs):
        #super().__init__(path=path, **kwargs)
        self.model_name = model_name
        self.chat_template_config = chat_template_config
        self.chat_template = chat_template
        # TODO get_logger issue in multi processing
        import lmdeploy
        self.client = lmdeploy.serve(
            model_path=path,
            model_name=model_name,
            server_name=server_name,
            server_port=server_port,
            log_level=log_level,
            **serve_cfg)
        
    def chat_completion(self,
                    inputs: List[dict],
                    session_id=2679,
                    sequence_start: bool = True,
                    sequence_end: bool = True,
                    stream: bool = True,
                    ignore_eos: bool = False,
                    skip_special_tokens: Optional[bool] = False,
                    timeout: int = 30,
                    **kwargs):
        # gen_params = self.update_gen_params(**kwargs)
        # max_new_tokens = gen_params.pop('max_new_tokens')
        # gen_params.update(max_tokens=max_new_tokens)
        # prompt = self.template_parser(inputs)

        # finished = False
        # stop_words = self.gen_params.get('stop_words')
        # stop_words = gen_params.get('stop_words')
        resp = ''

        for text in self.client.chat_completions_v1(
                self.model_name,
                inputs,
                session_id=session_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                stream=stream,
                ignore_eos=ignore_eos,
                skip_special_tokens=skip_special_tokens,
                timeout=timeout,
                **kwargs):
            resp += text['choices'][0]['delta']['content']
            if not resp:
                continue
            # remove stop_words
            # for sw in stop_words:
            #     if sw in resp:
            #         resp = filter_suffix(resp, stop_words)
            #         finished = True
            #         break
            yield ModelStatusCode.STREAM_ING, resp, None
            # if finished:
            #     break
        yield ModelStatusCode.END, resp, None

    def completion(self,
                 inputs: Union[str, List[str]],
                 session_id: int = 2967,
                 sequence_start: bool = True,
                 sequence_end: bool = True,
                 ignore_eos: bool = False,
                 skip_special_tokens: Optional[bool] = False,
                 timeout: int = 30,
                 **kwargs):
        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False

        # gen_params = self.update_gen_params(**kwargs)
        # max_new_tokens = gen_params.pop('max_new_tokens')
        # gen_params.update(max_tokens=max_new_tokens)

        resp = [''] * len(inputs)
        for text in self.client.completions_v1(
                self.model_name,
                inputs,
                session_id=session_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                stream=False,
                ignore_eos=ignore_eos,
                skip_special_tokens=skip_special_tokens,
                timeout=timeout,
                **kwargs):
            resp = [
                resp[i] + item['text']
                for i, item in enumerate(text['choices'])
            ]
        # remove stop_words
        # resp = filter_suffix(resp, self.gen_params.get('stop_words'))
        if not batched:
            return resp[0]
        return resp
    
    def chat_completion_custom(self,
            inputs: List[dict],
            session_id=0,
            sequence_start: bool = True,
            sequence_end: bool = True,
            stream: bool = True,
            ignore_eos: bool = False,
            skip_special_tokens: Optional[bool] = False,
            timeout: int = 30,
            **kwargs):
        prompt = self.render_chat(template=self.chat_template,messages=inputs)
        print(prompt)
        stop_words = self.chat_template_config.get('stop_words')
        resp = ''
        finished = False

        for text in self.client.completions_v1(
                        self.model_name,
                        prompt,
                        session_id=session_id,
                        sequence_start=sequence_start,
                        sequence_end=sequence_end,
                        stream=stream,
                        ignore_eos=ignore_eos,
                        skip_special_tokens=skip_special_tokens,
                        timeout=timeout,
                        **kwargs):
            resp += text['choices'][0]['text']
            if not resp:
                continue
            # remove stop_words
            for sw in stop_words:
                if sw in resp:
                    resp = filter_suffix(resp, stop_words)
                    finished = True
                    break
            yield ModelStatusCode.STREAM_ING, resp, None
            if finished:
                break
        yield ModelStatusCode.END, resp, None

class LMDeployClientBackend(LMDeployServerBackend):
    """

    Args:
        url (str): communicating address 'http://<ip>:<port>' of
            api_server
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on.
    """

    def __init__(self, url: str, model_name: str, **kwargs):
        BaseLLM.__init__(self, path=url, **kwargs)
        from lmdeploy.serve.openai.api_client import APIClient
        self.client = APIClient(url)
        self.model_name = model_name

class TransformerBackend(BaseLLM, LLMMixin):
    def __init__(self, **kwargs):
        pass
        
    def chat_completion():
        pass
    def completion():
        pass



class ApiBackend(LLMMixin):
    def __init__(self,
                 model_type: str = 'gpt-3.5-turbo',
                 retry: int = 2,
                 json_mode: bool = False,
                 api_keys: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = [
                     dict(role='system', api_role='system'),
                     dict(role='user', api_role='user'),
                     dict(role='assistant', api_role='assistant')
                 ],
                 url: str = 'https://api.openai.com/v1/chat/completions',
                 proxies: Optional[Dict] = None,
                 chat_template:str = None,
                 generation_config_init = None,
                 **gen_params):
        self.model_type = model_type
        self.meta_template = meta_template
        self.retry = retry
        self.chat_template = chat_template

        # if isinstance(stop_words, str):
        #     stop_words = [stop_words]
        self.gen_params = generation_config_init


        if isinstance(api_keys, str):
            self.keys = [os.getenv('OPENAI_API_KEY') if api_keys == 'ENV' else api_keys]
        else:
            self.keys = api_keys

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0
        self.url = url
        self.model_type = model_type
        self.proxies = proxies
        self.json_mode = json_mode
        print(self.url, self.model_type, self.keys)

    def chat_completion(
        self,
        inputs: Union[List[dict], List[List[dict]]],
        **gen_params,
    ) -> Union[str, List[str]]:
        """Generate responses given the contexts.

        Args:
            inputs (Union[List[dict], List[List[dict]]]): a list of messages
                or list of lists of messages
            gen_params: additional generation configuration

        Returns:
            Union[str, List[str]]: generated string(s)
        """
        assert isinstance(inputs, list)
        if 'max_tokens' in gen_params:
            raise NotImplementedError('unsupported parameter: max_tokens')
        gen_params = {**self.gen_params, **gen_params}
        # gen_params = {**gen_params}
        with ThreadPoolExecutor(max_workers=20) as executor:
            tasks = [
                executor.submit(self._chat, messages, **gen_params)
                for messages in (
                    [inputs] if isinstance(inputs[0], dict) else inputs)
            ]
        ret = [task.result() for task in tasks]
        return ret[0] if isinstance(inputs[0], dict) else ret

    def _chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompt dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """
        assert isinstance(messages, list)
        gen_params = gen_params.copy()

        # Hold out 100 tokens due to potential errors in tiktoken calculation
        # max_tokens = min(gen_params.pop('max_new_tokens'), 4096)
        max_tokens = 4096
        if max_tokens <= 0:
            return ''

        max_num_retries = 0
        while max_num_retries < self.retry:
            with Lock():
                if len(self.invalid_keys) == len(self.keys):
                    raise RuntimeError('All keys have insufficient quota.')

                # find the next valid key
                while True:
                    self.key_ctr += 1
                    if self.key_ctr == len(self.keys):
                        self.key_ctr = 0

                    if self.keys[self.key_ctr] not in self.invalid_keys:
                        break

                key = self.keys[self.key_ctr]

            header = {
                'Authorization': f'Bearer {key}',
                'content-type': 'application/json',
            }

            if self.orgs:
                with Lock():
                    self.org_ctr += 1
                    if self.org_ctr == len(self.orgs):
                        self.org_ctr = 0
                header['OpenAI-Organization'] = self.orgs[self.org_ctr]

            try:
                gen_params_new = gen_params.copy()
                data = dict(
                    model=self.model_type,
                    messages=messages,
                    max_tokens=max_tokens,
                    n=1,
                    stop=gen_params_new.pop('stop_words'),
                    frequency_penalty=gen_params_new.pop('repetition_penalty'),
                    **gen_params_new,
                )
                if self.json_mode:
                    data['response_format'] = {'type': 'json_object'}
                raw_response = requests.post(
                    self.url,
                    headers=header,
                    data=json.dumps(data),
                    proxies=self.proxies)
            except requests.ConnectionError:
                print('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                print('JsonDecode error, got', str(raw_response.content))
                continue
            try:
                return response['choices'][0]['message']['content'].strip()
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(1)
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        self.invalid_keys.add(key)
                        self.logger.warn(f'insufficient_quota key: {key}')
                        continue

                    print('Find error message in response: ',
                          str(response['error']))
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def tokenize(self, prompt: str) -> list:
        """Tokenize the input prompt.

        Args:
            prompt (str): Input string.

        Returns:
            list: token ids
        """
        import tiktoken
        self.tiktoken = tiktoken
        enc = self.tiktoken.encoding_for_model(self.model_type)
        return enc.encode(prompt)


class vLLMBackend(BaseLLM, LLMMixin):
    def __init__(self, **kwargs):
        pass
        
    def chat_completion():
        pass
    def completion():
        pass

class HFTransformer(LLMMixin):
    """Model wrapper around HuggingFace general models.

    Adapted from Internlm (https://github.com/InternLM/InternLM/blob/main/
        chat/web_demo.py)

    Args:
        path (str): The name or path to HuggingFace's model.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
    """

    def __init__(self,
                 path: str,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 meta_template: Optional[Dict] = None,
                 stop_words_id: Union[List[int], int] = None,
                 chat_template:str = None,
                 generation_config_init = None,
                 **kwargs):
        # super().__init__(
        #     path=path,
        #     tokenizer_only=tokenizer_only,
        #     meta_template=meta_template,
        #     **kwargs)
        self.gen_params = generation_config_init
        self.chat_template = chat_template

        if isinstance(stop_words_id, int):
            stop_words_id = [stop_words_id]
        self.gen_params.update(stop_words_id=stop_words_id)
        # if self.gen_params['stop_words'] is not None and \
        #         self.gen_params['stop_words_id'] is not None:
        #     logger.warning('Both stop_words and stop_words_id are specified,'
        #                    'only stop_words_id will be used.')

        self._load_tokenizer(
            path=path,
            tokenizer_path=tokenizer_path,
            tokenizer_kwargs=tokenizer_kwargs)
        if not tokenizer_only:
            self._load_model(path=path, model_kwargs=model_kwargs)

        from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList  # noqa: E501
        self.logits_processor = LogitsProcessorList()
        self.stopping_criteria = StoppingCriteriaList()
        self.prefix_allowed_tokens_fn = None

        stop_words_id = []
        if self.gen_params.get('stop_words_id'):
            stop_words_id = self.gen_params.get('stop_words_id')
        elif self.gen_params.get('stop_words'):
            for sw in self.gen_params.get('stop_words'):
                stop_words_id.append(self.tokenizer(sw)['input_ids'][-1])
        self.additional_eos_token_id = stop_words_id

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        print('tokenizer_kwargs:',tokenizer_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path,
            trust_remote_code=True,
            **tokenizer_kwargs)

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                logger.warning(
                    f'Using eos_token_id {self.tokenizer.eos_token} '
                    'as pad_token_id.')
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                from transformers.generation import GenerationConfig
                self.gcfg = GenerationConfig.from_pretrained(path)

                if self.gcfg.pad_token_id is not None:
                    logger.warning(
                        f'Using pad_token_id {self.gcfg.pad_token_id} '
                        'as pad_token_id.')
                    self.tokenizer.pad_token_id = self.gcfg.pad_token_id
                else:
                    raise ValueError(
                        'pad_token_id is not set for this tokenizer. Try to '
                        'set pad_token_id via passing '
                        '`pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

    def _load_model(self, path: str, model_kwargs: dict):
        import torch
        from transformers import AutoModel
        model_kwargs.setdefault('torch_dtype', torch.float16)
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True, **model_kwargs)
        self.model.eval()

    def tokenize(self, inputs: str):
        assert isinstance(inputs, str)
        inputs = self.tokenizer(
            inputs, return_tensors='pt', return_length=True)
        return inputs['input_ids'].tolist()

    def chat_completion_custom(
        self,
        inputs: Union[str, List[str]],
        do_sample: bool = True,
        **kwargs,
    ):
        """Return the chat completions in non-stream mode.

        Args:
            inputs (Union[str, List[str]]): input texts to be completed.
            do_sample (bool): do sampling if enabled
        Returns:
            (a list of/batched) text/chat completion
        """
        for status, chunk, _ in self.stream_generate(inputs, do_sample,
                                                     **kwargs):
            response = chunk
        return response

    def stream_generate(
        self,
        inputs: List[str],
        do_sample: bool = True,
        **kwargs,
    ):
        """Return the chat completions in stream mode.

        Args:
            inputs (Union[str, List[str]]): input texts to be completed.
            do_sample (bool): do sampling if enabled
        Returns:
            tuple(Status, str, int): status, text/chat completion,
            generated token number
        """
        inputs = self.render_chat(template=self.chat_template,messages=inputs)

        import torch
        from torch import nn
        with torch.no_grad():
            batched = True
            if isinstance(inputs, str):
                inputs = [inputs]
                batched = False
            inputs = self.tokenizer(
                inputs, padding=True, return_tensors='pt', return_length=True)
            input_length = inputs['length']
            for k, v in inputs.items():
                inputs[k] = v.cuda()
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            batch_size = input_ids.shape[0]
            input_ids_seq_length = input_ids.shape[-1]
            generation_config = self.model.generation_config
            generation_config = deepcopy(generation_config)
            new_gen_params = self.update_gen_params(**kwargs)
            generation_config.update(**new_gen_params)
            generation_config.update(**kwargs)
            model_kwargs = generation_config.to_dict()
            model_kwargs['attention_mask'] = attention_mask
            _, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
                generation_config.bos_token_id,
                generation_config.eos_token_id,
            )
            if eos_token_id is None:
                if self.gcfg.eos_token_id is not None:
                    eos_token_id = self.gcfg.eos_token_id
                else:
                    eos_token_id = []
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            if self.additional_eos_token_id is not None:
                eos_token_id.extend(self.additional_eos_token_id)
            eos_token_id_tensor = torch.tensor(eos_token_id).to(
                input_ids.device) if eos_token_id is not None else None
            generation_config.max_length = (
                generation_config.max_new_tokens + input_ids_seq_length)
            # Set generation parameters if not already defined
            logits_processor = self.logits_processor
            stopping_criteria = self.stopping_criteria
            print('generation_config:',generation_config)
            logits_processor = self.model._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_seq_length,
                encoder_input_ids=input_ids,
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                logits_processor=logits_processor,
            )

            stopping_criteria = self.model._get_stopping_criteria(
                generation_config=generation_config,
                stopping_criteria=stopping_criteria)
            logits_warper = self.model._get_logits_warper(generation_config)

            unfinished_sequences = input_ids.new(batch_size).fill_(1)
            scores = None
            while True:
                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids, **model_kwargs)
                # forward pass to get next token
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                next_token_logits = outputs.logits[:, -1, :]

                # pre-process distribution
                next_token_scores = logits_processor(input_ids,
                                                     next_token_logits)
                next_token_scores = logits_warper(input_ids, next_token_scores)

                # sample
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                if do_sample:
                    next_tokens = torch.multinomial(
                        probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(probs, dim=-1)

                # update generated ids, model inputs,
                # and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]],
                                      dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(  # noqa: E501
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=False)
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(
                        eos_token_id_tensor.unsqueeze(1)).prod(dim=0))
                output_token_ids = input_ids.cpu().tolist()
                for i in range(len(output_token_ids)):
                    output_token_ids[i] = output_token_ids[i][:][
                        input_length[i]:]
                    # Find the first occurrence of
                    # an EOS token in the sequence
                    first_eos_idx = next(
                        (idx
                         for idx, token_id in enumerate(output_token_ids[i])
                         if token_id in eos_token_id), None)
                    # If an EOS token is found, only the previous
                    # part of it is retained
                    if first_eos_idx is not None:
                        output_token_ids[i] = output_token_ids[
                            i][:first_eos_idx]

                response = self.tokenizer.batch_decode(output_token_ids)
                # print(response)
                if not batched:
                    response = response[0]
                yield ModelStatusCode.STREAM_ING, response, None
                # stop when each sentence is finished,
                # or if we exceed the maximum length
                if (unfinished_sequences.max() == 0
                        or stopping_criteria(input_ids, scores)):
                    break
            yield ModelStatusCode.END, response, None

    def stream_chat(
        self,
        inputs: List[dict],
        do_sample: bool = True,
        **kwargs,
    ):
        """Return the chat completions in stream mode.

        Args:
            inputs (List[dict]): input messages to be completed.
            do_sample (bool): do sampling if enabled
        Returns:
            the text/chat completion
        """
        prompt = self.template_parser(inputs)
        yield from self.stream_generate(prompt, do_sample, **kwargs)

class TransformerBackend(HFTransformer):

    def _load_model(self, path: str, model_kwargs: dict):
        import torch
        from transformers import AutoModelForCausalLM
        model_kwargs.setdefault('torch_dtype', torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, **model_kwargs)
        self.model.eval()


class ModelClient:
    _backends = {
        'lmdeploy_server':LMDeployServerBackend,
        'lmdeploy_client':LMDeployClientBackend,
        'huggingface':TransformerBackend,
        'api':ApiBackend,
        'vllm':vLLMBackend
            }

    _instances: dict = {}

    client: Any

    def __new__(cls, backend=None, **kwargs):
        if backend is not None and backend not in cls._backends:
            raise ValueError(
                f'Backend {backend} is not supported. Currently supported ones'
                f' are {list(cls._backends.keys())}')

        arg_key = f'{backend}'
        for key, value in kwargs.items():
            arg_key += f':{key}:{value}'

        # do not create a new instance if exists
        if arg_key in cls._instances:
            _instance = cls._instances[arg_key]
        else:
            # create a new object and put it to _instance
            _instance = super().__new__(cls)
            if backend is not None:
                _instance.client = cls._backends[backend](**kwargs)
            else:
                raise ValueError(
                f'Backend cannot be None. Currently supported ones'
                f' are {list(cls._backends.keys())}')

            cls._instances[arg_key] = _instance

        return _instance
