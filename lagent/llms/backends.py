from typing import Dict, List, Optional, Tuple, Union

from lagent.llms.base_llm import BaseLLM
from lagent.schema import ModelStatusCode
from lagent.utils.util import filter_suffix

class LLMMixin:
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
            # tp=int(kwargs.get('tp', 1)),
            # meta_template=chat_template,
            **kwargs
        )
    elif backend=='lmdeploy_client':
        return LMDeployClientBackend(
            model_name=kwargs.get('model_name'),
            url=kwargs.get('url', '0.0.0.0:23333'),
                **kwargs
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
    
class LMDeployServerBackend(BaseLLM, LLMMixin):
    def __init__(self, path: str,
                 model_name: Optional[str] = None,
                 server_name: str = '0.0.0.0',
                 server_port: int = 23333,
                 log_level: str = 'WARNING',
                 serve_cfg=dict(),
                 **kwargs):
        super().__init__(path=path, **kwargs)
        self.model_name = model_name
        # TODO get_logger issue in multi processing
        import lmdeploy
        self.client = lmdeploy.serve(
            model_path=self.path,
            model_name=model_name,
            server_name=server_name,
            server_port=server_port,
            log_level=log_level,
            **serve_cfg)
        
    def chat_completion(self,
                    inputs: List[dict],
                    session_id=2679,
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
        resp = filter_suffix(resp, self.gen_params.get('stop_words'))
        if not batched:
            return resp[0]
        return resp

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



class ApiBackend(BaseLLM, LLMMixin):
    def __init__(self, **kwargs):
        pass
        
    def chat_completion():
        pass
    def completion():
        pass

class vLLMBackend(BaseLLM, LLMMixin):
    def __init__(self, **kwargs):
        pass
        
    def chat_completion():
        pass
    def completion():
        pass
