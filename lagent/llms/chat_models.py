from typing import Dict, List, Optional, Tuple, Union

from lagent.llms.backends import LLMMixin, dispatch



INTERNLM2_TEMPLATE="{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
QWEN_TEMPLATE='''{% if messages[0]['role'] == 'system' %}
    {% set offset = 1 %}
{% else %}
    {% set offset = 0 %}
{% endif %}

{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{% endfor %}

{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}'''

class Interlm2Chat(LLMMixin):
    def __init__(self, backend: Union[str, object]='lmdeploy', path:str='', model_name: Optional[str] = None, chat_template:str=INTERNLM2_TEMPLATE, **kwargs):
        self.model = dispatch(backend, path, model_name, chat_template, **kwargs)
    
    def completion(self,
                 inputs: Union[str, List[str]],
                 session_id: int = 2967,
                 sequence_start: bool = True,
                 sequence_end: bool = True,
                 ignore_eos: bool = False,
                 skip_special_tokens: Optional[bool] = False,
                 timeout: int = 30,
                 **kwargs):
        return self.model.completion(inputs)

    def chat_completion(self,
                    inputs: List[dict],
                    session_id=0,
                    sequence_start: bool = True,
                    sequence_end: bool = True,
                    stream: bool = True,
                    ignore_eos: bool = False,
                    skip_special_tokens: Optional[bool] = False,
                    timeout: int = 30,
                    **kwargs):
        return self.model.chat_completion(inputs)

class Qwen2Chat(LLMMixin):
    def __init__(self, backend: Union[str, object]='lmdeploy', chat_template:str=QWEN_TEMPLATE, **kwargs):
        self.model = dispatch(backend, chat_template, **kwargs)


if __name__==  '__main__':
    root_path = '/mnt/datawow/lyq/model/' 
    model_name = 'internlm2-chat-1_8b'
    path = root_path + model_name
    print('====path:', path)
    for backend in ['lmdeploy_server','lmdeploy_clinet','transformer','api']:
        chat_obj = Interlm2Chat(backend=backend,path=path,model_name=model_name,
                                top_p=0.8,
                                top_k=1,
                                temperature=0,
                                repetition_penalty=1.02,
                                stop_words=['<|im_end|>']
                                # ,chat_template=INTERNLM2_TEMPLATE
                                )
        str_prompt = '介绍一下你自己'
        dict_prompt = [
    {"role": "system", "content": "你是一个有用的AI助手，专门解答与Python编程相关的问题。"},
    {"role": "user", "content": "介绍一下你自己"},
]
        print(f'============completion test===============')
        output = chat_obj.completion(str_prompt)
        print(output)

        print(f'============chat_completion test===============')
        output_gen = chat_obj.chat_completion(dict_prompt)
        for item in output_gen:
            print(item)
        break
