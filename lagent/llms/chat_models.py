from typing import Dict, List, Optional, Tuple, Union

from lagent.llms.backends import LLMMixin, dispatch
from jinja2 import Template

def render_chat(template, messages, bos_token="<s>", add_generation_prompt=False):
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
    def __init__(self, backend: Union[str, object]='lmdeploy',
                path:str='', 
                model_name: Optional[str] = None,
                url: Optional[str] = None,
                api_keys: Optional[Union[List[str], str]] = None,
                chat_template_config: Optional[dict] = dict(system='<|im_start|>system\n',
                                                            user='<|im_start|>user\n',
                                                            assistant='<|im_start|>assistant\n',
                                                            environment='<|im_start|>environment\n',
                                                            plugin='<|plugin|>',
                                                            interpreter='<|interpreter|>',
                                                            eosys='<|im_end|>\n',
                                                            eoh='<|im_end|>\n',
                                                            eoa='<|im_end|>',
                                                            eoenv='<|im_end|>\n',
                                                            separator='\n',
                                                            stop_words=['<|im_end|>', '<|action_end|>'],),
                chat_template:str=INTERNLM2_TEMPLATE,
                engine_config: Optional[dict] = None,
                **kwargs):
        self.model = dispatch(backend=backend, path=path, model_name=model_name, chat_template=chat_template, engine_config=engine_config, chat_template_config=chat_template_config, **kwargs)
    
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
        print('kw0:',kwargs)
        return self.model.chat_completion(inputs, **kwargs)

class Qwen2Chat(LLMMixin):
    def __init__(self, backend: Union[str, object]='lmdeploy', chat_template:str=QWEN_TEMPLATE, **kwargs):
        self.model = dispatch(backend, chat_template, **kwargs)


if __name__==  '__main__':
    root_path = '/mnt/datawow/lyq/model/' 
    model_name = 'internlm2_5-7b-chat'
    path = root_path + model_name
    engine_config = dict(tp=1,session_len=1024)
    for backend in ['lmdeploy_server','lmdeploy_clinet','transformer','api']:
        chat_obj = Interlm2Chat(backend=backend,path=path,model_name=model_name,
                                engine_config = engine_config,
                                )
        str_prompt = '介绍一下你自己'
        dict_prompt = [
    {"role": "system", "content": "你是一个有用的AI助手，专门解答与Python编程相关的问题。"},
    {"role": "user", "content": "介绍一下interlm的历史"},
]
        print(f'============completion test===============')
        output = chat_obj.completion(str_prompt)
        print(output)

        print(f'============chat_completion test===============')
        generation_config = dict(top_p=0.8,
                            top_k=1,
                            temperature=0,
                            repetition_penalty=1.02,
                            max_token=50,
                    )
        output_gen = chat_obj.chat_completion(dict_prompt, **generation_config)
        for item in output_gen:
            print(item)
        break
