from typing import Dict, List, Optional, Tuple, Union

from lagent.llms.backends import LLMMixin, ModelClient


INTERNLM2_TEMPLATE="{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

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
                chat_template:str = None,
                engine_config: Optional[dict] = None,
                **kwargs):
        self.modelClient = ModelClient(backend=backend, path=path, model_name=model_name, url=url, api_keys=api_keys,
                                       chat_template=chat_template, engine_config=engine_config, chat_template_config=chat_template_config, **kwargs)
        self.model = self.modelClient.client

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
        if self.model.chat_template: # use jinja2 template+completion api
            return self.model.chat_completion_custom(inputs, **kwargs)
        else: # use chat_completion api
            return self.model.chat_completion(inputs, **kwargs)

class Qwen2Chat(LLMMixin):
    def __init__(self, backend: Union[str, object]='lmdeploy', chat_template:str=QWEN_TEMPLATE, **kwargs):
        self.model = dispatch(backend, chat_template, **kwargs)


if __name__==  '__main__':
    root_path = '/mnt/datawow/lyq/model/' 
    root_path = '/Users/qiyusama/Documents/work/model/' 

    # model_name = 'internlm2_5-7b-chat'
    model_name = 'internlm2-chat-1_8b'

    path = root_path + model_name
    engine_config = dict(tp=1
                        #  ,session_len=1024
                         )
    backend = 'huggingface'
    custom = False
    generation_config_init = dict(top_p=0.8,
                            top_k=1,
                            temperature=0,
                            max_new_tokens=8192,
                            repetition_penalty=1.02,
                            stop_words=['<|im_end|>']
            )
    if backend == 'lmdeploy_server':
        if custom:
            chat_obj = Interlm2Chat(backend=backend,path=path,model_name=model_name,
                                    engine_config = engine_config,chat_template=INTERNLM2_TEMPLATE
                                    )
        else:
            chat_obj = Interlm2Chat(backend=backend,path=path,model_name=model_name,
                    engine_config = engine_config
                    )
    silicon_key = 'xxxxxxxxxxxxxx'            
    if backend == 'api':
        chat_obj = Interlm2Chat(backend='api',model_type='internlm/internlm2_5-7b-chat',
                                url='https://api.siliconflow.cn/v1/chat/completions',
                                api_keys=silicon_key,
                                meta_template=[
                                dict(role='system', api_role='system'),
                                dict(role='user', api_role='user'),
                                dict(role='assistant', api_role='assistant'),
                                dict(role='environment', api_role='system')
                                ],
                                engine_config = engine_config,
                                generation_config_init = generation_config_init
            )  

    if backend == 'huggingface':
        chat_obj = Interlm2Chat(backend='huggingface',path=path,
                                engine_config = engine_config,
                                chat_template=INTERNLM2_TEMPLATE,
                                generation_config_init = generation_config_init
        )
    str_prompt = '介绍一下你自己'
    dict_prompt = [
{"role": "system", "content": "你是一个有用的AI助手，专门解答与Python编程相关的问题。"},
{"role": "user", "content": "介绍一下python的历史"},
]    
    generation_config = dict(top_p=0.8,
                    top_k=1,
                    temperature=0,
                    repetition_penalty=1.02,
                    # max_tokens=10240,
            )
    print(f'============completion test===============')
    # output = chat_obj.completion(str_prompt, **generation_config)
    # print(output)

    print(f'============chat_completion test===============')
    output_gen = chat_obj.chat_completion(dict_prompt, **generation_config)
    # print('output:',output_gen)
    for item in output_gen:
        output=item
        print('output:',output)

