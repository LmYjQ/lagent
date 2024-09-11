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
    def __init__(self, backend: Union[str, object]='lmdeploy', chat_template:str=INTERNLM2_TEMPLATE, **kwargs):
        self.model = dispatch(backend, chat_template, **kwargs)
    
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
    root_path = '/Users/qiyusama/Documents/work/model/'
    root_path = '/mnt/datawow/lyq/model/' 
    model_name = 'internlm2-chat-1_8b'
    path = root_path + model_name
    print('====path:', path)
    for backend in ['lmdeploy_server','lmdeploy_clinet','transformer','api']:
        chat_obj = Interlm2Chat(backend=backend,path=path)
        str_prompt = '介绍下你自己'
        dict_prompt = [
    {"role": "system", "content": "你是一个有用的AI助手，专门解答与Python编程相关的问题。"},
    {"role": "user", "content": "如何在Python中创建一个列表？"},
    {"role": "assistant", "content": "在Python中，你可以通过几种方式创建列表：\n\n1. 使用方括号：\n   my_list = [1, 2, 3, 4, 5]\n\n2. 使用list()函数：\n   my_list = list((1, 2, 3, 4, 5))\n\n3. 使用列表推导式：\n   my_list = [x for x in range(1, 6)]\n\n这些方法都会创建一个包含数字1到5的列表。"},
    {"role": "user", "content": "谢谢，那么如何向列表中添加元素呢？"},
    {"role": "assistant", "content": "很高兴你问这个问题！在Python中，有几种方法可以向列表中添加元素：\n\n1. 使用append()方法添加单个元素到列表末尾：\n   my_list.append(6)\n\n2. 使用extend()方法添加多个元素（通常是另一个列表）到列表末尾：\n   my_list.extend([7, 8, 9])\n\n3. 使用insert()方法在特定位置插入元素：\n   my_list.insert(0, 0)  # 在索引0处插入0\n\n4. 使用+运算符连接两个列表：\n   my_list = my_list + [10, 11, 12]\n\n5. 使用切片赋值在特定位置插入多个元素：\n   my_list[1:1] = [13, 14, 15]\n\n这些方法让你能够灵活地管理列表中的元素。"},
    {"role": "user", "content": "能给我一个完整的例子吗？"}
]
        #chat_obj.completion(str_prompt)
        chat_obj.chat_completion(dict_prompt)
        break
