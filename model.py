from transformers import AutoModel,AutoTokenizer
from typing import Any,List

class ChatGLM():
    def __init__(self,model_name = "chatglm2-6b",quantize_bit = 4 ) ->None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code = True)
        self.model = AutoModel.from_pretrained(model_name,trust_remote_code = True).half().quantize(quantize_bit).cuda().eval()
    
    def __call__(self,prompt) -> Any:  #一次性输出全部回答内容
        response , _ = self.model.chat(self.tokenizer,prompt)
        return response
    
    



    
