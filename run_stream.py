import os
import platform
from langchain import PromptTemplate
from transformers import AutoModel,AutoTokenizer
from vector_prompt import config_vs
#from langchain.chains.base import Chain

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'

model_name = "chatglm2-6b"
quantize_bit = 8
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code = True)
model = AutoModel.from_pretrained(model_name,trust_remote_code = True).half().quantize(quantize_bit).cuda().eval()

def get_temp():
    template = "已知信息：{knowledge}. 请问：{query}"
    prompt1 = PromptTemplate(input_variables=["knowledge","query"],template = template)
    return prompt1

class get_prompt():
    def __init__(self,promptT) -> None:
        self.promptT = promptT
    def run(self,query,knowledge):
        if knowledge is not None:
            prompt = self.promptT.format(query = query,knowledge = knowledge)
        else :
            prompt = self.promptT.format(query = query)
       
        return prompt

def main():
    
    promptTem = get_temp()
    prompt_object = get_prompt(promptT=promptTem)
    config_vs1 = config_vs()
    #vec_store = config_vs1.load_vs(save_path= "vs")
        
    print("\n欢迎使用Chatbot-KB问答系统，输入内容即可进行问答对话，输入clear 清空对话历史，输入stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            os.system(clear_command)
            print("\n欢迎使用Chatbot-KB问答系统，输入内容即可进行问答对话，输入clear 清空对话历史，输入stop 终止程序")
            continue
        print("\nChatbot-KB：", end="")
        knowledge = config_vs1.get_knowledge(query)

        #流式输出
        prompt = prompt_object.run(query,knowledge)
        
        current_length = 0
        for response, _ in model.stream_chat(tokenizer, prompt):
            print(response[current_length:], end="", flush=True)
            current_length = len(response)
        print('\n')
        
        
        
        
    

if __name__ == "__main__":
    main()






