import os
import platform
from langchain import PromptTemplate
from model import ChatGLM
from vector_prompt import config_vs
#from langchain.chains.base import Chain

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'

def get_temp():
    template = "已知信息：{knowledge}. 请问：{query}"
    prompt1 = PromptTemplate(input_variables=["knowledge","query"],template = template)
    return prompt1

class llm_chain():
    def __init__(self,llm,promptT) -> None:
        self.llm = llm
        self.promptT = promptT
    def run(self,query,knowledge):
        if knowledge is not None:
            prompt = self.promptT.format(query = query,knowledge = knowledge)
        else :
            prompt = self.promptT.format(query = query)
        response = self.llm(prompt)  #一次性输出
        return response

def main():
    
    promptTem = get_temp()
    config_vs1 = config_vs()
    #vec_store = config_vs1.load_vs(save_path= "vs")
    llm = ChatGLM(quantize_bit=8)
    chat = llm_chain(llm=llm,promptT=promptTem)    
    print("欢迎使用Chatbot-KB问答系统，输入内容即可进行问答对话，输入clear 清空对话历史，输入stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            os.system(clear_command)
            print("欢迎使用Chatbot-KB问答系统，输入内容即可进行问答对话，输入clear 清空对话历史，输入stop 终止程序")
            continue
        print("\nChatbot-KB：", end="")
        knowledge = config_vs1.get_knowledge(query)
        response = chat.run(query,knowledge)
        
        print(response)

    

if __name__ == "__main__":
    main()






