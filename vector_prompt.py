from typing import Any, List
from langchain import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

from langchain.vectorstores import FAISS


vs_save_path = "vs"
class TextSpliter(CharacterTextSplitter):
    def __init__(self, separator: str = "\n\n", is_separator_regex: bool = False, **kwargs: Any) -> None:
        super().__init__(separator, is_separator_regex, **kwargs)
    def split_text(self, text: str) -> List[str]:
        texts = text.split("\n")
        texts = [Document(page_content=text, metadata={"from": "filename or book.txt"}) for text in texts]
        return texts

class config_vs():
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name="text2vec-large-chinese",
                                    model_kwargs={'device': "cuda"})
    
    def store_vs(self,texts,vs_save_path):
        text_spliter = TextSpliter()
        texts = text_spliter.split_text(texts)
        texts1 = [text.page_content for text in texts]

        docs = self.embeddings.embed_documents(texts1)
        vector_store = FAISS.from_documents(texts,self.embeddings)
        vector_store.save_local(vs_save_path)
    
    def load_vs(self,save_path=vs_save_path):
        vector_store2 = FAISS.load_local(save_path,self.embeddings)
        return vector_store2
    
    def get_knowledge(self,query,k = 2):
        
        vector_store = self.load_vs()
        related_docs_with_score = vector_store.similarity_search_with_score(query = query,k = k)

        knowledge = ""
        for docs in related_docs_with_score:
            dos,score = docs
            knowledge = knowledge + dos.page_content
        
        return knowledge
    
        
if __name__ == "__main__":
    text_path = "text.txt"
    vs_save_path = "vs"

    with open(text_path,"r") as f:
        text = f.read()
    
    config = config_vs()
    config.store_vs(text,vs_save_path)


    








