import time
from tqdm import tqdm
from abc import abstractmethod

import torch
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
import faiss



class BaseIndex:
    """ 
    functions
        add(texts) -> None: 
        query_indexs(query, top_k=5) -> list[int]: 
        get_texts(idxs) -> list[str]: 
        query(query, top_k=5) -> list[str]: query_indexs + get_texts
    properties
        num_indexed_items:
    """
    texts = []
    index = None
    def __init__(self) -> None:
        pass

    def add(self, texts):
        self.texts.extend(texts)

    @property
    def num_indexed_items(self):
        return len(self.texts)

    @abstractmethod
    def query_indexs(self, query, top_k=5) -> list[int]:
        raise NotImplementedError

    def get_texts(self, idxs):
        return [self.texts[i] for i in idxs]

    def query(self, query, top_k=5):
        matched_idxs = self.query_indexs(query, top_k)
        return self.get_texts(matched_idxs)

class DummyIndex(BaseIndex):
    def query_indexs(self, query, top_k=5):
        return list(range(top_k))

class SimCSEIndex(BaseIndex):
    """ SimCSE+Faiss 
    model for encode: SimCSE("princeton-nlp/sup-simcse-roberta-large"); 
    model for index: faiss.IndexFlatIP
    """
    def __init__(self, init_texts=None) -> None:
        from simcse import SimCSE
        self.encoder = SimCSE("princeton-nlp/sup-simcse-roberta-large")
        if init_texts is not None:
            self.add(init_texts)
        self.index = faiss.IndexFlatL2(self.encoder.get_embedding_dim())        # DEBUG: any API?

    def add(self, texts):
        super().add(texts)
        embeddings = self.encoder.encode(texts, batch_size=128, normalize_to_unit=True, return_numpy=True)
        self.index.add(embeddings)

    def query_indexs(self, query, top_k=5):
        query_embedding = self.encoder.encode([query], batch_size=128, normalize_to_unit=True, return_numpy=True)
        D, I = self.index.search(query_embedding, top_k)
        return I[0]

class BGEIndex(BaseIndex):
    """ BGE  https://huggingface.co/BAAI/bge-large-zh-v1.5 
    model for encode: BGE("BAAI/bge-reranker-large"); 
    model for index: faiss.IndexFlatIP
    """
    def __init__(self, init_texts=None) -> None:
        from transformers import AutoModel, AutoTokenizer
        model_id = "BAAI/bge-large-zh-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.model.to(device)  # to cuda

        self.index = faiss.IndexFlatIP(self.model.config.hidden_size)

    def add(self, texts):
        super().add(texts)
        # t_start = time.time()
        embeddings = self.get_embedding_batch(texts, batch_size=128)
        # print(f"Encoding time: {time.time()-t_start:.2f}s")
        self.index.add(embeddings.numpy())  # -> numpy

    def query_indexs(self, query, top_k=5):
        inputs = self.tokenizer([query], padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        query_embedding = self.model(**inputs, return_dict=True)[0][:, 0].cpu().detach()
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).numpy()
        D, I = self.index.search(query_embedding, top_k)
        return I[0]

    def get_embedding_batch(self, texts, batch_size=128):
        inputs_batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        embeddings = []
        # for i in tqdm(range(0, len(inputs_batch), batch_size)):
        for i in range(0, len(inputs_batch), batch_size):
            inputs = {k: v[i:i+batch_size] for k, v in inputs_batch.items()}
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                model_output = self.model(**inputs, return_dict=True)
                sentence_embeddings = model_output[0][:, 0]
                embeddings.append(sentence_embeddings)
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu()
        return embeddings

MODE2INDEX = {
    "dummy": DummyIndex,
    "simcse": SimCSEIndex,
    "bge": BGEIndex
}
