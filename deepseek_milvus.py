mport os
import logging
import asyncio
import numpy as np
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage, BaseVectorStorage
from nano_graphrag._utils import compute_args_hash
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from nano_graphrag._utils import logger
from dataclasses import dataclass

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

DEEPSEEK_API_KEY = "sk-8ca48355ffc241e7b25061aa07e8f653"
MODEL = "deepseek-chat"
WORKING_DIR = "./nano_graphrag_cache_deepseek_TEST"

EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cpu"
)

# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

@dataclass
class MilvusLiteStorge(BaseVectorStorage):

    @staticmethod
    def create_collection_if_not_exist(client, collection_name: str, **kwargs):
        if client.has_collection(collection_name):
            return
        # TODO add constants for ID max length to 32
        client.create_collection(
            collection_name, max_length=32, id_type="string", **kwargs
        )

    def __post_init__(self):
        from pymilvus import MilvusClient

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], "milvus_lite.db"
        )
        self._client = MilvusClient(self._client_file_name)
        self._max_batch_size = self.global_config["embedding_batch_num"]
        MilvusLiteStorge.create_collection_if_not_exist(
            self._client,
            self.namespace,
            dimension=self.embedding_func.embedding_dim,
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["vector"] = embeddings[i]
        results = self._client.upsert(collection_name=self.namespace, data=list_data)
        return results

    async def query(self, query, top_k=5):
        embedding = await self.embedding_func([query])
        results = self._client.search(
            collection_name=self.namespace,
            data=embedding,
            limit=top_k,
            output_fields=list(self.meta_fields),
            search_params={"metric_type": "COSINE", "params": {"radius": 0.2}},
        )
        return [
            {**dp["entity"], "id": dp["id"], "distance": dp["distance"]}
            for dp in results[0]
        ]


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content

rag: GraphRAG = None

def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

def query():
    global rag
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="local")
        )
    )


def insert():
    from time import time
    global rag

    with open("./tests/mock_data_2.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        embedding_func=local_embedding,
        vector_db_storage_cls=MilvusLiteStorge,
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)
    # rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    # rag.insert(FAKE_TEXT[half_len:])


if __name__ == "__main__":
    insert()
    query()

