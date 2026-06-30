from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder

from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
        # 初始化 Rerank 模型
        self.rerank_enabled = chroma_conf.get("rerank_enabled", False)
        if self.rerank_enabled:
            model_name = chroma_conf.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.rerank_model = CrossEncoder(model_name,
                                             local_files_only=True)
            self.rerank_initial_k = chroma_conf.get("rerank_initial_k", 20)
            self.rerank_final_k = chroma_conf.get("rerank_final_k", 5)
        else:
            self.rerank_model = None

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def get_rerank_retriever(self):
        """返回带有 Rerank 功能的检索器"""
        if not self.rerank_enabled:
            logger.warning("Rerank 未启用，返回原始检索器")
            return self.get_retriever()

        # 创建自定义检索器
        class RerankRetriever(BaseRetriever):
            vector_store: Any
            rerank_model: Any
            initial_k: int
            final_k: int

            def _get_relevant_documents(self, query, *, run_manager=None):
                # 1. 向量检索获取更多候选
                retriever = self.vector_store.as_retriever(search_kwargs={"k": self.initial_k})
                docs = retriever.invoke(query)

                if not docs:
                    return []

                # 2. 构建 (query, doc) 对用于 Rerank
                pairs = [(query, doc.page_content) for doc in docs]

                # 3. 计算相关性分数
                scores = self.rerank_model.predict(pairs)

                # 4. 按分数降序排序，取 top-k
                scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                top_docs = [doc for doc, _ in scored_docs[:self.final_k]]

                return top_docs

        return RerankRetriever(
            vector_store=self.vector_store,
            rerank_model=self.rerank_model,
            initial_k=self.rerank_initial_k,
            final_k=self.rerank_final_k
        )

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False  # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True  # md5 处理过

                return False  # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
