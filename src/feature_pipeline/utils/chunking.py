from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from config import settings

##  Trường hợp thông thường muốn cắt
# def chunk_text(text: str) -> list[str]:
#     character_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n"], chunk_size=500, chunk_overlap=0
#     )
#     text_split = character_splitter.split_text(text)

#     token_splitter = SentenceTransformersTokenTextSplitter(
#         chunk_overlap=50,
#         tokens_per_chunk=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
#         model_name=settings.EMBEDDING_MODEL_ID,
#     )
#     chunks = []

#     for section in text_split:
#         chunks.extend(token_splitter.split_text(section))

#     return chunks


# Trong bài toán hiện tại, mỗi đoạn text là một đoạn có ý nghĩa rồi, tôi không muốn tách thêm
def chunk_text(text: str) -> list[str]:
    return [text]