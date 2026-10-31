"""Construct the same configured agent for console, web and batch entrypoints."""

from .config import ConfigurationError
from .conversation import ConversationAgent
from .tools.operations import SupportOperations


def create_agent(settings, *, dataset_dir=None, live=False, tool_dir=None):
    root = dataset_dir or settings.dataset_dir
    if settings.runtime_dir.is_relative_to(root):
        raise ConfigurationError("RUNTIME_DIR must be outside the dataset directory")
    rag = None
    if live:
        from .llm import ChatCompletionClient
        from .rag import RagAgent
        from .retrieval.provider import create_embedder,index_directory
        from .retrieval.index import KnowledgeIndex
        client = ChatCompletionClient(settings)
        embedder = create_embedder(settings)
        index = KnowledgeIndex.load(index_directory(settings), embedder, root)
        from .dataset import PublicKnowledge
        rag = RagAgent(client, index, knowledge=PublicKnowledge.load(root))
    return ConversationAgent(SupportOperations(root, tool_dir or settings.runtime_dir / "tools"), rag=rag)
