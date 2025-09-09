"""CosmosDB Vector Database implementation."""
from collections.abc import Awaitable
from typing import Any, Dict, List, Optional

from agno.document import Document
from agno.embedder import Embedder
from agno.utils.log import log_debug, log_error, log_info, log_warning
from agno.utils.string import safe_content_hash
from agno.vectordb.base import VectorDb
from agno.vectordb.distance import Distance
from agno.vectordb.search import SearchType

try:
    from azure.cosmos.aio import (
        CosmosClient as AsyncCosmosClient,
        DatabaseProxy as AsyncDatabaseProxy,
        ContainerProxy as AsyncContainerProxy,
    )
    from azure.cosmos import (
        CosmosClient,
        DatabaseProxy,
        ContainerProxy,
        PartitionKey,
        exceptions,
        IndexingMode,
    )
    from azure.cosmos.exceptions import CosmosHttpResponseError
except ImportError as e:
    raise ImportError("`azure.cosmos` not installed. Please install using `pip install azure-cosmos`") from e


class CosmosDb(VectorDb):
    """
    CosmosDB Vector Database implementation.
    """

    def __init__(
        self,
        connection_string: str,
        database_name: str = "VectorDB",
        embedder: Optional[Embedder] = None,
        distance_metric: str = Distance.cosine,
        retry_total: int = 10,
        retry_connect: int = 3,
        retry_read: int = 3,
        retry_status: int = 3,
        retry_backoff_factor: float = 0.3,
        retry_backoff_max: int = 120,
        retry_fixed_interval: int | None = None,
        search_type: SearchType = SearchType.vector,
    ):
        """
        Initialize the MongoDb with MongoDB container details.

        Args:
            connection_string (str): The connection string for the Cosmos DB account.
            database_name (str): The name of the Cosmos DB database. Defaults to "VectorDB".
            embedder (Embedder): The embedder to use for document embeddings.
            distance_metric (str): The distance metric to use for vector similarity. Defaults to 'cosine'.
            retry_total (int): Total number of retries to allow. Defaults to 10.
            retry_connect (int): Number of retries on connection errors. Defaults to 3.
            retry_read (int): Number of retries on read errors. Defaults to 3.
            retry_status (int): Number of retries on bad status codes. Defaults to 3.
            retry_backoff_factor (float): A backoff factor to apply between attempts after the second
                try (most errors are resolved immediately by a second try without a delay).
                Defaults to 0.3.
            retry_backoff_max (int): The maximum value of backoff time in seconds. Defaults to 120.
            retry_fixed_interval (int | None): If set, the backoff time will be fixed to this value in seconds.
                If None, the backoff time will be calculated using the backoff factor. Defaults to None.
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.embedder = embedder
        self.distance_metric = distance_metric
        self.retry_total = retry_total
        self.retry_connect = retry_connect
        self.retry_read = retry_read
        self.retry_status = retry_status
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_backoff_max = retry_backoff_max
        self.retry_fixed_interval = retry_fixed_interval
        self.retry_kwargs = {
            "retry_total": self.retry_total,
            "retry_connect": self.retry_connect,
            "retry_read": self.retry_read,
            "retry_status": self.retry_status,
            "retry_backoff_factor": self.retry_backoff_factor,
            "retry_backoff_max": self.retry_backoff_max,
            "retry_fixed_interval": self.retry_fixed_interval,
        }
        self.search_type = search_type
        self.partition_key = PartitionKey(path="/id", kind="Hash")

    def _get_client(self) -> CosmosClient:
        """Returns an instance of the Cosmos DB client."""
        return CosmosClient.from_connection_string(self.connection_string, kwargs=self.retry_kwargs)

    def _get_async_client(self) -> AsyncCosmosClient:
        """Returns an instance of the asynchronous Cosmos DB client."""
        return AsyncCosmosClient.from_connection_string(self.connection_string, kwargs=self.retry_kwargs)

    def _create_db(self, client: CosmosClient) -> DatabaseProxy:
        """Creates the Cosmos DB database if it does not exist."""
        return client.create_database_if_not_exists(id=self.database_name)

    def _create_db_async(self, client: AsyncCosmosClient) -> Awaitable[AsyncDatabaseProxy]:
        """Creates the Cosmos DB database if it does not exist."""
        return client.create_database_if_not_exists(id=self.database_name)

    def _get_vector_embedding_policy(self) -> dict[str, Any]:
        """Returns the vector embedding policy for the container."""
        if self.embedder is None:
            raise ValueError("Embedder must be provided to create vector embedding policy.")
        return {
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": "float32",
                    "distanceFunction": {self.distance_metric},
                    "dimensions": {self.embedder.dimensions},
                }
            ]
        }

    def _get_index_policy(self) -> dict[str, Any]:
        return {
            "indexingMode": IndexingMode.Consistent,
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": "/_etag/?"}, {"path": "/embedding/*"}],
            "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
            "fullTextIndexes": [{"path": "/content"}],
        }

    def _get_full_text_policy(self) -> dict[str, Any]:
        return {"defaultLanguage": "en-US", "fullTextPaths": [{"path": "/content", "language": "en-US"}]}

    def _create_container(self, database: DatabaseProxy) -> ContainerProxy:
        """Creates the Cosmos DB container if it does not exist."""
        return database.create_container_if_not_exists(
            id="Documents",
            indexing_policy=self._get_index_policy(),
            vector_embedding_policy=self._get_vector_embedding_policy(),
            full_text_policy=self._get_full_text_policy(),
            partition_key=self.partition_key,
            offer_throughput=400,
        )

    def _create_container_async(self, database: AsyncDatabaseProxy) -> Awaitable[AsyncContainerProxy]:
        """Creates the Cosmos DB container if it does not exist."""
        return database.create_container_if_not_exists(
            id="Documents",
            indexing_policy=self._get_index_policy(),
            vector_embedding_policy=self._get_vector_embedding_policy(),
            full_text_policy=self._get_full_text_policy(),
            partition_key=self.partition_key,
            offer_throughput=400,
        )

    def _clean_content(self, content: str) -> str:
        """
        Clean the content by replacing null characters.

        Args:
            content (str): The content to clean.

        Returns:
            str: The cleaned content.
        """
        return content.replace("\x00", "\ufffd")

    def _prepare_doc(self, document: Document, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare a Document for insertion or upsertion into CosmosDB."""
        document.embed(self.embedder)
        if document.embedding is None:
            raise ValueError(f"Failed to generate embedding for document: {document.id}")

        cleaned_content = self._clean_content(document.content)
        content_hash = safe_content_hash(cleaned_content)
        # Add filters to document
        if filters:
            meta_data = document.meta_data.copy() if document.meta_data else {}
            meta_data.update(filters)
            document.meta_data = meta_data

        record = {
            "id": content_hash,
            "name": document.name,
            "meta_data": document.meta_data,
            "filters": filters,
            "content": cleaned_content,
            "embedding": document.embedding,
            "usage": document.usage,
            "content_hash": content_hash,
        }
        log_debug(f"Prepared document for Cosmos DB: {record}")
        return record

    def create(self) -> None:
        """Creates the Cosmos DB database and container if they do not exist."""
        client = self._get_client()
        try:
            database = self._create_db(client)
            self._create_container(database)
            log_info("Cosmos DB database and container created or already exist.")
            return
        except exceptions.CosmosClientTimeoutError:
            log_info("Cosmos DB request timed out.")
        except Exception as e:
            log_warning(f"An error occurred while creating Cosmos DB database or container: {e}")

    async def async_create(self) -> None:
        client = self._get_async_client()
        try:
            database = self._create_db_async(client)
            await self._create_container_async(await database)
            log_info("Cosmos DB database and container created or already exist.")
            return
        except exceptions.CosmosClientTimeoutError:
            log_info("Cosmos DB request timed out.")
        except Exception as e:
            log_warning(f"An error occurred while creating Cosmos DB database or container: {e}")

    def doc_exists(self, document: Document) -> bool:
        client = self._get_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        try:
            content_hash = safe_content_hash(document.content)
            exists = container.read_item(item=content_hash, partition_key=content_hash) is not None
            log_debug(f"Document with content hash '{content_hash}' {'exists' if exists else 'does not exist'}")
            return exists
        except exceptions.CosmosResourceNotFoundError:
            log_debug(f"Document with content hash '{content_hash}' does not exist")
            return False
        except Exception as e:
            log_warning(f"Error checking document existence: {e}")
            return False

    async def async_doc_exists(self, document: Document) -> bool:
        client = self._get_async_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        try:
            content_hash = safe_content_hash(document.content)
            exists = await container.read_item(item=content_hash, partition_key=content_hash) is not None
            log_debug(f"Document with content hash '{content_hash}' {'exists' if exists else 'does not exist'}")
            return exists
        except exceptions.CosmosResourceNotFoundError:
            log_debug(f"Document with content hash '{content_hash}' does not exist")
            return False
        except Exception as e:
            log_warning(f"Error checking document existence: {e}")
            return False

    def name_exists(self, name: str) -> bool:
        client = self._get_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        try:
            # Assuming Document has a 'name' attribute
            query = f"SELECT * FROM Documents WHERE name = '{name}'"
            items = list(container.query_items(query=query, enable_cross_partition_query=True))
            exists = len(items) > 0
            log_debug(f"Document with name '{name}' {'exists' if exists else 'does not exist'}")
            return exists
        except Exception as e:
            log_warning(f"Error checking document existence: {e}")
            return False

    def async_name_exists(self, name: str) -> bool:
        client = self._get_async_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        try:
            # Assuming Document has a 'name' attribute
            query = f"SELECT * FROM Documents WHERE name = '{name}'"
            items = container.query_items(query=query, enable_cross_partition_query=True)
            exists = len(await items) > 0
            log_debug(f"Document with name '{name}' {'exists' if exists else 'does not exist'}")
            return exists
        except Exception as e:
            log_warning(f"Error checking document existence: {e}")
            return False

    def id_exists(self, id: str) -> bool:
        client = self._get_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        try:
            exists = container.read_item(item=str(id), partition_key=str(id)) is not None
            log_debug(f"Document with ID '{id}' {'exists' if exists else 'does not exist'}")
            return exists
        except exceptions.CosmosResourceNotFoundError:
            return False
        except Exception as e:
            log_warning(f"Error checking document existence: {e}")
            return False

    def insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        client = self._get_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        try:
            for doc in documents:
                prepared_doc = self._prepare_doc(doc, filters)
                container.create_item(prepared_doc)
        except Exception as e:
            log_warning(f"Error inserting documents: {e}")

    async def async_insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        client = self._get_async_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        try:
            for doc in documents:
                prepared_doc = self._prepare_doc(doc, filters)
                container.create_item(prepared_doc)
        except Exception as e:
            log_warning(f"Error inserting documents: {e}")

    def upsert_available(self) -> bool:
        return True

    def upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        client = self._get_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        try:
            for doc in documents:
                prepared_doc = self._prepare_doc(doc, filters)
                container.upsert_item(prepared_doc)
        except Exception as e:
            log_warning(f"Error upserting documents: {e}")

    async def async_upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        client = self._get_async_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        try:
            for doc in documents:
                prepared_doc = self._prepare_doc(doc, filters)
                container.upsert_item(prepared_doc)
        except Exception as e:
            log_warning(f"Error upserting documents: {e}")

    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for documents using vector similarity in the CosmosDB."""
        if self.search_type == SearchType.vector:
            return self.vector_search(query=query, limit=limit, filters=filters)
        elif self.search_type == SearchType.keyword:
            return self.keyword_search(query=query, limit=limit, filters=filters)
        elif self.search_type == SearchType.hybrid:
            return self.hybrid_search(query=query, limit=limit, filters=filters)
        else:
            log_error(f"Invalid search type '{self.search_type}'.")
            return []

    async def async_search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search asynchronously by running in a thread."""
        if self.search_type == SearchType.vector:
            return await self.async_vector_search(query=query, limit=limit, filters=filters)
        elif self.search_type == SearchType.keyword:
            return await self.async_keyword_search(query=query, limit=limit, filters=filters)
        elif self.search_type == SearchType.hybrid:
            return await self.async_hybrid_search(query=query, limit=limit, filters=filters)
        else:
            log_error(f"Invalid search type '{self.search_type}'.")
            return []

    def _build_vector_query(self, query, limit):
        if self.embedder is None:
            raise ValueError("Embedder must be provided for vector search.")
        embedded_query = self.embedder.get_embedding(query)
        if embedded_query is None:
            raise ValueError("Failed to create vector embedding.")
        vector_query = (
            f"SELECT TOP {limit} FROM Documents ORDER BY VECTOR_DISTANCE(Documents.embedding, {embedded_query}) ASC"
        )
        return vector_query

    def _map_item_to_document(self, item) -> Document:
        # Map the item from the database to the Document model
        return Document(id=item["id"], name=item["name"], content=item["content"], meta_data=item.get("metadata", {}))

    def vector_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform a vector similarity search.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.

        Returns:
            List[Document]: List of matching documents.
        """
        client = self._get_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        # Perform vector search using the container
        results = container.query_items(
            query=self._build_vector_query(query, limit), enable_cross_partition_optimization=True
        )
        return [self._map_item_to_document(item) for item in results]

    async def async_vector_search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform a vector similarity search.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.

        Returns:
            List[Document]: List of matching documents.
        """
        client = self._get_async_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        # Perform vector search using the container
        results = container.query_items(
            query=self._build_vector_query(query, limit), enable_cross_partition_optimization=True
        )
        items = []
        async for item in results:
            items.append(item)
        return [self._map_item_to_document(item) for item in items]

    def _build_keyword_query(self, query, limit):
        split_keywords = [f"'{kw}'" for kw in query.split(" ") if kw]
        return f"SELECT TOP {limit} FROM Documents ORDER BY RANK FullTextScore(Document.content, {','.join(split_keywords)})"

    def keyword_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform a keyword search on the 'content' column.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.

        Returns:
            List[Document]: List of matching documents.
        """
        client = self._get_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        # Perform keyword search using the container
        results = container.query_items(
            query=self._build_keyword_query(query, limit), enable_cross_partition_optimization=True
        )
        return [self._map_item_to_document(item) for item in results]

    async def async_keyword_search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform a keyword search on the 'content' column asynchronously.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.

        Returns:
            List[Document]: List of matching documents.
        """
        client = self._get_async_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        # Perform keyword search using the container
        results = container.query_items(
            query=self._build_keyword_query(query, limit), enable_cross_partition_optimization=True
        )
        items = []
        async for item in results:
            items.append(item)
        return [self._map_item_to_document(item) for item in items]

    def _build_hybrid_query(self, query, limit):
        if self.embedder is None:
            raise ValueError("Embedder must be provided for vector search.")
        embedded_query = self.embedder.get_embedding(query)
        if embedded_query is None:
            raise ValueError("Failed to create vector embedding.")
        split_keywords = [f"'{kw}'" for kw in query.split(" ") if kw]
        hybrid_query = f"SELECT TOP {limit} FROM Documents ORDER BY RANK RRF(\
                            VECTOR_DISTANCE(Documents.embedding, {embedded_query})\
                            FullTextScore(Document.content, {','.join(split_keywords)})\
                        )"
        return hybrid_query

    def hybrid_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform a hybrid search combining vector similarity and full-text search.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.

        Returns:
            List[Document]: List of matching documents.
        """
        client = self._get_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        # Perform keyword search using the container
        results = container.query_items(
            query=self._build_hybrid_query(query, limit), enable_cross_partition_optimization=True
        )
        return [self._map_item_to_document(item) for item in results]

    async def async_hybrid_search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform a hybrid search combining vector similarity and full-text search asynchronously.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.

        Returns:
            List[Document]: List of matching documents.
        """
        client = self._get_async_client()
        database = client.get_database_client(self.database_name)
        container = database.get_container_client("Documents")
        # Perform keyword search using the container
        results = container.query_items(
            query=self._build_hybrid_query(query, limit), enable_cross_partition_optimization=True
        )
        items = []
        async for item in results:
            items.append(item)
        return [self._map_item_to_document(item) for item in items]

    def drop(self) -> None:
        """Drop the table from the Database."""
        try:
            client = self._get_client()
            database = client.get_database_client(self.database_name)
            container = database.get_container_client("Documents")
            database.delete_container(container)
        except CosmosHttpResponseError as e:
            print(f"Error dropping container: {e}")

    async def async_drop(self) -> None:
        try:
            client = self._get_async_client()
            database = client.get_database_client(self.database_name)
            container = database.get_container_client("Documents")
            await database.delete_container(container)
        except CosmosHttpResponseError as e:
            print(f"Error dropping container: {e}")

    def exists(self) -> bool:
        """
        Check if the table exists in the database.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        raise NotImplementedError

    async def async_exists(self) -> bool:
        try:
            client = self._get_async_client()
            database = client.get_database_client(self.database_name)
            container = database.get_container_client("Documents")
            await container.read()
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False
        except Exception as e:
            log_warning(f"Error checking container existence: {e}")
            return False

    def optimize(self) -> None:
        """TODO: Optimize the table for better performance."""
        raise NotImplementedError

    def delete(self) -> bool:
        """
        Delete all documents from the collection.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        try:
            client = self._get_client()
            database = client.get_database_client(self.database_name)
            container = database.get_container_client("Documents")
            # Iterating all the items of a container
            for item in container.query_items(query="SELECT * FROM c", enable_cross_partition_query=True):
                # Deleting the current item
                container.delete_item(item, partition_key=item[self.partition_key.path])
                container.delete_all_items_by_partition_key(item["id"])
            return True
        except Exception as e:
            log_warning(f"Error deleting documents: {e}")
            return False
