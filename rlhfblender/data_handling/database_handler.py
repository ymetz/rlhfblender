from typing import Dict, List, Optional, Type, TypeVar

from databases import Database
from pydantic import BaseModel

"""
    database_handler.py - Wrapper with static methods which wrap database access operations. For each method, the db
    connection has to be supplied

    Commonly used from Trainer & DataHandler/Server instances to make sure schemas are in sync
"""

T = TypeVar("T", BaseModel, str)


def dict_factory(crs, row) -> Dict:
    d = {}
    for idx, col in enumerate(crs.description):
        d[col[0]] = row[idx]
    return d


async def create_table_from_model(cursor: Database, model: Type[BaseModel], table_name: Optional[str] = None) -> None:
    """
    Creates a new project table in the database dynamically based on the Project DataModel
    :param cursor: sqlite3.Cursor
    :param model: pydantics.BaseModel
    :param table_name:  Optional[str] - If not specified, the model name is used
    :return: None
    """
    table_name = model.__name__ if table_name is None else table_name
    query = "CREATE TABLE IF NOT EXISTS " + table_name + " ("
    for field in model.__annotations__:
        if field == "id":
            query += "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        else:
            # If int type is not specified, default to TEXT
            if model.__annotations__[field] == int:
                query += field + " INTEGER,"
            else:
                query += field + " TEXT,"
    query = query[:-1] + ")"
    return await cursor.execute(query)


async def get_columns_names(cursor: Database, model: Type[BaseModel], table_name: Optional[str] = None) -> List[str]:
    """
    Returns all column nmaes from a table with a given model
    :param cursor: sqlite3.Cursor
    :param model: pydantics.BaseModel
    :param table_name:  Optional[str] - If not specified, the model name is used
    :return: List[str]
    """
    table_name = model.__name__ if table_name is None else table_name
    query = "PRAGMA table_info(" + table_name + ")"
    rows = await cursor.fetch_all(query)
    return [row[1] for row in rows]


async def get_all(cursor: Database, model: Type[T], table_name: Optional[str] = None) -> List[T]:
    """
    Returns all rows from a table with a given model
    :param cursor: sqlite3.Cursor
    :param model: pydantics.BaseModel
    :param table_name:  Optional[str] - If not specified, the model name is used
    :return: List[model]
    """
    table_name = model.__name__ if table_name is None else table_name
    query = "SELECT * FROM " + table_name
    rows = await cursor.fetch_all(query)
    return [model(**{**row}) for row in rows]


async def get_single_entry(cursor: Database, model: Type[T], id: int, table_name: Optional[str] = None) -> T:
    """
    Returns a single entry from a table with a given model
    :param cursor: sqlite3.Cursor
    :param model: pydantics.BaseModel
    :param id: int
    :param table_name:  Optional[str] - If not specified, the model name is used
    :return: model
    """
    table_name = model.__name__ if table_name is None else table_name
    query = "SELECT * FROM " + table_name + " WHERE id = " + str(id)
    row = await cursor.fetch_one(query)
    return model(**{**row})


async def check_if_exists(
    cursor: Database, model: Type[T], value: any, column: Optional[str] = None, table_name: Optional[str] = None
) -> bool:
    """
    Checks if an entry exists in a table with a given model. If no column is specified, the id column is used, otherwise
    the specified column is used.
    :param cursor: sqlite3.Cursor
    :param model: pydantics.BaseModel
    :param value: any
    :param column: Optional[str] - If not specified, the id column is used
    :param table_name:  Optional[str] - If not specified, the model name is used
    """
    table_name = model.__name__ if table_name is None else table_name
    column = "id" if column is None else column
    query = "SELECT * FROM " + table_name + " WHERE " + column + " = " + str(value)
    row = await cursor.fetch_one(query)
    return row is not None


async def add_entry(
    cursor: Database,
    model: Type[BaseModel],
    data: dict,
    table_name: Optional[str] = None,
) -> None:
    """
    Adds a single entry to a table with a given model
    :param cursor: sqlite3.Cursor
    :param model: pydantics.BaseModel
    :param data: dict
    :param table_name:  Optional[str] - If not specified, the model name is used
    :return: None
    """
    table_name = model.__name__ if table_name is None else table_name
    default_model = model(id=-1)
    query = "INSERT INTO " + table_name + " ("
    for field in model.__annotations__:
        if field == "id":
            continue
        query += field + ","
    query = query[:-1] + ") VALUES ("
    # for each field not in data, take default from data model
    for field in model.__annotations__:
        if field == "id":
            continue
        data_field = data[field] if field in data else default_model.__getattribute__(field)

        if model.__annotations__[field] == int:
            query += str(data_field) + ","
        elif model.__annotations__[field] == dict:
            query += '"' + str(data_field) + '",'
        elif model.__annotations__[field] == list:
            query += '"' + str(data_field) + '",'
        else:
            query += '"' + str(data_field) + '",'

    query = query[:-1] + ")"
    await cursor.execute(query)
    return cursor.lastrowid


async def update_entry(
    cursor: Database,
    model: Type[BaseModel],
    id: int,
    data: dict,
    table_name: Optional[str] = None,
) -> None:
    """
    Updates a single entry from a table with a given model
    :param cursor: sqlite3.Cursor
    :param model: pydantics.BaseModel
    :param id: int
    :param data: dict
    :param table_name:  Optional[str] - If not specified, the model name is used
    :return: None
    """
    table_name = model.__name__ if table_name is None else table_name
    query = "UPDATE " + table_name + " SET "
    for field in data:
        data_field = data[field]

        if model.__annotations__[field] == int:
            query += field + "=" + str(data_field) + ","
        elif model.__annotations__[field] == dict:
            query += field + "=" + '"' + str(data_field) + '",'
        elif model.__annotations__[field] == list:
            query += field + "=" + '"' + str(data_field) + '",'
        else:
            query += field + "=" + '"' + str(data_field) + '",'
    query = query[:-1] + " WHERE id = " + str(id)
    await cursor.execute(query)


async def delete_entry(cursor: Database, model: Type[BaseModel], id: int, table_name: Optional[str] = None) -> None:
    """
    Deletes a single entry from a table with a given model
    :param cursor: sqlite3.Cursor
    :param model: pydantics.BaseModel
    :param id: int
    :param table_name:  Optional[str] - If not specified, the model name is used
    :return: None
    """
    table_name = model.__name__ if table_name is None else table_name
    query = "DELETE FROM " + table_name + " WHERE id = " + str(id)
    await cursor.execute(query)
