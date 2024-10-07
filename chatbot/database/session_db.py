import asyncio
import uuid
from datetime import datetime
from typing import List
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

DEFAULT_TITLE = 'New Chat'

class SessionInfo(BaseModel):
    id: uuid.UUID
    last_modified: datetime
    agent_name: str
    title: str

    @staticmethod
    def from_db_row(row):
        return SessionInfo(id=row['id'], last_modified=row['last_modified'], agent_name=row['agent_name'], title=row['title'])

class SessionManager:
    def __init__(self, conninfo:str):
        self.pool = AsyncConnectionPool(conninfo=conninfo, open=False)

    async def setup(self) -> None:
        await self.pool.open()
        await self.create_tables_if_not_exists()

    async def create_tables_if_not_exists(self) -> None:
        async with self.pool.connection() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id UUID PRIMARY KEY,
                    username VARCHAR(255) NOT NULL,
                    last_modified TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    archived BOOLEAN DEFAULT FALSE,
                    agent_name VARCHAR(255) NOT NULL,
                    title TEXT NOT NULL
                )
            ''')

    async def create_session(self, username: str, agent_name: str, title: str = DEFAULT_TITLE) -> SessionInfo:
        id = uuid.uuid4()
        last_modified = datetime.now()
        async with self.pool.connection() as conn:
            await conn.execute('''
                INSERT INTO user_sessions(id, username, last_modified, archived, agent_name, title)
                VALUES(%s, %s, %s, %s, %s, %s)
            ''', (str(id), username, last_modified, False, agent_name, title))
        return SessionInfo(id=id, last_modified=last_modified, agent_name=agent_name, title=title)

    async def update_session(self, id: uuid.UUID, title:str|None=None) -> SessionInfo | None:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute('''
                    UPDATE user_sessions
                    SET last_modified = %s,
                        title = COALESCE(%s, title)
                    WHERE id = %s
                    RETURNING *
                ''', (datetime.now(), title, str(id)))
                row = await cur.fetchone()
                
                if row is None:
                    return None
                
                return SessionInfo.from_db_row(row)

    async def archive_session(self, id: uuid.UUID) -> None:
        async with self.pool.connection() as conn:
            await conn.execute('''
                UPDATE user_sessions
                SET archived = TRUE
                WHERE id = %s
            ''', (str(id),))

    async def get_sessions(self, username: str) -> List[SessionInfo]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute('''
                    SELECT *
                    FROM user_sessions
                    WHERE username = %s AND archived = FALSE
                    ORDER BY last_modified DESC
                ''', (username,))
                rows = await cur.fetchall()
                return [SessionInfo.from_db_row(row) for row in rows]

    async def get_session(self, id: uuid.UUID) -> SessionInfo | None:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute('''
                    SELECT *
                    FROM user_sessions
                    WHERE id = %s
                ''', (str(id),))
                row = await cur.fetchone()
                if row:
                    return SessionInfo.from_db_row(row)
                else:
                    return None

async def main():
    from chatbot import config
    conn_string = f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}/{config.DB_DATABASE}?sslmode=disable"
    session_manager = SessionManager(conn_string)
    await session_manager.setup()
    username = 'example_user'
    session_info = await session_manager.create_session(username, 'abc')
    print(session_info.model_dump_json(indent=2))
    session_info = await session_manager.get_session(session_info.id)
    assert session_info is not None
    print(session_info.model_dump_json(indent=2))

    print(f"Sessions: {await session_manager.get_sessions(username)}")
    await session_manager.update_session(session_info.id)
    print("Session updated with new timestamp.")
    print(f"Sessions: {await session_manager.get_sessions(username)}")
    await session_manager.archive_session(session_info.id)
    print("Session archived.")
    print(f"Sessions: {await session_manager.get_sessions(username)}")

if __name__ == '__main__':
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())