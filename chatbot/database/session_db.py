import asyncio
import uuid
from datetime import datetime
from typing import List, TypedDict

import asyncpg
from pydantic import BaseModel

class SessionInfo(BaseModel):
    session_id: uuid.UUID
    last_modified: datetime

class SessionManager:
    def __init__(self):
        self._pool: asyncpg.pool.Pool | None = None
    
    @property
    def pool(self):
        if self._pool is None:
            raise RuntimeError("must create_pool before other operations")
        return self._pool

    async def create_pool(self, user: str, password: str, database: str, host: str) -> None:
        self._pool = await asyncpg.create_pool(
            user=user,
            password=password,
            database=database,
            host=host
        )


    async def create_table_if_not_exists(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id UUID PRIMARY KEY,
                    username VARCHAR(255) NOT NULL,
                    last_modified TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    archived BOOLEAN DEFAULT FALSE
                )
            ''')

    async def session_exists(self, session_id: uuid.UUID) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                '''
                SELECT EXISTS(
                    SELECT 1 FROM user_sessions WHERE session_id = $1
                )
                ''', str(session_id))
            return bool(result)

    async def create_session(self, username: str) -> SessionInfo:
        session_id = uuid.uuid4()
        async with self.pool.acquire() as conn:
            last_modified = datetime.now()
            await conn.execute(
                '''
                INSERT INTO user_sessions(session_id, username, last_modified, archived)
                VALUES($1, $2, $3, $4)
                ''', str(session_id), username, last_modified, False)
        return SessionInfo(session_id=session_id, last_modified=last_modified)

    async def update_session(self, session_id: uuid.UUID) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                '''
                UPDATE user_sessions
                SET last_modified = $2
                WHERE session_id = $1
                ''', str(session_id), datetime.now())

    async def archive_session(self, session_id: uuid.UUID) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                '''
                UPDATE user_sessions
                SET archived = TRUE
                WHERE session_id = $1
                ''', str(session_id))

    async def get_sessions(self, username: str) -> List[SessionInfo]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                '''
                SELECT session_id, last_modified
                FROM user_sessions
                WHERE username = $1 AND archived = FALSE
                ORDER BY last_modified DESC
                ''', username)
            return [SessionInfo(session_id=row['session_id'], last_modified=row['last_modified']) for row in rows]

async def main():
    session_manager = SessionManager()
    await session_manager.create_pool('postgres', 'password', 'postgres', 'localhost')
    await session_manager.create_table_if_not_exists()

    username = 'example_user'
    session_info = await session_manager.create_session(username)
    print(f"Session created with ID: {session_info}")

    print(f"Sessions: {await session_manager.get_sessions(username)}")

    await session_manager.update_session(session_info.session_id)
    print("Session updated with new timestamp.")

    print(f"Sessions: {await session_manager.get_sessions(username)}")

    await session_manager.archive_session(session_info.session_id)
    print("Session archived.")

    print(f"Sessions: {await session_manager.get_sessions(username)}")

if __name__ == '__main__':
    asyncio.run(main())