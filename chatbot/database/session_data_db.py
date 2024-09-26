import uuid
from typing import Annotated, Dict, Literal, Tuple, TypedDict
import asyncio

from pydantic import BaseModel, conint

# Simulate a database or shared storage for task statuses

class DataStatus(BaseModel):
    status: Literal['started','loading','embedding','completed']
    progress: Annotated[int, conint(ge=0,le=100)] | None = None

task_statuses: Dict[str, DataStatus] = {}

async def simulate_processing(task_id, session_id):
    """
    Simulate file processing with asynchronous sleep.
    """
    await asyncio.sleep(5)  # Simulate time-consuming processing
    task_statuses[task_id] = DataStatus(status="completed")

def start_file_processing(file, session_id):
    """
    Start processing the file, update task status in an asynchronous manner.
    """
    task_id = str(uuid.uuid4())
    task_statuses[task_id] = DataStatus(status="started")
    
    # Simulate processing in background
    asyncio.create_task(simulate_processing(task_id, session_id))
    
    return task_id

def get_task_status(task_id):
    """
    Retrieve the current status of a task.
    """
    if task_id in task_statuses:
        return task_statuses[task_id]
    else:
        return task_statuses[task_id]