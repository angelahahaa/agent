
import random
from typing import Annotated, Literal
from langchain.tools.base import StructuredTool
from langchain_community.tools import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from chatbot.database import vector_db


def _creeate_fake_tool(name:str, return_direct:bool=False) -> BaseTool:
    if return_direct:
        def fn() -> None:
            return None, {"return_direct": 'image' in name}
        return StructuredTool.from_function(func=fn,name=name,description=name,response_format='content_and_artifact')
    else:
        def fn() -> None:
            return 
        return StructuredTool.from_function(func=fn,name=name,description=name,response_format='content')

@tool
def search_session_data(
    query:str, 
    config:RunnableConfig
    ):
    """A search engine optimized for comprehensive, accurate, and trusted results. 
    Useful for when you need to answer questions about data or files uploaded in current session.
    Input should be a search query.

    Args:
        query: search query to look up

    Returns:
        A list of dictionaries, each containing the 'source' of the data and the 'content' that matches the query.
    """
    # return [
    #     {"source":"zzz_characters.csv","content":"Name:Jane Doe\nHair colour:Black\nDMG:Physical"},
    #     {"source":"zzz_characters.csv","content":"Name:Qingyi\nHair colour:Green\nDMG:Electrical"},
    # ]
    configuration = config.get("configurable", {})
    collection_id = configuration['thread_id']
    docuemnts = vector_db.retriever(collection_id).get_relevant_documents(query)
    return [{'source':d.metadata['source'],'content':d.page_content} for d in docuemnts]

search_internet = TavilySearchResults(max_results=1, name='search_internet')

@tool
def get_session_data_summary(config:RunnableConfig) -> str:
    """ fetch a string that describes what users have in their session file database.
    """
    # return """
    # zzz_characters.csv: data for Jane Doe (Black hair, Physical DMG), Qingyi (Green hair, Electro DMG)
    # Character Design.pptx: details Elara Stormwind, a mage with elemental spirit summoning ability, questing against a dark prophecy.
    # """
    configuration = config.get("configurable", {})
    collection_name = configuration['thread_id']
    data = vector_db.get_database_info(collection_name)
    if not data['data_source']:
        return "No session data."
    return f"Session has data from sources: {data['data_source']}"

@tool
def get_user_info(config: RunnableConfig, include_session_data_summary=True):
    """ Fetch all user information.
    Returns:
        A dictionary of user information. Returns an empty dictionary if no user information found.
    """
    configuration = config.get("configurable", {})
    email = configuration.get("email", "")
    info = {}
    if ('angela' in email) or ('shiya' in email):
        info = {'name':'Peng Shiya', 'department':'RnD', 'studio':'SHA', 'position':'Engineer', 'location':'Shanghai'}
    elif 'pengseng' in email:
        info =  {'name':'Ang Peng Seng', 'department':'RnD', 'studio':'SGP', 'position':'Lead Engineer', 'location':'Singapore'}
    elif 'yuyong' in email:
        info =  {'name':'Ma Yuyong', 'department':'RnD', 'studio':'SHA', 'position':'Senior Producer', 'location':'Shanghai'}
    elif 'art' in email:
        info =  {'name':'Jane Doe', 'department':'Art', 'studio':'CDU', 'position':'Art Director', 'location':'Chengdu'}
    if include_session_data_summary:
        session_data_summary = get_session_data_summary.with_config(config).invoke({})
        info['session_data_summary'] = session_data_summary
    return info


@tool(response_format='content_and_artifact')
def generate_image_with_text(text:str, size=(256,256)):
    """ generates an image with the given text on it. Default size is 256 x 256.
    """
    import base64
    import io

    from PIL import Image, ImageDraw, ImageFont

        # Create a new image with white background
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    
    # Define the text and font (defaulting to a built-in PIL font)
    font = ImageFont.load_default()
    
    # Calculate the bounding box of the text
    bbox = draw.textbbox((0, 0), text, font=font)
    
    # Calculate position to center the text
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2
    
    # Draw the text on the image
    draw.text((x, y), text, font=font, fill="black")

    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return "Image generated and displayed to user.", {'image':encoded_image, 'return_direct':True}

@tool
def spell_backwards(word:str, config: RunnableConfig) -> str:
    """ Spells the word backwards.
    """
    return word[::-1]

@tool
def get_jira_tickets(
    config: RunnableConfig, # for getting project_key and credentials, probably 
    max_results=10):
    """
    Get information about tickets from JIRA, a project management tool.
    This function would interact with the JIRA API to fetch issues.

    Parameters:
        max_results: The maximum number of results to return.

    Returns:
        list: A list of dictionaries representing the retrieved JIRA tickets.
              Each dictionary contains information about a single ticket.
    """
    # Mock implementation: return an empty list of tickets
    return []

@tool
def create_jira_ticket(
        title: str,
        description: str,
        config: RunnableConfig, # for getting project_key and credentials, probably 
        issue_type:Literal['Bug','Task','Story']='Task', 
        priority:Literal['Low','Medium','High']='Medium', 
        labels=None,
        ):
    """
    Create a new ticket in JIRA, a project management system.

    Parameters:
        title: A title that summarises the task.
        description: A detailed description of the task.
        issue_type: The type of issue.
        priority: The priority level of the issue.
        labels: A list of labels to associate with the issue. Defaults to None.

    Returns:
        dict: A dictionary representing the created JIRA ticket, including
              a 'key' attribute which would be the unique identifier of the ticket.
    """
    # Mock implementation: return a dictionary with a fabricated ticket key
    return {'key': f'MOCK-{random.randint(0,9999):04}'}