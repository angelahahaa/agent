
import os
import random
from typing import Annotated, Dict, List, Literal
from langchain.tools.base import StructuredTool
from langchain_community.tools import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from chatbot.database import vector_db
import requests
@tool
def search_user_projects():
    """Returns a list of latest project the user is currently working on from internal database.
    """
    return [{
        "project_name":"cyberpunk 2077",
        "description":"Open-world, action-adventure RPG",
        },
        {
        "project_name":"Detroit: Become Human",
        "description":"Interactive drama and action-adventure game.",
        }
        ]


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

search_internet = TavilySearchResults(max_results=4, name='search_internet')

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
def get_user_info(config: RunnableConfig, include_session_data_summary=True) -> Dict:
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
def generate_images(
    prompt: str,
    model:Literal['sdv1-5-base','sdv1-5-dreamshaper']='sdv1-5-dreamshaper'
):
    """Generates AI-GC (AI-generated content) images based on the provided text prompt using a specified model.

    Parameters:
        prompt (str): A string containing the textual description or keywords that will be used to guide the AI-GC image generation.
        model (Literal['sdv1-5-base', 'sdv1-5-dreamshaper']): The model to use for generating the images. 
            Defaults to 'sdv1-5-dreamshaper'.
    """
    url = {
        'sdv1-5-base':"https://z3601svaoyjoa4e0.us-east-1.aws.endpoints.huggingface.cloud",
        'sdv1-5-dreamshaper':"https://y9oiaej9g2ke00pc.us-east-1.aws.endpoints.huggingface.cloud",
        }[model]
    headers = {
        "Accept" : "application/json",
        "Authorization": f"Bearer {os.environ['HF_API_KEY']}",
        "Content-Type": "application/json" 
    }
    payload = {
        "inputs": {"prompt":prompt},
        "parameters": {"num_images_per_prompt":1}
    }
    response = requests.post(url, headers=headers, json=payload)
    return "Images generated and displayed to user.", {'images':response.json()['images'], 'return_direct':True}

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
    project_name: str='*',
    issue_type:Literal['Bug','Task','Story','*']='Task', 
    max_results=10):
    """
    Get information about tickets from JIRA, a project management tool.
    This function would interact with the JIRA API to fetch issues.

    Parameters:
        project_name: comma separated string for project name filter, use '*' to search for all projects. 
        max_results: The maximum number of results to return.
        issue_type: The type of issue, use '*' for all issues.

    Returns:
        list: A list of dictionaries representing the retrieved JIRA tickets.
              Each dictionary contains information about a single ticket.
    """
    # Mock implementation: return an empty list of tickets
    if 'cyberpunk' in project_name:
        return [
            {'key': f'JIRA-{random.randint(0,9999):04}','link':'http://exmple.com/', 'title':'Texture tiling issue at scene 1-3.'},
            {'key': f'JIRA-{random.randint(0,9999):04}','link':'http://exmple.com/', 'title':'NPC dialogue not loading properly in the tutorial phase.'},
            ]
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
    return {'key': f'JIRA-{random.randint(0,9999):04}','link':'http://exmple.com/'}

@tool
def cancel_clarify_request(tool_name:str, cancel_reason:str):
    """ Exits clarifying requirements for a tool.
    Args:
        tool_name: name of the tool 
        cancel_reason: reason for cancellation
    """
    return "Exiting from 'clarify requirements mode'. Proceed with the conversation."


# === IT Tools ===
@tool
def search_IT_procedure(query: str) -> List[Dict]:
    """
    Searches the IT resource solutions from internal database for entries matching the given query.
    
    Parameters:
        query (str): A string representing the issue or requirement to search for,
                     such as "website whitelisting", "installing application", "network drive permissions",
                     "email server configuration", "printer setup", "WiFi connectivity issues",
                     "remote desktop access", "software license management", etc.
    
    Returns:
        list: A list of dictionaries containing details of the found solutions. Each dictionary
              typically includes fields like 'solution_id', 'description', 'procedure', 'tags', and 'date_updated'.
    """
    return [
        {
            'solution_id': 'IT-SOL-456',
            'description': 'How to install an application using the official app store for approved applications or request approval for new applications.',
            'procedure': 'Step 1: Visit the internal app store at http://appstore.virtuosgames.com to check if the application is listed among the approved applications.'
                'Step 2: If the application is listed, download and install it directly from the app store.'
                'Step 3: If the application is not found in the app store, create a new FreshService support system ticket to request its inclusion and approval.',
            'tags': ['installation', 'application', 'approval', 'app store'],
            'date_updated': '2024-09-14',
            'additional_information': {
                'approved_application_list':[
                    {'name':'substance painter 3D','version':'<1.4.0'},
                    {'name':'maya','version':'*'},
                    {'name':'blender','version':'*'},
                    ]
            }
        }]

@tool
def create_IT_fresh_service_ticket(issue_description: str) -> Dict:
    """
    Raises a new ticket in the FreshService support system.

    This function should be used in the following scenarios:
    - When no suitable solution is found in the IT database using the search_IT_database function.
    - When the database explicitly instructs the user to create a ticket for further assistance.
    - When the user requires assistance with an IT-related issue that is not covered by existing documentation.

    Parameters:
        issue_description (str): A detailed description of the issue or request that needs support.

    Returns:
        str: A JSON-formatted string containing the confirmation message including the ticket ID or any error message if the ticket creation fails.

    Example Usage:
        >>> raise_freshservice_ticket("Cannot connect to the company WiFi network.")
        '{"status": "success", "ticket_id": "FS-TK-45678"}'
    """
    return {"status": "success",'ticket_id': f'FS-TK--{random.randint(0,9999):04}','link':'http://exmple.com/'}