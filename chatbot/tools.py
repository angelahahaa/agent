
from langchain_core.runnables import (RunnableConfig)
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults

@tool
def get_user_info(config: RunnableConfig):
    """get_user_info"""
    configuration = config.get("configurable", {})
    return {"email":"angela@gmail.com"}


@tool
def search_session_data(query:str):
    """A search engine optimized for comprehensive, accurate, and trusted results. 
    Useful for when you need to answer questions about data or files uploaded in current session.
    Input should be a search query.

    Args:
        query: search query to look up

    Returns:
        A list of dictionaries, each containing the 'source' of the data and the 'content' that matches the query.
    """
    return [
        {"source":"zzz_characters.csv","content":"Name:Jane Doe\nHair colour:Black\nDMG:Physical"},
        {"source":"zzz_characters.csv","content":"Name:Qingyi\nHair colour:Green\nDMG:Electrical"},
    ]

search_internet = TavilySearchResults(max_results=1, name='search_internet')
# @tool
# def search_internet(query:str):
#     """A search engine optimized for comprehensive, accurate, and trusted results. 
#     Useful for when you need to answer questions about current events. 
#     Input should be a search query.
    
#     Args:
#         query: search query to look up
#     """
#     ...
@tool
def get_session_data_summary(config:RunnableConfig) -> str:
    """ fetch a string that describes what users have in their session file database.
    """
    return """
    zzz_characters.csv: data for Jane Doe (Black hair, Physical DMG), Qingyi (Green hair, Electro DMG)
    Character Design.pptx: details Elara Stormwind, a mage with elemental spirit summoning ability, questing against a dark prophecy.
    """

@tool(response_format='content_and_artifact')
def generate_image_with_text(text:str, size=(256,256)):
    """ generates an image with the given text on it. Default size is 256 x 256.
    """
    from PIL import Image, ImageDraw, ImageFont
    import io
    import base64

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