import os
import boto3
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
import chainlit as cl
from chainlit.input_widget import Select, Slider
from typing import Optional
from langchain.agents import Tool, AgentExecutor, initialize_agent

aws_region = os.environ["AWS_REGION"]
#aws_profile = os.environ["AWS_PROFILE"]

knowledge_base_id = os.environ["BEDROCK_KB_ID"]

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
  # Fetch the user matching username from your database
  # and compare the hashed password with the value stored in the database
  if (username, password) == ("admin", "admin"):
    return cl.User(identifier="admin", metadata={"role": "admin", "provider": "credentials"})
  else:
    return None

async def setup_settings():

    model_ids = [
        "anthropic.claude-v2", #"anthropic.claude-v2:0:18k",
        "anthropic.claude-v2:1", #"anthropic.claude-v2:1:18k", "anthropic.claude-v2:1:200k", 
        "anthropic.claude-instant-v1"
    ]

    settings = await cl.ChatSettings(
        [
            Select(
                id = "Model",
                label = "Foundation Model",
                values = model_ids,
                initial_index = model_ids.index("anthropic.claude-v2"),
            ),
            Slider(
                id = "Temperature",
                label = "Temperature",
                initial = 0.0,
                min = 0,
                max = 1,
                step = 0.1,
            ),
            Slider(
                id = "TopP",
                label = "Top P",
                initial = 1,
                min = 0,
                max = 1,
                step = 0.1,
            ),
            Slider(
                id="MaxTokenCount",
                label="Max Token Size",
                initial = 2048,
                min = 256,
                max = 4096,
                step = 256,
            ),
            Slider(
                id = "DocumentCount",
                label = "Document Count",
                initial = 1,
                min = 1,
                max = 5,
                step = 1,
            )
        ]
    ).send()

    print("setup_settings complete: ", settings)

    return settings

@cl.on_settings_update
async def setup_agent(settings):

    print("Setup agent with following settings: ", settings)

    bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=aws_region)

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = settings["Model"], 
        model_kwargs = {
            "temperature": settings["Temperature"],
            "top_p": settings["TopP"],
            "top_k": 250,
            "max_tokens_to_sample": int(settings["MaxTokenCount"]),
        },
        streaming = True
    )

    message_history = ChatMessageHistory()
    
    retriever = AmazonKnowledgeBasesRetriever(
        client = bedrock_agent_runtime,
        knowledge_base_id = knowledge_base_id,
        retrieval_config = {
            "vectorSearchConfiguration": {
                "numberOfResults": settings["DocumentCount"]
            }
        }
    )

    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        input_key="question",
        #output_key = "answer",
        chat_memory = message_history,
        return_messages = True,
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": True,
            "memory": memory
            #"prompt": prompt,
            #"memory": ConversationBufferMemory(
            #    memory_key="history",
            #    input_key="question"),
        }
    )

    # Store the chain in the user session
    cl.user_session.set("chain", chain)

def bedrock_list_models(bedrock):
    response = bedrock.list_foundation_models(byOutputModality="TEXT")

    for item in response["modelSummaries"]:
        print(item['modelId'])

@cl.on_chat_start
async def main():

    ##
    #print(f"Profile: {aws_profile} Region: {aws_region}")
    bedrock = boto3.client("bedrock", region_name=aws_region)
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=aws_region)

    #bedrock_list_models(bedrock)

    ##
        
    settings = await setup_settings()

    await setup_agent(settings)


@cl.on_message
async def main(message: cl.Message):

    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain") #RetrievalQA

    res = await chain.ainvoke(
        message.content, 
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    #print(res)
    answer = res["result"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            #print(source_doc.metadata['location']) # {'type': 'S3', 's3Location': {'uri': 's3://xxx'}}
            source_content = f"{source_doc.metadata['score']} | {source_doc.page_content}"
            text_elements.append(
                cl.Text(content=source_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()

