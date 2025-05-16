from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, TypedDict, Optional
from dotenv import load_dotenv
import os
from pinecone import Pinecone
import json
from openai import OpenAI
from pymongo import MongoClient
from langchain.output_parsers import PydanticOutputParser
from src.models.schema import PlanExecute, reasoningOutput , MongoQuery
from src.models.prompts import reasoning_prompt_template,prompt_template
from src.database.mongodb_connector import MongoDBConnector
from src.database.pinecone_connector import PineconeConnector
from langgraph.types import Command, interrupt

def retrieve_or_answer(state: PlanExecute):
    """Decide whether to retrieve or answer the question based on the current state.
    Args:
        state: The current state of the plan execution.
    Returns:
        updates the tool to use .
    """
    state["curr_state"] = "decide_tool"
    reasoning_step = {
        "stage": "decide_tool",
        "content": f"Deciding whether to retrieve or answer"
    }
    state["reasoning_steps"].append(reasoning_step)

    print("deciding whether to retrieve or answer")
    if state["tool"] == "MongoDB_retrieval":
        reasoning_step = {
            "stage": "decide_tool",
            "content": f"Chosen tool is MongoDB_retrieval"
        }
        state["reasoning_steps"].append(reasoning_step)
        return "chosen_tool_is_MongoDB_retrieval"
    elif state["tool"] == "pinecone_retrieval":
        reasoning_step = {
            "stage": "decide_tool",
            "content": f"Chosen tool is pinecone_retrieval"
        }
        state["reasoning_steps"].append(reasoning_step)
        return "chosen_tool_is_pinecone_retrieval"
    elif state["tool"] == "combined_search":
        reasoning_step = {
            "stage": "decide_tool",
            "content": f"Chosen tool is combine_search"
        }
        state["reasoning_steps"].append(reasoning_step)
        return "chosen_tool_is_combine_search"
    elif state["tool"] == "Greet":
        reasoning_step = {
            "stage": "decide_tool",
            "content": f"Chosen tool is Greet"
        }
        state["reasoning_steps"].append(reasoning_step)
        return "chosen_tool_is_answer"
    elif state["tool"] == "human_in_the_loop":
        reasoning_step = {
            "stage": "decide_tool",
            "content": f"Chosen tool is human_in_the_loop"
        }
        state["reasoning_steps"].append(reasoning_step)
        return "chosen_tool_is_human_in_the_loop"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")  
def check_query_sufficiency(state: PlanExecute):
    """Determine if the query is sufficient or needs human clarification.
    Args:
        state: The current state of the plan execution.
    Returns:
        String indicating the next node to go to.
    """
    if state.get("query_sufficient") is False:
        print("Query is insufficient, requesting human input")
        return "needs_human_input"
    else:
        print("Query is sufficient, proceeding with reasoning")
        return "query_is_sufficient"
    
def create_reasoning_chain():
    my_reasoning_prompt_template=reasoning_prompt_template
    template  = my_reasoning_prompt_template+"\n User Message : {message} \n Human Feedback : {human_feedback} \n Aggregated Context : {aggregated_context}"

    reasoning_prompt = PromptTemplate(
        template=template,
        input_variables=["message", "human_feedback", "aggregated_context"],
    )

    reasoning_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
    reasoning_chain = reasoning_prompt | reasoning_llm.with_structured_output(reasoningOutput)
    return reasoning_chain

def reasoningNode(state: PlanExecute):
    """ Run the task handler chain to decide which tool to use to execute the task.
    Args:
       state: The current state of the plan execution.
    Returns:
       The updated state of the plan execution.
    """
    state["curr_state"] = "task_handler"
    if "reasoning_steps" not in state:
        state["reasoning_steps"] = []
    # print("aggregated context:")
    # print(state["aggregated_context"])
    print("--------------------------------")
    inputs = {
                "message": state["message"],
                "human_feedback": state["human_feedback"],
                "aggregated_context": state["aggregated_context"],
            }
    # print("inputs:")
    # print(inputs)
    reasoning_chain = create_reasoning_chain()
    
    output = reasoning_chain.invoke(inputs)
    # print("reasoning output:")
    # print(output)
    print("--------------------------------")
    print(output.tool)
    if output.tool == "MongoDB_retrieval":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"]="MongoDB_retrieval"
    
    elif output.tool == "pinecone_retrieval":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"]="pinecone_retrieval"
    elif output.tool == "combined_search":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"]="combined_search"
    elif output.tool == "Greet":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"]="Greet"
    elif output.tool == "Human-in-the-Loop":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"]="human_in_the_loop"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")
    return state  

def MongoDBretrievalNode(state: PlanExecute):
    """Retrieve the relevant information from the MongoDB database using ChatGPT to generate queries.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state of the plan execution.
    """
    # ... existing code ...
    
    # At the end of the function, before returning state:


    connector=MongoDBConnector()
    collection = connector.collection

    # Create the parser
    parser = PydanticOutputParser(pydantic_object=MongoQuery)

    # Create the prompt template
    template = prompt_template+"\n User Message : {message}"

    prompt = PromptTemplate(
        template=template,
        input_variables=["message"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    # Generate the query
    query = state["query_to_retrieve_or_answer"]
    prompt_value = prompt.format(message=query)
    response = llm.invoke(prompt_value)
    def strip_comments(json_string):
        import re
        # Remove single-line comments (// comment)
        pattern = r'//.*?(?=\n|$)'
        clean_json = re.sub(pattern, '', json_string)
        return clean_json

    # Then use it before parsing
    clean_content = strip_comments(response.content)
    
    # print("Response:")
    # print(response)
    print("--------------------------------")
    print("MongoDB query:",clean_content)
    mongo_query = json.loads(clean_content)
    # print(mongo_query)
    print("--------------------------------")
    # Execute the query
    if mongo_query.get("query_type") == "aggregate":
        pipeline = []
        if mongo_query.get("filter_conditions"):
            pipeline.append({"$match": mongo_query.get("filter_conditions")})
        
        if mongo_query.get("aggregation_pipeline"):
            pipeline.extend(mongo_query.get("aggregation_pipeline"))
        
        if mongo_query.get("sort_conditions"):
            pipeline.append({"$sort": mongo_query.get("sort_conditions")})
        
        if mongo_query.get("limit") and mongo_query.get("limit") > 0:
            pipeline.append({"$limit": mongo_query.limit})

        results = list(collection.aggregate(pipeline))
    else:

        cursor = collection.find(
            mongo_query.get("filter_conditions"),
            mongo_query.get("projection") or {
                "name": 1,
                "brand": 1,
                "price": 1,
                "pricePer": 1,
                "department": 1,
                "weight_each": 1,
                "weight_unit": 1,
                "text": 1,
                "showImage": 1,
                "href": 1,
                "sku": 1,
                "relateds": 1,
                "price_each": 1,
                "popularity": 1,
                "_id": 0
            }
        )

        if mongo_query.get("sort_conditions"):
            cursor = cursor.sort(list(mongo_query.get("sort_conditions").items()))
        if mongo_query.get("limit") and mongo_query.get("limit") > 0:
            cursor = cursor.limit(mongo_query.get("limit"))

        results = list(cursor)
    # print("Results:")
    # print(results)
    # print("--------------------------------")

    # Format the results
    formatted_results = []
    for result in results:
        # print(result)
        if mongo_query.get("query_type") == "aggregate":
            formatted_result = {
                "result": result  # You might want to format this differently based on the aggregation
            }
        else:
            formatted_result = {
                "name": result.get("name", "N/A"),
                "brand": result.get("brand", "N/A"),
                "price": f"${result.get('price', 'N/A')}",
                "price_per_unit": f"${result.get('pricePer', 'N/A')}/{result.get('weight_unit', 'unit')}",
                "department": result.get("department", "N/A"),
                "weight": f"{result.get('weight_each', 'N/A')} {result.get('weight_unit', 'units')}",
                "description": result.get("text", "N/A"),
                "image": result.get("showImage", "N/A"),
                "href": result.get("href", "N/A"),
                "sku": result.get("sku", "N/A"),
                "relateds": result.get("relateds", "N/A"),
                "price_each": result.get("price_each", "N/A"),
                "popularity": result.get("popularity", "N/A")
            }
        formatted_results.append(formatted_result)
    
    # Update the state with the results
    print("Formatted results:")
    print(formatted_results)
    print(len(formatted_results))
    
    if(mongo_query.get("query_type") == "find"):
        state["curr_context"] = "The number of products is "+str(len(formatted_results))+" and the products are "+str(formatted_results)+"\n The correct total number of products is "+str(len(formatted_results))
    else:
        state["curr_context"] = str(formatted_results)
    state["aggregated_context"] += "\n" + state["curr_context"]

    reasoning_step = {
        "stage": "mongodb_retrieval",
        "content": f"Retrieved {len(formatted_results)} products from MongoDB.\nQuery: {state['query_to_retrieve_or_answer']}."
    }
    state["reasoning_steps"].append(reasoning_step)

    return state

def pineconeretrievalNode(state: PlanExecute):
    """Retrieve the relevant information from the Pinecone database.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state of the plan execution.
    """
    # ... existing code ...
    
    # At the end of the function, before returning state:


    client = OpenAI()
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("INDEX_NAME")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    query_embedding = client.embeddings.create(
        input=state["query_to_retrieve_or_answer"],
        model=EMBEDDING_MODEL
    ).data[0].embedding

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=5,
        namespace="Cheese_Data",
        include_metadata=True
    )

    relevant_products = results.matches
    formatted_info = []

    for product in relevant_products:
        info = f"Product: {product.metadata.get('name', 'N/A')}\n"
        info += f"Department: {product.metadata.get('department', 'N/A')}\n"
        info += f"Price: ${product.metadata.get('price', 'N/A')}\n"
        if product.metadata.get('LB_price'):
            info += f"Price per pound: ${product.metadata.get('price_each')}/lb\n"
        info += f"Brand: {product.metadata.get('brand', 'N/A')}\n"
        info += f"Similarity Score: {product.score:.2f}\n"
        info += f"Product URL: {product.metadata.get('href', 'N/A')}\n"
        info += f"image_url: {product.metadata.get('showImage', 'N/A')}\n"
        info += f"UPC: {product.metadata.get('sku', 'N/A')}\n"
        info += f"SKU: {product.metadata.get('sku', 'N/A')}\n"
        info += f"Related Products: {product.metadata.get('relateds', 'N/A')}\n"
        formatted_info.append(info)

    context = "\n".join(formatted_info)
    state["curr_context"] = context
    state["aggregated_context"] += "\n" + state["curr_context"]

    reasoning_step = {
        "stage": "pinecone_retrieval",
        "content": f"Retrieved semantic search results from Pinecone.\nQuery: {state['query_to_retrieve_or_answer']}."
    }
    state["reasoning_steps"].append(reasoning_step)

    return state


def answerNode(state: PlanExecute):
    """Answer the question from the given context.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state of the plan execution.
    """


    # Create a prompt template for generating answers
    answer_prompt_template = """You are a helpful cheese expert assistant. Your task is to answer the user's question about cheese products based on the provided context.


User Question: {question}

Guidelines:
1. Provide clear, concise, and accurate answers
2. If the context doesn't contain enough information, say so
3. Format prices and measurements appropriately
4. Highlight key features and benefits of the products
5. If multiple products are mentioned, compare them when relevant
6. Be friendly and professional in your tone
7. if the user ask for all products, first return the correct number of products and then return all products.
9. If the use greet you, greet them back.
10 You must return the number of products.
11. You must get image link from the products and show image.

Context:
{context}
Your answer:"""

    # Create the prompt
    answer_prompt = PromptTemplate(
        template=answer_prompt_template,
        input_variables=["question","context"]
    )

    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    # Generate the answer
    # print("Context:", state["curr_context"])
    # print("Question:", state["query_to_retrieve_or_answer"])
    prompt_value = answer_prompt.format(
        question=state["query_to_retrieve_or_answer"],
        context=state["curr_context"]
    )
    # print(state['curr_context'])
    
    
    response = llm.invoke(prompt_value)
    # print(response.content)
    client = OpenAI()
    evaluation_prompt = f"""
    You are an AI assistant evaluating the quality of an answer about cheese products.
    
    Original Question: {state["message"]}
    
    Answer to evaluate: {response.content}
    
    Please evaluate if this answer is satisfactory and provides useful information.
    Return only "GOOD" if the answer is informative and addresses the question well.
    Return only "POOR" if the answer is vague, uninformative, or doesn't properly address the question.
    """
    
    evaluation_response = client.chat.completions.create(
        model="gpt-4o",  # Using a faster model for evaluation
        messages=[
            {"role": "system", "content": "You are a quality evaluation assistant."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0,
        max_tokens=10
    )
    
    quality_assessment = evaluation_response.choices[0].message.content.strip()
    print("Quality assessment:", quality_assessment)
    
    # Store the answer and quality assessment in state
    state["answer_quality"] = quality_assessment
    # print(response.content)
    # Update the state with the answer
    state["message"].append(HumanMessage(content=response.content))

    reasoning_step = {
        "stage": "check_quality_of_answer",
        "content": f"Check the quality of the answer: {quality_assessment}."
    }
    state["reasoning_steps"].append(reasoning_step)
    # ... existing code ...
    
    # After generating the response and before returning state:

    # print("Answer:")
    # print(state["message"])
    # print("--------------------------------")
    state["retry_count"] = state["retry_count"] + 1
    return state

def check_answer_quality(state: PlanExecute):
    """Decide whether to end or retry based on answer quality.
    Args:
        state: The current state of the plan execution.
    Returns:
        String indicating the next node to go to.
    """
    quality = state.get("answer_quality", "POOR")
    # print(quality)
    
    # Check if we've tried too many times (to prevent infinite loops)
    retry_count = state.get("retry_count", 0)
    print(retry_count)
    # print(state['aggregated_context'])
    if retry_count >= 2:
        print("Maximum retry count reached, ending workflow")
        return "end_workflow"
    
    if "GOOD" in quality:
        print("Answer quality is GOOD, ending workflow")
        return "end_workflow"
    else:
        print("Answer quality is POOR, retrying reasoning")
        # Increment retry count
        # Optional: Append feedback to help improve next reasoning cycle
        state["aggregated_context"] += "\nNote: Previous answer was insufficient. Please provide more specific information."

        reasoning_step = {
            "stage": "check_answer_quality",
            "content": f"Answer quality is POOR, retrying reasoning"
        }
        state["reasoning_steps"].append(reasoning_step)
        
        return "retry_reasoning"
        
def human_in_the_loopNode(state: PlanExecute):
    """
    This function is used to handle queries that are not clear or ambiguous.
    It will ask the user for more information and then update the state with the new query.
    """
    # ... existing code ...
    
    # After getting human feedback and before returning state:


    print("Human in the loop")
    state["curr_state"] = "human_in_the_loop"
    response = interrupt({"query": state["query_to_retrieve_or_answer"]})
    # Command(resume=[{"args":"Help me."}])
    # print("response:")
    # print(response) 
    # print("--------------------------------")
    state["human_feedback"] = response[0]['args']

    reasoning_step = {
        "stage": "human_in_the_loop",
        "content": f"Requested human feedback.\nQuery: {state['query_to_retrieve_or_answer']}.\nFeedback received: {state['human_feedback']}"
    }
    state["reasoning_steps"].append(reasoning_step)


    return state


def combineSearchNode(state: PlanExecute):
    """Retrieve information from both MongoDB and Pinecone and combine the results.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state of the plan execution.
    """
    # Store original query
    original_query = state["query_to_retrieve_or_answer"]
    print(original_query)
    # Run MongoDB retrieval
    mongo_state = MongoDBretrievalNode(state.copy())
    mongo_context = mongo_state["curr_context"]
    
    # Run Pinecone retrieval
    pinecone_state = pineconeretrievalNode(state.copy())
    pinecone_context = pinecone_state["curr_context"]

    
    # Combine results
    state["curr_context"] = f"MongoDB Results:\n{mongo_context}\n\nPinecone Results:\n{pinecone_context}"
    state["aggregated_context"] += "\n" + state["curr_context"]
    
    reasoning_step = {
        "stage": "combined_search",
        "content": f"Retrieved results from both MongoDB and Pinecone.\nQuery: {original_query}"
    }
    state["reasoning_steps"].append(reasoning_step)
    
    return state



def create_workflow():
    agent_workflow = StateGraph(PlanExecute)
    agent_workflow.add_node("reasoning", reasoningNode)
    agent_workflow.add_node("MongoDB_retrieval", MongoDBretrievalNode)
    agent_workflow.add_node("pinecone_retrieval", pineconeretrievalNode)
    agent_workflow.add_node("answer", answerNode)
    agent_workflow.add_node("combineSearchNode", combineSearchNode)
    agent_workflow.add_node("human_in_the_loop", human_in_the_loopNode)
    agent_workflow.add_edge(START, "reasoning")
    agent_workflow.add_edge("MongoDB_retrieval", "answer")
    agent_workflow.add_edge("pinecone_retrieval", "answer")
    agent_workflow.add_edge("human_in_the_loop", "reasoning")
    agent_workflow.add_edge("combineSearchNode", "answer")
    
    agent_workflow.add_conditional_edges(
        "reasoning",
        retrieve_or_answer,
        {
            "chosen_tool_is_MongoDB_retrieval": "MongoDB_retrieval",
            "chosen_tool_is_pinecone_retrieval": "pinecone_retrieval",
            "chosen_tool_is_answer": "answer",
            "chosen_tool_is_human_in_the_loop": "human_in_the_loop",
            "chosen_tool_is_combine_search": "combineSearchNode"
        },
    )
    agent_workflow.add_conditional_edges(
        "answer",
        check_answer_quality,
        {
            "end_workflow": END,
            "retry_reasoning": "reasoning"
        }
    )

    # agent_workflow.add_edge("answer", END)


    checkpointer = MemorySaver()

    app = agent_workflow.compile(checkpointer=checkpointer)
    mermaid_code = app.get_graph().draw_mermaid()
    # print(mermaid_code)

    return app



