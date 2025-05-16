# Execute the workflow
from src.graph.workflow import create_workflow
from langgraph.types import Command

app = create_workflow()

initial_state = {
    "message": ["plz show me all products that are expensive than $50."],
    "aggregated_context": "",  # Initialize with empty string
    "curr_context": "",
    "query_to_retrieve_or_answer": "",
    "tool": "",
    "curr_state": "",
    "answer_quality": "GOOD",
    "human_feedback": ""
}

final_state = app.invoke(
    initial_state,
    config={"configurable": {"thread_id": 1}}
)

print("final_state:")
# print(final_state)
print("--------------------------------")
if "__interrupt__" in final_state.keys():
    input_query = input("Enter new query:")
    app.invoke(Command(resume=[{"args": input_query}]), config={"configurable": {"thread_id": 1}})
# Show the final response
# print(final_state["message"])