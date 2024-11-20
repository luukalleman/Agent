# main.py

import json
import os
from agent.assistant import Agent
from prompts.prompt import instructions
from functions.function import get_order_status_function, get_estimated_delivery_date_function, escalate_to_human_function


# ---------------------------------------------------
# Main execution
# ---------------------------------------------------

def main():
    """
    Main function to run the assistant chatbot.
    """
    # Create the agent with the Function instances
    agent = Agent(
        instructions=instructions,
        functions=[
            get_order_status_function,
            get_estimated_delivery_date_function,
            escalate_to_human_function
        ],
        vector_store_name="Product Knowledge Base"

    )
    agent.upload_files_to_vector_store(["data/input/ShopWise_Company_Information.pdf"])
    agent.link_vector_store_to_assistant()
    agent.send_message("what is the status of order 12345")
    print(agent.get_last_response())
    # # Start the conversation with the agent
    # agent.send_message("I don't want to talk with an AI chatbot.")
    # print(agent.get_last_response())

    # # User provides contact info
    # agent.send_message("My email is luuk@alleman.com")
    # print(agent.get_last_response())

if __name__ == "__main__":
    main()