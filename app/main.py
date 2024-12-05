# main.py

from agent.assistant import Agent
from prompts.prompt import instructions
from functions.function import get_order_status_function, get_estimated_delivery_date_function, escalate_to_human_function


def main():
    agent = Agent(instructions=instructions, 
                  functions=[get_order_status_function, get_estimated_delivery_date_function, escalate_to_human_function],
                  vector_store_name="Product Info ShowWise")
    agent.upload_files_to_vector_store(["data/input/ShopWise_Company_Information.pdf"])
    agent.link_vector_store_to_assistant()
    print("Hi, welcome to the ShopWise Assistant, what can I do to help you?")
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input.lower() == "exit":
            print("Thanks for this conversation!")
            break
        
        agent.send_message(user_input)
        ai_response = agent.get_last_response()
        print(f"Assistant: {ai_response}")
        print("\n")
    
if __name__ == "__main__":
    main()