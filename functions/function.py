# function.py
import json
import os
from data.input.data import df_orders


class Function:
    def __init__(self, func, name, description, parameters):
        self.func = func  # The actual Python function
        self.name = name
        self.description = description
        self.parameters = parameters

    def execute(self, **kwargs):
        """Execute the encapsulated function with provided arguments."""
        return self.func(**kwargs)
    
def get_order_status(order_number):
    """
    Retrieve the current status of an order given its order number.
    """
    order_number = str(order_number)
    status = df_orders.loc[df_orders['order_number'] == order_number, 'status']
    if not status.empty:
        return f"The status of order number {order_number} is {status.values[0]}."
    else:
        return f"Order number {order_number} not found."

def get_estimated_delivery_date(order_number):
    """
    Provide the estimated delivery date for an order given its order number.
    """
    order_number = str(order_number)
    delivery_date = df_orders.loc[df_orders['order_number'] == order_number, 'estimated_delivery']
    if not delivery_date.empty:
        return f"The estimated delivery date for order number {order_number} is {delivery_date.values[0]}."
    else:
        return f"Order number {order_number} not found."

def escalate_to_human(reason, thread_id, contact_info):
    """
    Escalate the conversation to a human by saving the thread ID, reason, and contact info to a JSON file.
    """
    escalation_data = {
        'thread_id': thread_id,
        'reason': reason,
        'contact_info': contact_info
    }
    
    # Check if file exists and load or initialize the data
    if os.path.exists('data/output/escalations.json'):
        with open('escalations.json', 'r') as f:
            try:
                escalations = json.load(f)
            except json.JSONDecodeError:
                escalations = []  # If file exists but is empty or invalid
    else:
        escalations = []

    # Append the new escalation
    escalations.append(escalation_data)

    # Write the updated list back to the file
    with open('data/output/escalations.json', 'w') as f:
        json.dump(escalations, f, indent=4)

    return "Thank you. I've escalated your request to a human representative, and they will contact you shortly."

# ---------------------------------------------------
# Create Function instances
# ---------------------------------------------------

get_order_status_function = Function(
    func=get_order_status,
    name="get_order_status",
    description="Retrieve the current status of an order given its order number.",
    parameters={
        "type": "object",
        "properties": {
            "order_number": {
                "type": "string",
                "description": "The unique order number assigned to the customer's order."
            }
        },
        "required": ["order_number"],
        "additionalProperties": False
    }
)

get_estimated_delivery_date_function = Function(
    func=get_estimated_delivery_date,
    name="get_estimated_delivery_date",
    description="Provide the estimated delivery date for an order given its order number.",
    parameters={
        "type": "object",
        "properties": {
            "order_number": {
                "type": "string",
                "description": "The unique order number assigned to the customer's order."
            }
        },
        "required": ["order_number"],
        "additionalProperties": False
    }
)

escalate_to_human_function = Function(
    func=escalate_to_human,
    name="escalate_to_human",
    description="Escalate the conversation to a human representative when the assistant cannot assist the user.",
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "The reason why the conversation should be escalated to a human."
            },
            "contact_info": {
                "type": "string",
                "description": "The contact_info of the person that wants to speak with a human."
            }
        },
        "required": ["reason", "contact_info"],
        "additionalProperties": False
    }
)
