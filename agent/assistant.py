import json
from openai import OpenAI

class Agent:
    def __init__(self, instructions, model="gpt-4o", functions=None, vector_store_name=None):
        self.client = OpenAI()
        self.functions = {}
        self.tools = [{"type": "file_search"}]
        self.model = model
        self.instructions = instructions
        self.vector_store = None  # To manage file search resources

        # Register functions before creating the assistant
        if functions:
            for function in functions:
                self.add_function(function)

        # Create the assistant after functions are added
        self.assistant = self.client.beta.assistants.create(
            instructions=self.instructions,
            model=self.model,
            tools=self.tools
        )
        print("[INFO] Assistant created with file search enabled.")

        # Optionally initialize vector store
        if vector_store_name:
            self.init_vector_store(vector_store_name)

        self.thread = None
        self.contact_info = None
        self.awaiting_contact_info = False
        self.escalation_reason = None

    def add_function(self, function):
        """Register a function as a tool for the assistant."""
        func_metadata = {
            "type": "function",
            "function": {
                "name": function.name,
                "description": function.description,
                "parameters": function.parameters,
                "strict": True
            }
        }

        # Add the function to the assistant's tools
        self.tools.append(func_metadata)
        # Register the function for execution
        self.functions[function.name] = function
        print(f"[REGISTERED FUNCTION] {function.name}")

    def init_vector_store(self, name):
        """Initialize the vector store for file search."""
        self.vector_store = self.client.beta.vector_stores.create(name=name)
        print(f"[INFO] Vector store '{name}' created with ID: {self.vector_store.id}")

    def upload_files_to_vector_store(self, file_paths):
        """
        Upload files to the vector store and poll for completion.
        :param file_paths: List of file paths to upload.
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")

        file_streams = [open(path, "rb") for path in file_paths]
        file_batch = self.client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=self.vector_store.id, files=file_streams
        )
        print(f"[INFO] Files uploaded to vector store. Batch status: {file_batch.status}")
        print(f"[INFO] File counts: {file_batch.file_counts}")

    def link_vector_store_to_assistant(self):
        """Link the vector store to the assistant."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")

        self.assistant = self.client.beta.assistants.update(
            assistant_id=self.assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [self.vector_store.id]}}
        )
        print(f"[INFO] Vector store linked to assistant. ID: {self.vector_store.id}")

    def start_conversation(self):
        """Create a new conversation thread."""
        self.thread = self.client.beta.threads.create()
        print(f"[INFO] Conversation thread started. Thread ID: {self.thread.id}")

    def send_message(self, content):
        """Send a message to the assistant and handle the response."""
        if not self.thread:
            self.start_conversation()
        print(f"[USER] {content}")

        if self.awaiting_contact_info:
            self.contact_info = content
            self.awaiting_contact_info = False
            self._execute_escalate_to_human()
        else:
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=content
            )
            self._process_run()

    def _execute_escalate_to_human(self):
        """Execute the escalate_to_human function with the collected contact info."""
        args = {
            'reason': self.escalation_reason,
            'thread_id': self.thread.id,
            'contact_info': self.contact_info
        }
        function = self.functions['escalate_to_human']
        result = function.execute(**args)
        print(f"[FUNCTION RESULT] {result}")
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="assistant",
            content=result
        )
        print(f"[ASSISTANT] {result}")

    def _process_run(self):
        """Initiate a run and handle required actions."""
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            temperature=0.0
        )
        print(f"[RUN STATUS] {run.status}")

        while run.status != 'completed':
            if run.status == 'requires_action':
                required_action = run.required_action
                if required_action.type == 'submit_tool_outputs':
                    tool_outputs = self._handle_function_calls(required_action.submit_tool_outputs.tool_calls)
                    run = self.client.beta.threads.runs.submit_tool_outputs_and_poll(
                        thread_id=self.thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                else:
                    break
            else:
                break

        print("[INFO] Run completed.")

    def _handle_function_calls(self, tool_calls):
        """Execute the functions requested by the assistant."""
        tool_outputs = []
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            if func_name in self.functions:
                args = json.loads(tool_call.function.arguments)
                if func_name == 'escalate_to_human':
                    args['thread_id'] = self.thread.id
                result = self.functions[func_name].execute(**args)
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": result
                })
            else:
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": f"Function {func_name} not found."
                })
        return tool_outputs

    def get_messages(self):
        """Retrieve conversation messages."""
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id
        )
        processed_messages = []
        for message in messages:
            text_content = ''
            for content_block in message.content:
                if content_block.type == 'text':
                    text_content += content_block.text.value
            processed_messages.append({
                'role': message.role,
                'content': text_content
            })
        return processed_messages

    def get_last_response(self):
        """Retrieve the assistant's last response."""
        messages = self.get_messages()
        for message in reversed(messages):
            if message['role'] == 'assistant':
                return message['content']
        return None