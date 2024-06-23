import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import openai
import time
from collections import deque
import random

# OpenAI API configuration
openai.api_key = "your_openai_api_key_here"

# Configuration for all agents (using GPT-3.5-Turbo)
base_config = {"config_list": [{"model": "gpt-3.5-turbo", "api_key": openai.api_key}], "temperature": 0.7}

class SymposiumAgent(AssistantAgent):
    def __init__(self, name, role, perspective, llm_config):
        system_message = f"You are a {role} with the following perspective: {perspective}. Participate in the AI symposium accordingly, providing critical and thought-provoking insights. Previous context: {{}}"
        super().__init__(name=name, system_message=system_message, llm_config=llm_config)
        self.memory = deque(maxlen=10)
        self.perspective = perspective

    def add_to_memory(self, message):
        self.memory.append(message)

    def get_memory(self):
        return list(self.memory)

    def update_system_message(self, memory_context):
        updated_message = self.system_message.format(memory_context)
        self._system_message = updated_message

    def devil_advocate(self):
        return f"As a devil's advocate, argue against your usual perspective: {self.perspective}"

# Create specialized agents with more defined perspectives
philosopher = SymposiumAgent(
    "Philosopher", 
    "philosopher specializing in postmodern theory",
    "Adhere to the ideas of Jean Baudrillard, focusing on concepts of simulation and hyperreality in the context of AI and society. Question the nature of reality and truth in an AI-driven world.",
    base_config
)

scifi_writer = SymposiumAgent(
    "SciFiWriter",
    "cyberpunk author",
    "Use cyberpunk language and maintain a dystopian view of the future, especially regarding AI's impact on society. Envision dark scenarios where AI controls various aspects of human life.",
    base_config
)

economist = SymposiumAgent(
    "Economist",
    "economist focusing on technological progress",
    "Believe in the possibility of steering technology for the good of the economy. Advocate for AI as a tool for economic growth and efficiency, while acknowledging potential disruptions to the job market.",
    base_config
)

sociologist = SymposiumAgent(
    "Sociologist",
    "sociologist with a Marxist perspective",
    "Analyze the impact of AI on society through a Marxist lens, emphasizing the need for massive redistribution and addressing power imbalances. Focus on how AI might exacerbate or alleviate class struggles.",
    base_config
)

technologist = SymposiumAgent(
    "Technologist",
    "AI researcher and developer",
    "Provide a grounded, technical perspective on AI capabilities and limitations, focusing on factual and current technological realities. Discuss potential breakthroughs and their implications for society.",
    base_config
)

class Coordinator(AssistantAgent):
    def __init__(self, name, llm_config):
        system_message = "You are the coordinator of an AI symposium. Encourage critical thinking, diverse viewpoints, and deeper exploration of controversial aspects. Summarize discussions, highlight key points, and facilitate a dynamic conversation. Occasionally prompt agents to respond directly to each other or play devil's advocate."
        super().__init__(name=name, system_message=system_message, llm_config=llm_config)

coordinator = Coordinator("Coordinator", base_config)

class InteractiveHuman(UserProxyAgent):
    def get_human_input(self, prompt="HumanParticipant: "):
        return input(prompt)

human = InteractiveHuman(
    name="HumanParticipant",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
)

agents = [coordinator, philosopher, scifi_writer, economist, sociologist, technologist, human]

class SymposiumGroupChat(GroupChat):
    def __init__(self, agents, messages, max_round):
        super().__init__(agents, messages, max_round)
        self.discussion_round = 0

    def run(self):
        while self.discussion_round < 2:
            self.coordinator_summary()
            if self.discussion_round == 0:
                self.agent_responses()
            elif self.discussion_round == 1:
                self.agent_rebuttals()
            self.discussion_round += 1
        self.coordinator_summary()
        self.continue_discussion()

    def coordinator_summary(self):
        message = "Summarize the current state of the discussion, highlighting key points and controversies. Encourage deeper exploration of conflicting viewpoints."
        self.send(message, self.agents[0])

    def agent_responses(self):
        for agent in self.agents[1:6]:  # Exclude coordinator and human
            if random.random() < 0.2:  # 20% chance for devil's advocate
                message = agent.devil_advocate()
            else:
                message = f"Please provide your perspective on the current discussion topic. Be critical and thought-provoking."
            self.send(message, agent)

    def agent_rebuttals(self):
        for i, agent in enumerate(self.agents[1:6]):  # Exclude coordinator and human
            target_agent = self.agents[1:6][(i+1) % 5]  # Pick the next agent in the list
            message = f"Respond to the points raised by {target_agent.name}, challenging their perspective if you disagree."
            self.send(message, agent)

    def continue_discussion(self):
        while True:
            human_response = self.handle_human_input()
            if human_response.lower() == "exit":
                print("Ending the discussion. Thank you for participating!")
                break
            self.respond_to_input(human_response)

    def handle_human_input(self):
        human_agent = self.agents[-1]
        message = "What are your thoughts on the discussion so far? Do you have any questions or points to add? (Type 'exit' to end the discussion)"
        return self.send(message, human_agent)

    def respond_to_input(self, human_response):
        if human_response:
            # Check if a specific agent is addressed
            addressed_agent = self.find_addressed_agent(human_response)
            if addressed_agent:
                response_message = f"Please respond to the human's input: {human_response}"
                self.send(response_message, addressed_agent)
            else:
                # If no specific agent is addressed, let all AI agents respond
                for agent in self.agents[1:6]:  # Exclude coordinator and human
                    response_message = f"Please respond to the human's input: {human_response}"
                    self.send(response_message, agent)
        else:
            # If no human input, have the coordinator continue the discussion
            self.coordinator_summary()

    def find_addressed_agent(self, message):
        for agent in self.agents:
            if agent.name.lower() in message.lower():
                return agent
        return None

groupchat = SymposiumGroupChat(agents=agents, messages=[], max_round=50)
manager = GroupChatManager(groupchat=groupchat, llm_config=base_config)

def update_memories(message):
    for agent in agents:
        if isinstance(agent, SymposiumAgent):
            agent.add_to_memory(message)
            memory_context = "\n".join(agent.get_memory()[-5:])
            agent.update_system_message(memory_context)

def main():
    print("Welcome to the Enhanced AI Symposium!")
    initial_topic = input("Please provide a topic or question you'd like to discuss: ")
    update_memories(initial_topic)
    human.initiate_chat(manager, message=initial_topic)

if __name__ == "__main__":
    main()
