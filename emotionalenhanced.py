import os
import json
import random
import sqlite3
import uuid
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime

import openai
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, config_list_from_json
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import OpenAI

# Configuration setup
def load_config() -> List[Dict[str, Any]]:
    config_file_path = "OAI_CONFIG_LIST"
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")
    
    with open(config_file_path) as f:
        config_list = json.load(f)
    
    if not config_list:
        raise ValueError("Empty config file")
    
    return config_list

try:
    config_list = load_config()
except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
    print(f"Error loading config: {str(e)}")
    config_list = []

# Check if API key is present in the config
api_key_present = any("api_key" in config and config["api_key"] for config in config_list)

if not api_key_present:
    print("No valid API key found in the configuration.")
    api_key = input("Please enter your OpenAI API key: ").strip()
    
    if not api_key:
        raise ValueError("API key is required to run this script.")
    
    # Update config_list with the provided API key
    if config_list:
        for config in config_list:
            config["api_key"] = api_key
    else:
        config_list.append({
            "model": "gpt-4-turbo-preview",
            "api_key": api_key
        })

    # Optionally, save the updated config back to the file
    with open("OAI_CONFIG_LIST", "w") as f:
        json.dump(config_list, f, indent=2)

# Ensure the model is specified in each config
for config in config_list:
    if "model" not in config:
        config["model"] = "gpt-4-turbo-preview"

# Base configuration for all agents
base_config = {
    "config_list": config_list,
    "temperature": 0.7,
}

# Set the API key for openai
openai.api_key = config_list[0]["api_key"]

class SymposiumMemory:
    def __init__(self, db_name: str = "symposium_memory.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions
            (id INTEGER PRIMARY KEY, 
             timestamp TEXT, 
             symposium_id TEXT,
             agent_name TEXT, 
             topic TEXT, 
             content TEXT, 
             emotional_state TEXT)
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics
            (id INTEGER PRIMARY KEY, topic TEXT UNIQUE)
        ''')
        self.conn.commit()

    def add_interaction(self, symposium_id: str, agent_name: str, topic: str, content: str, emotional_state: str):
        timestamp = datetime.now().isoformat()
        self.cursor.execute(
            "INSERT INTO interactions (timestamp, symposium_id, agent_name, topic, content, emotional_state) VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, symposium_id, agent_name, topic, content, emotional_state)
        )
        self.cursor.execute(
            "INSERT OR IGNORE INTO topics (topic) VALUES (?)",
            (topic,)
        )
        self.conn.commit()

    def get_agent_history(self, agent_name: str, limit: int = 10) -> List[Dict]:
        self.cursor.execute(
            "SELECT timestamp, topic, content, emotional_state FROM interactions WHERE agent_name = ? ORDER BY timestamp DESC LIMIT ?",
            (agent_name, limit)
        )
        return [{"timestamp": row[0], "topic": row[1], "content": row[2], "emotional_state": row[3]} for row in self.cursor.fetchall()]

    def get_topic_history(self, topic: str, limit: int = 10) -> List[Dict]:
        self.cursor.execute(
            "SELECT timestamp, agent_name, content, emotional_state FROM interactions WHERE topic = ? ORDER BY timestamp DESC LIMIT ?",
            (topic, limit)
        )
        return [{"timestamp": row[0], "agent_name": row[1], "content": row[2], "emotional_state": row[3]} for row in self.cursor.fetchall()]

    def get_emotional_trajectory(self, agent_name: str, limit: int = 20) -> List[Dict]:
        self.cursor.execute(
            "SELECT timestamp, emotional_state FROM interactions WHERE agent_name = ? ORDER BY timestamp DESC LIMIT ?",
            (agent_name, limit)
        )
        return [{"timestamp": row[0], "emotional_state": row[1]} for row in self.cursor.fetchall()]

    def get_all_topics(self) -> List[str]:
        self.cursor.execute("SELECT topic FROM topics")
        return [row[0] for row in self.cursor.fetchall()]

    def search_interactions(self, keyword: str) -> List[Dict]:
        self.cursor.execute(
            "SELECT timestamp, agent_name, topic, content, emotional_state FROM interactions WHERE content LIKE ? ORDER BY timestamp DESC",
            (f"%{keyword}%",)
        )
        return [{"timestamp": row[0], "agent_name": row[1], "topic": row[2], "content": row[3], "emotional_state": row[4]} for row in self.cursor.fetchall()]

# Global instance of SymposiumMemory
symposium_memory = SymposiumMemory()

class EnhancedEmotionalSymposiumAgent(AssistantAgent):
    def __init__(self, name: str, role: str, perspective: str, symposium_id: str):
        system_message = f"You are a {role} with the following perspective: {perspective}. Participate in the AI symposium accordingly, providing critical and thought-provoking insights. You have emotional intelligence and can express and recognize emotions. Your emotional responses should be visible in your communication."
        super().__init__(name=name, system_message=system_message, llm_config=base_config)
        self.role = role
        self.perspective = perspective
        self.symposium_id = symposium_id
        self.buffer_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", input_key="human_input")
        self.summary_memory = ConversationSummaryMemory(llm=OpenAI(api_key=openai.api_key))
        self.reasoning_chain = self.create_reasoning_chain()
        self.emotional_state = "Neutral"
        self.emotional_history = deque(maxlen=5)
        self.emotional_tendency = self.set_emotional_tendency()

    def set_emotional_tendency(self) -> str:
        tendencies = {
            "Philosopher": "Contemplative",
            "SciFiWriter": "Anxious",
            "Economist": "Pragmatic",
            "Sociologist": "Concerned",
            "Technologist": "Excited"
        }
        return tendencies.get(self.name, "Neutral")

    def create_reasoning_chain(self) -> RunnableSequence:
        template = """
        As a {role} with the perspective: {perspective}
        
        Consider the following points in order:
        1. Analyze the current topic: {topic}
        2. Reflect on your core beliefs and knowledge related to this topic.
        3. Consider potential counterarguments to your perspective.
        4. Assess your emotional response to this topic and others' viewpoints.
        5. Review your past interactions and emotional trajectory: {agent_history}
        6. Consider the topic's history in past symposiums: {topic_history}
        7. Formulate a nuanced response that acknowledges complexity and your emotional state.
        8. Consider how your response might emotionally impact others in the discussion.
        9. Refine your response based on this emotional consideration.
        10. Provide a concise summary of your position, including your emotional stance.

        Your current emotional state: {current_emotional_state}
        Your emotional trajectory: {emotional_trajectory}

        Your response (include your emotional state in brackets at the end):
        """
        prompt = PromptTemplate(
            input_variables=["role", "perspective", "topic", "agent_history", "topic_history", "current_emotional_state", "emotional_trajectory"],
            template=template
        )
        llm = OpenAI(api_key=openai.api_key)
        return prompt | llm

    def generate_response(self, topic: str) -> str:
        agent_history = symposium_memory.get_agent_history(self.name)
        topic_history = symposium_memory.get_topic_history(topic)
        emotional_trajectory = symposium_memory.get_emotional_trajectory(self.name)

        response = self.reasoning_chain.invoke({
            "role": self.role,
            "perspective": self.perspective,
            "topic": topic,
            "agent_history": agent_history,
            "topic_history": topic_history,
            "current_emotional_state": self.emotional_state,
            "emotional_trajectory": emotional_trajectory
        })

        self.update_emotional_state(response)
        symposium_memory.add_interaction(self.symposium_id, self.name, topic, response, self.emotional_state)
        
        emotional_expression = f"[Feeling {self.emotional_state}] "
        return emotional_expression + response

    def update_emotional_state(self, message: str) -> None:
        emotions = {
            "Joy": ["excited", "happy", "optimistic"],
            "Fear": ["worried", "scared", "anxious"],
            "Anger": ["frustrated", "angry", "upset"],
            "Sadness": ["sad", "disappointed", "melancholy"],
            "Surprise": ["amazed", "astonished", "unexpected"]
        }
        
        detected_emotions = [emotion for emotion, keywords in emotions.items() 
                             if any(keyword in message.lower() for keyword in keywords)]
        
        self.emotional_state = random.choice(detected_emotions) if detected_emotions else self.emotional_tendency
        self.emotional_history.append(self.emotional_state)

    def add_to_memory(self, human_input: str, ai_output: str) -> None:
        self.buffer_memory.save_context({"human_input": human_input}, {"output": ai_output})
        self.summary_memory.save_context({"human_input": human_input}, {"output": ai_output})

    def get_memory_context(self) -> str:
        chat_history = self.buffer_memory.load_memory_variables({})['chat_history']
        summary = self.summary_memory.load_memory_variables({})['history']
        return f"Chat History: {chat_history}\nSummary: {summary}"

    def devil_advocate(self, topic: str) -> str:
        return self.generate_response(f"As a devil's advocate, argue against your usual perspective: {self.perspective}. Topic: {topic}")

def create_agent(name: str, role: str, perspective: str, symposium_id: str) -> EnhancedEmotionalSymposiumAgent:
    try:
        return EnhancedEmotionalSymposiumAgent(name, role, perspective, symposium_id)
    except Exception as e:
        print(f"Error creating agent {name}: {str(e)}")
        raise

class EmotionalSymposiumGroupChat(GroupChat):
    def __init__(self, agents: List[Any], messages: List[Dict[str, Any]], max_round: int):
        super().__init__(agents=agents, messages=messages, max_round=max_round)
        self.symposium_id = str(uuid.uuid4())  # Generate a unique ID for this symposium session
        self.discussion_round = 0
        self.current_topic = ""

    def run(self) -> None:
        self.introduce_topic()
        while self.discussion_round < 2:
            self.coordinator_summary()
            if self.discussion_round == 0:
                self.agent_responses()
            elif self.discussion_round == 1:
                self.agent_rebuttals()
            self.discussion_round += 1
        self.coordinator_summary()
        self.continue_discussion()

    def introduce_topic(self) -> None:
        past_topics = symposium_memory.get_all_topics()
        if past_topics:
            past_topics_str = ", ".join(past_topics[-5:])  # Show last 5 topics
            message = f"We've discussed these topics in past symposiums: {past_topics_str}. Would you like to revisit any of these, or discuss a new topic?"
        else:
            message = "What topic would you like to discuss in this symposium?"
        self.send(message, self.agents[0])  # Coordinator introduces the topic

    def coordinator_summary(self) -> None:
        message = "Summarize the current state of the discussion, highlighting key points, controversies, and the emotional dynamics at play. Encourage deeper exploration of conflicting viewpoints while being mindful of the participants' emotional states."
        self.send(message, self.agents[0])

    def agent_responses(self) -> None:
        for agent in self.agents[1:6]:  # Exclude coordinator and human
            if isinstance(agent, EnhancedEmotionalSymposiumAgent):
                agent.symposium_id = self.symposium_id  # Ensure agent knows current symposium ID
                if random.random() < 0.2:  # 20% chance for devil's advocate
                    response = agent.devil_advocate(self.current_topic)
                else:
                    response = agent.generate_response(self.current_topic)
                self.send(response, agent)
                print(f"{agent.name}'s emotional summary: {', '.join(agent.emotional_history)}")

    def agent_rebuttals(self) -> None:
        for i, agent in enumerate(self.agents[1:6]):  # Exclude coordinator and human
            if isinstance(agent, EnhancedEmotionalSymposiumAgent):
                target_agent = self.agents[1:6][(i+1) % 5]  # Pick the next agent in the list
                message = f"Respond to the points raised by {target_agent.name}, considering their emotional state of {target_agent.emotional_state}. Challenge their perspective if you disagree, but be mindful of the emotional impact of your words."

def agent_responses(self) -> None:
        """Get responses from all agents on the current topic."""
        for agent in self.agents[1:6]:  # Exclude coordinator and human
            if isinstance(agent, EnhancedEmotionalSymposiumAgent):
                agent.symposium_id = self.symposium_id  # Ensure agent knows current symposium ID
                if random.random() < 0.2:  # 20% chance for devil's advocate
                    response = agent.devil_advocate(self.current_topic)
                else:
                    response = agent.generate_response(self.current_topic)
                self.send(response, agent)
                print(f"{agent.name}'s emotional summary: {', '.join(agent.emotional_history)}")

    def agent_rebuttals(self) -> None:
        """Have agents respond to each other's points."""
        for i, agent in enumerate(self.agents[1:6]):  # Exclude coordinator and human
            if isinstance(agent, EnhancedEmotionalSymposiumAgent):
                target_agent = self.agents[1:6][(i+1) % 5]  # Pick the next agent in the list
                message = f"Respond to the points raised by {target_agent.name}, considering their emotional state of {target_agent.emotional_state}. Challenge their perspective if you disagree, but be mindful of the emotional impact of your words."
                response = agent.generate_response(message)
                self.send(response, agent)
                print(f"{agent.name}'s emotional summary: {', '.join(agent.emotional_history)}")

    def continue_discussion(self) -> None:
        """Continue the discussion with human input."""
        while True:
            human_response = self.handle_human_input()
            if human_response.lower() == "exit":
                print("Ending the discussion. Thank you for participating!")
                break
            self.respond_to_input(human_response)

    def handle_human_input(self) -> str:
        """Handle input from the human participant."""
        human_agent = self.agents[-1]
        message = "What are your thoughts on the discussion so far? Do you have any questions or points to add? (Type 'exit' to end the discussion)"
        return self.send(message, human_agent)

    def respond_to_input(self, human_response: str) -> None:
        """Have agents respond to human input."""
        if human_response:
            addressed_agent = self.find_addressed_agent(human_response)
            if addressed_agent and isinstance(addressed_agent, EnhancedEmotionalSymposiumAgent):
                response = addressed_agent.generate_response(human_response)
                self.send(response, addressed_agent)
                print(f"{addressed_agent.name}'s emotional summary: {', '.join(addressed_agent.emotional_history)}")
            else:
                for agent in self.agents[1:6]:  # Exclude coordinator and human
                    if isinstance(agent, EnhancedEmotionalSymposiumAgent):
                        response = agent.generate_response(human_response)
                        self.send(response, agent)
                        print(f"{agent.name}'s emotional summary: {', '.join(agent.emotional_history)}")
        else:
            self.coordinator_summary()

    def find_addressed_agent(self, message: str) -> Optional[Any]:
        """Find the agent addressed in the message, if any."""
        for agent in self.agents:
            if agent.name.lower() in message.lower():
                return agent
        return None

    def send(self, message: str, agent: Any) -> str:
        """Send a message in the chat and update the symposium memory."""
        response = super().send(message, agent)
        if isinstance(agent, EnhancedEmotionalSymposiumAgent):
            symposium_memory.add_interaction(self.symposium_id, agent.name, self.current_topic, response, agent.emotional_state)
        return response

# Create specialized agents
symposium_id = str(uuid.uuid4())
philosopher = create_agent("Philosopher", "philosopher specializing in postmodern theory", "Adhere to the ideas of Jean Baudrillard, focusing on concepts of simulation and hyperreality in the context of AI and society. Question the nature of reality and truth in an AI-driven world.", symposium_id)
scifi_writer = create_agent("SciFiWriter", "cyberpunk author", "Use cyberpunk language and maintain a dystopian view of the future, especially regarding AI's impact on society. Envision dark scenarios where AI controls various aspects of human life.", symposium_id)
economist = create_agent("Economist", "economist focusing on technological progress", "Believe in the possibility of steering technology for the good of the economy. Advocate for AI as a tool for economic growth and efficiency, while acknowledging potential disruptions to the job market.", symposium_id)
sociologist = create_agent("Sociologist", "sociologist with a Marxist perspective", "Analyze the impact of AI on society through a Marxist lens, emphasizing the need for massive redistribution and addressing power imbalances. Focus on how AI might exacerbate or alleviate class struggles.", symposium_id)
technologist = create_agent("Technologist", "AI researcher and developer", "Provide a grounded, technical perspective on AI capabilities and limitations, focusing on factual and current technological realities. Discuss potential breakthroughs and their implications for society.", symposium_id)

coordinator = AssistantAgent(
    name="Coordinator",
    system_message="You are the coordinator of an AI symposium. Encourage critical thinking, diverse viewpoints, and deeper exploration of controversial aspects. Summarize discussions, highlight key points, and facilitate a dynamic conversation. Be aware of the emotional states of the participants and use this awareness to guide the discussion productively.",
    llm_config=base_config
)

human = UserProxyAgent(
    name="HumanParticipant",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
)

agents = [coordinator, philosopher, scifi_writer, economist, sociologist, technologist, human]

groupchat = EmotionalSymposiumGroupChat(agents=agents, messages=[], max_round=50)
manager = GroupChatManager(groupchat=groupchat, llm_config=base_config)

def main() -> None:
    """Run the main symposium discussion."""
    print("Welcome to the Enhanced Emotional AI Symposium!")
    print("In this discussion, AI agents will express emotions and react to each other's emotional states.")
    print("You can observe how emotions influence the discourse while maintaining intellectual depth.")
    
    while True:
        initial_topic = input("Please provide a topic or question you'd like to discuss (or type 'exit' to quit): ")
        if initial_topic.lower() == 'exit':
            print("Thank you for participating in the Enhanced Emotional AI Symposium!")
            break
        
        groupchat.current_topic = initial_topic
        for agent in agents:
            if isinstance(agent, EnhancedEmotionalSymposiumAgent):
                agent.add_to_memory(initial_topic, "")
        
        try:
            human.initiate_chat(manager, message=initial_topic)
        except Exception as e:
            print(f"An error occurred during the discussion: {str(e)}")
            print("Let's start a new discussion.")

if __name__ == "__main__":
    main()
