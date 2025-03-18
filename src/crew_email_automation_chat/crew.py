import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from src.crew_email_automation_chat.tools.custom_tool import (
    FetchEmailTool,
    ExtractFreshBodyTool,
    FetchThreadBodyTool,
    ExtractPatientDetailsTool,
    ValidateDataTool,
    SendReplyEmailTool,
    SendConfirmationEmailTool,
    SendReminderAndEscalationTool,
    PatientDataUploadTool
)

load_dotenv()

# llm = LLM(
#     model="ollama/llama-3.1-8b-instant",
#     api_key="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",
#     temperature=0.3
#     )

llm = LLM(
    model=os.getenv('MODEL'),
    temperature=0.7,
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPEN_ROUTER_KEY')
)

# llm = LLM(
#     model="ollama/gemma2:latest",
#     api_base="http://localhost:11434"
#    )


@CrewBase
class CrewEmailAutomationChatCrew():
    """CrewEmailAutomationChat crew"""

    @agent
    def email_fetcher(self) -> Agent:
        return Agent(
            config=self.agents_config['email_fetcher'],
            tools=[FetchEmailTool()],
            llm=llm
        )

    @agent
    def email_content_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['email_content_extractor'],
            tools=[ExtractFreshBodyTool(), FetchThreadBodyTool()],
            llm=llm
        )

    @agent
    def patient_data_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['patient_data_extractor'],
            tools=[ExtractPatientDetailsTool()],
            llm=llm
        )

    @agent
    def data_validation(self) -> Agent:
        return Agent(
            config=self.agents_config['data_validation'],
            tools=[ValidateDataTool()],
            llm=llm
        )

    @agent
    def email_sender(self) -> Agent:
        return Agent(
            config=self.agents_config['email_sender'],
            tools=[SendReplyEmailTool(),SendConfirmationEmailTool()],
            llm=llm
        )
    
    @agent
    def data_storage_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['data_storage_agent'],
            tools=[PatientDataUploadTool()],
            llm=llm
        )

    @agent
    def reminder_email_sender(self) -> Agent:
        return Agent(
            config=self.agents_config['reminder_email_sender'],
            tools=[SendReminderAndEscalationTool()],
            llm=llm
        )

    @task
    def fetch_unread_email(self) -> Task:
        return Task(
            config=self.tasks_config['fetch_unread_email'],
            tools=[],
        )

    @task
    def extract_email_body(self) -> Task:
        return Task(
            config=self.tasks_config['extract_email_body'],
            tools=[],
        )

    @task
    def extract_patient_data(self) -> Task:
        return Task(
            config=self.tasks_config['extract_patient_data'],
            tools=[],
        )

    @task
    def validate_extracted_data(self) -> Task:
        return Task(
            config=self.tasks_config['validate_extracted_data'],
            tools=[],
        )

    @task
    def send_follow_up_email(self) -> Task:
        return Task(
            config=self.tasks_config['send_follow_up_email'],
            tools=[],
        )
    
    @task
    def store_patient_data(self) -> Task:
        return Task(
            config=self.tasks_config['store_patient_data'],
            tools=[],
        )

    @task
    def send_reminder_email(self) -> Task:
        return Task(
            config=self.tasks_config['send_reminder_email'],
            tools=[],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewEmailAutomationChat crew"""
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )
