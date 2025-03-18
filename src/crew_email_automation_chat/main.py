#!/usr/bin/env python
import sys
import time

from crew_email_automation_chat.crew import CrewEmailAutomationChatCrew

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew in an infinite loop.
    """
    # inputs = {
    #     'email_body': 'sample_value',
    #     'patient_data_json': 'sample_value',
    #     'required_fields': 'sample_value',
    #     'validation_result': 'sample_value',
    #     'smtp_connection': 'sample_value',
    #     'mail_type': 'sample_value',
    #     'missing_fields': 'sample_value',
    #     'email_sent_confirmation': 'sample_value'
    # }
    CrewEmailAutomationChatCrew().crew().kickoff()

    # while True:
    #     print("🚀 Running CrewEmailAutomationChatCrew...")

    #     # Execute the workflow
    #     CrewEmailAutomationChatCrew().crew().kickoff(inputs=inputs)

    #     print("✅ Crew executed. Waiting before next run...")

    #     # Prevent excessive CPU usage (adjust the delay as needed)
    #     time.sleep(5)  


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        # 'imap_config': 'sample_value',
        # 'smtp_config': 'sample_value',
        # 'imap_connection': 'sample_value',
        'email_body': 'sample_value',
        'patient_data_json': 'sample_value',
        'required_fields': 'sample_value',
        'validation_result': 'sample_value',
        'smtp_connection': 'sample_value',
        'mail_type': 'sample_value',
        'missing_fields': 'sample_value',
        'email_sent_confirmation': 'sample_value'
    }
    try:
        CrewEmailAutomationChatCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        CrewEmailAutomationChatCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        # 'imap_config': 'sample_value',
        # 'smtp_config': 'sample_value',
        # 'imap_connection': 'sample_value',
        'email_body': 'sample_value',
        'patient_data_json': 'sample_value',
        'required_fields': 'sample_value',
        'validation_result': 'sample_value',
        'smtp_connection': 'sample_value',
        'mail_type': 'sample_value',
        'missing_fields': 'sample_value',
        'email_sent_confirmation': 'sample_value'
    }
    try:
        CrewEmailAutomationChatCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
