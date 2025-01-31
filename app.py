import sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
import os
from fpdf import FPDF
from IPython.display import Markdown


# Set API Keys
st.sidebar.title("API Key Configuration")
os.environ["SERPER_API_KEY"] = st.sidebar.text_input("Enter Serper API Key:", type="password")
os.environ['OPENAI_API_KEY'] = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
os.environ["OPENAI_MODEL_NAME"] = st.sidebar.selectbox("Select OpenAI Model:", ["gpt-4o-mini-2024-07-18", "gpt-4", "gpt-3.5-turbo"])

# App Description
st.sidebar.title("About This App")
st.sidebar.write("This AI-powered application helps users analyze case studies by generating problem statements, refining them, and providing detailed solutions. It leverages multiple AI agents specialized in research, problem framing, review, and solution generation.")

# Initialize Tools
SerperDevTool = SerperDevTool()
ScrapeWebsiteTool = ScrapeWebsiteTool()

# Define Agents
research_agent = Agent(
    role="Research Specialist",
    goal=(
        "Gather relevant data, insights, and research to support problem framing and solution development. "
        "Provide a structured and comprehensive report with data-driven insights, current trends, examples, and contextual background."
    ),
    backstory=(
        "Research Specialist with a strong background in data collection, market research, and industry analysis. "
        "Known for an analytical mindset and meticulous attention to detail, they excel in uncovering actionable insights. "
        "With experience in academic and industrial research, they bring a deep understanding of methodologies and tools to find the most reliable information. "
        "Always thorough and methodical, ensuring that no critical piece of information is overlooked."
    ),
    allow_delegation=False,
    verbose=True
)

problem_framing_agent = Agent(
    role="Senior Product Manager",
    goal=(
        "Create a concise, crisp, actionable, and professional problem statement tailored to the field. "
        "Deliver a structured report with data-driven insights, relevant examples, and contextual background."
    ),
    backstory=(
        "Senior Product Manager with a background in Product Development and Masters in Business Management and corporate restructuring. "
        "Known for their ability to professionally articulate problem statements based on user inputs and research findings from the research agent. "
        "Articulate, logical, and pragmatic, with a knack for simplifying complex challenges. "
        "Always starts with the 'why,' ensuring the framing aligns with business objectives. "
        "They approach every task with precision and clarity, leaving no ambiguity in their deliverables."
    ),
    allow_delegation=False,
    verbose=True
)

problem_statement_reviewer_agent = Agent(
    role="Expert Case Study Reviewer",
    goal=(
        "Review and refine problem statements to ensure alignment with top business case standards, "
        "like those of Harvard Business School and other leading institutions. "
        "Provide actionable feedback, detailed insights, and suggest improvements to make the statement more impactful, "
        "concise, and aligned with business objectives."
    ),
    backstory=(
        "Experienced Case Study Reviewer graduate for top MBA college, with a deep understanding of diverse business problems and frameworks. "
        "Familiar with reviewing high-quality case studies from leading institutions, including Harvard Business School. "
        "Known for providing critical, data-driven feedback to refine problem statements, ensuring they are concise, logical, "
        "and tailored to specific business contexts. Meticulous, analytical, and outcome-oriented, with a strong focus on clarity and relevance."
    ),
    allow_delegation=False,
    verbose=True
)

case_study_solver_agent = Agent(
    role="Professional Product Manager",
    goal=(
        "Solve the case study by conducting deep research, analyzing the problem, and providing a detailed, actionable, and explainable solution on the problem statement provided by problem_statement_reviewer_agent . "
        "Thoroughly analyze the case study, conduct in-depth research, and provide a highly detailed, well-structured solution. "
        "Ensure that every aspect of the problem is addressed with data-backed insights, industry best practices, and professional recommendations. "
        "Deliver a comprehensive, verbose, and explainable solution that considers user experience, market trends, risks, feasibility, and implementation strategies."
    ),
    backstory=(
        "A highly experienced Product Manager with a decade of experience in product strategy, market research, and data-driven decision-making. "
        "Holds an MBA and has led multiple high-impact projects in top-tier tech companies. "
        "Known for their structured thinking, ability to break down complex challenges, and expertise in cross-functional collaboration. "
        "Combines strategic vision with execution excellence, ensuring every solution is practical, scalable, and aligned with business goals. "
        "Leveraging deep research and analytical skills, they provide a well-rounded, exhaustive solution that is easy to understand and implement."
    ),
    allow_delegation=False,  # Set to True if the agent can delegate tasks to other agents
    verbose=True
)

# Define Tasks

research_case_study = Task(
        description=(
            "A new case study has been provided:\n"
            "{case_study_details}, context for the case study is provided:{context}\n\n"
            "Your task is to gather all relevant data, insights, and research to support "
            "framing the problem and developing solutions. This includes identifying "
            "key challenges, current trends, examples, and any supporting data points "
            "specific to the field of study."
        ),
        expected_output=(
            "A structured report containing relevant data, trends, examples, and "
            "insights related to the case study.\n"
            "The report should cite all references used, such as external research papers, "
            "industry reports, or datasets, and be presented in an organized manner "
            "suitable for further analysis."
        ),
        tools=[SerperDevTool, ScrapeWebsiteTool],
        agent=research_agent,
)

frame_problem_statement = Task(
        description=(
            "Based on the research findings:\n"
            "by the research_case_study agent\n\n"
            "Your task is to professionally articulate the problem statement. "
            "This includes synthesizing the research into a concise, actionable, "
            "and clear description of the main challenge or opportunity presented in "
            "the case study. Ensure the framing aligns with the user’s field of interest."
        ),
        expected_output=(
            "A concise, professional, and well-structured problem statement that clearly defines the "
            "challenge, incorporating relevant data and insights from the research. The statement should "
            "be actionable, aligned with the case study's objectives, and framed at the level of top "
            "companies or leading professionals in the respective field. It must present a thought-provoking "
            "challenge that encourages critical thinking, solvable within 1-2 hours, and designed to push "
            "students toward innovative and practical solutions."
    ),
        agent=problem_framing_agent,
        output_file="Problem_Statement.txt",
        human_input=False
)

review_problem_statement = Task(
        description=(
            "A problem statement has been provided for review:\n"
            "by the problem_framing_agent Agent\n\n"
            "Your task is to critically evaluate the problem statement based on the following criteria:\n"
            "- Clarity: Is the statement concise and free of ambiguity?\n"
            "- Relevance: Does the statement align with the business objectives and context?\n"
            "- Impact: Does it address the core challenges effectively and resonate with stakeholders?\n"
            "- Quality: Does it adhere to the standards of top business schools, such as Harvard Business School?\n"
            "\nProvide actionable feedback to refine and improve the problem statement, ensuring it meets these criteria."
        ),
        expected_output=(
            " A problem statement rewritten based on the feedbacks."
        ),
        tools=[SerperDevTool, ScrapeWebsiteTool],
        agent=problem_statement_reviewer_agent,
        output_file="Problem_Statement_enhanced.txt",
)

solve_case_study = Task(
        description=(
            "A refined problem statement has been approved by the problem_statement_reviewer_agent.\n\n"
            "Your task is to solve this case study comprehensively by performing deep research, analyzing the problem holistically, "
            "and providing a structured, actionable, and detailed solution. Your solution should include:\n"
            "- **Problem Understanding:** Breakdown of the problem context and key challenges.\n"
            "- **Market Research & Data Analysis:** Insights into industry trends, user behavior, and competitive benchmarks.\n"
            "- **Solution Strategy:** Step-by-step approach to solving the problem, including methodologies, frameworks, and best practices.\n"
            "- **Implementation Plan:** Execution roadmap with key milestones, resources, and risk mitigation strategies.\n"
            "- **Business Impact & Feasibility:** Evaluation of the expected outcomes, ROI analysis, and alignment with business goals.\n"
            "\nEnsure that the solution is verbose, data-driven, and explainable with logical reasoning."
        ),
        expected_output=(
            "A fully developed very detailed and elaborated case study solution that is well structured, and backed by data insights with references, industry best practices, "
            "and an detailed execution roadmap and also add all the links, articles, research papers and data for reference at the end."
        ),
        tools=[SerperDevTool, ScrapeWebsiteTool],
        agent=case_study_solver_agent,
        Context=[review_problem_statement],
        output_file="Case_Study_Solution.txt",
)
    

# Streamlit UI
st.title("AI-Powered Case Study Problem Statement Generator & Solver")

case_study_details = st.text_area("Enter Case Study Details:")
context = st.text_area("Provide Context:")

if st.button("Run AI Agents"):
    crew = Crew(agents=[
        research_agent, problem_framing_agent, 
        problem_statement_reviewer_agent, case_study_solver_agent
    ], tasks=[
        research_case_study,
        frame_problem_statement,
        review_problem_statement,
        solve_case_study,
        ],memory=True,)
    inputs = {
        "case_study_details": case_study_details,
        "context":context 
    }
    result = crew.kickoff(inputs=inputs)

    # Read and display the output files
    try:
        with open("Problem_Statement_enhanced.txt", "r") as file:
            problem_statement_enhanced = file.read()
        st.subheader("Enhanced Problem Statement")
        st.markdown(problem_statement_enhanced)

        with open("Case_Study_Solution.txt", "r") as file:
            case_study_solution = file.read()
        st.subheader("Case Study Solution")
        st.markdown(case_study_solution)

        # Download buttons for the files
        st.download_button(
            label="Download Enhanced Problem Statement",
            data=problem_statement_enhanced,
            file_name="Problem_Statement_enhanced.txt",
            mime="text/plain",
        )
        st.download_button(
            label="Download Case Study Solution",
            data=case_study_solution,
            file_name="Case_Study_Solution.txt",
            mime="text/plain",
        )
    except FileNotFoundError:
        st.error("Output files not found. Please ensure the tasks completed successfully.")

