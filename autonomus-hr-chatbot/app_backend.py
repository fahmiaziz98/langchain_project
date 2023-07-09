# load core modules
import pinecone
import pandas as pd
from io import StringIO
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain

from settings import SETTINGS

# initialize pinecone client and connect to pinecone index
pinecone.init(      
	api_key=SETTINGS["PINECONE_API_KEY"],      
	environment=SETTINGS["PINECONE_ENV_NAME"]      
)      
index = pinecone.Index('tk-policy')

# initialize embeddings object; for use with user query/input
embed = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=SETTINGS["OPENAI_API_KEY"])

# initialize langchain vectorstore(pinecone) object
text_field = 'text' # key of dict that stores the text metadata in the index
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

llm = ChatOpenAI(    
    model_name="gpt-3.5-turbo", 
    temperature=0.0,
    openai_api_key=SETTINGS["OPENAI_API_KEY"]
    )

# initialize vectorstore retriever object
timekeeping_policy = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

df = pd.read_csv("data/employee_data_new.csv") # load employee_data.csv as dataframe
python = PythonAstREPLTool(locals={"df": df}) # set access of python_repl tool to the dataframe

# create calculator tool
calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

# create variables for f strings embedded in the prompts
user = 'Alexander Verdad' # set user
df_columns = df.columns.to_list() # print column names of df

# prep the (tk policy) vectordb retriever, the python_repl(with df access) and langchain calculator as tools for the agent
tools = [
    Tool(
        name = "Timekeeping Policies",
        func=timekeeping_policy.run,
        description="""
        Useful for when you need to answer questions about employee timekeeping policies.

        <user>: What is the policy on unused vacation leave?
        <assistant>: I need to check the timekeeping policies to answer this question.
        <assistant>: Action: Timekeeping Policies
        <assistant>: Action Input: Vacation Leave Policy - Unused Leave
        ...
        """
    ),
    Tool(
        name = "Employee Data",
        func=python.run,
        description = f"""
        Useful for when you need to answer questions about employee data stored in pandas dataframe 'df'. 
        Run python pandas operations on 'df' to help you get the right answer.
        'df' has the following columns: {df_columns}
        
        <user>: How many Sick Leave do I have left?
        <assistant>: df[df['name'] == '{user}']['sick_leave']
        <assistant>: You have n sick leaves left.              
        """
    ),
    Tool(
        name = "Calculator",
        func=calculator.run,
        description = f"""
        Useful when you need to do math operations or arithmetic.

        <user>: How much will I be paid if I encash my unused VLs?
        <assistant>: df[df['name'] == '{user}'][['basic_pay_in_php', 'vacation_leave']]
        <assistant>: You will be paid Php n if you encash your unused VLs.'
        """
    )
]

# change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
agent_kwargs = {'prefix': f'You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. You have access to the following tools:'}


# initialize the LLM agent
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True, 
                         agent_kwargs=agent_kwargs,
                         handle_parsing_errors=True
                         )
# define q and a function for frontend
def get_response(user_input):
    response = agent.run(user_input)
    return response



# <user>: How much will I be paid if I encash my unused VLs?
# <assistant>: df[df['name'] == '{user}'][['basic_pay_in_php', 'vacation_leave']]
# <assistant>: You will be paid Php n if you encash your unused VLs.