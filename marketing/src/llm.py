from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_cohere import ChatCohere
from langchain.vectorstores import VectorStore

# Set Env
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

def generate_text(query: str, db: VectorStore) -> str:

    prompt_template = """
        # CONTEXT # I want to advertise my company's new product. My company name is Alpha and the product is called {text}
        # PURPOSE # Make a Facebook or Instagram post for me, which aims to get people to click on the product link to buy it.
        # STYLE # Follow the writing style of successful companies that advertise similar products, such as Dyson.
        # TONE # Persuasive
        # AUDIENCE # My company's audience profile on social media is typically the younger generation. Customize your posts to target what this audience is typically looking for in our products
        # RESPONSE # Social media posts such as Facebook or Instagram, are concise yet high-impact.
        
        ###
        EXAMPLE

        Embrace Effortless Elegance with Alpha Beta — The Ultra-Fast Hairdryer for the Wise Generation.
        Rediscover the joy of simple, effective hair care with Alpha Beta. Our latest innovation is more than just a hairdryer; it's a promise of swift, gentle, and reliable hair styling for those who appreciate the finer things in life.
        Easy and Intuitive Use: Say goodbye to complicated gadgets. Alpha Beta is crafted for comfort and simplicity, perfect for those who value straightforward, hassle-free technology. - Time-Saving Technology: We understand your time is precious. That's why Alpha Beta cuts down drying time significantly, giving you more moments to enjoy life's pleasures.
        Make every day a good hair day with Alpha Beta. Experience the blend of sophistication and simplicity today.
        [Your Product Link Here]

        #alphabeta #UltraFast #Hairdryer #Alpha
        
        ###

        Note: Make a post only, do not add sentences outside the post
        
    """

    llm = ChatCohere(
        cohere_api_key=COHERE_API_KEY,
        model="command-r-plus",
        temperature=0.1
    )
    docs = db.similarity_search(query)
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    generate = chain.run(docs)
   
    return generate