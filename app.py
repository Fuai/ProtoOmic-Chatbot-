import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import config
import os

# Set page title
st.set_page_config(page_title="ProtoOmic Chatbot", page_icon="🧬")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = ConversationBufferMemory(memory_key="chat_history")
    st.session_state.summary_generated = False
    st.session_state.waiting_for_question = True

# Title
st.title("🧬 ProtoOmic Chatbot")

# Sidebar for API key
with st.sidebar:
    st.header("OpenAI API Configuration")
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("TurboID_ASK1_ML_Final.csv")
    return df

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = load_data()

# Direct protein input
protein_query = st.text_input("Enter protein name or ID:", placeholder="e.g., AT1G09570 or DMR6")

def find_protein(query):
    if not query:
        return None
    
    query = query.strip().upper()
    matches = st.session_state.data[
        (st.session_state.data['Proteins'].str.contains(query, case=False, na=False)) |
        (st.session_state.data['Fasta_headers'].str.contains(query, case=False, na=False))
    ]
    
    if not matches.empty:
        return matches.iloc[0]
    return None

def get_protein_info(protein_data):
    fasta_header = protein_data['Fasta_headers']
    symbols = []
    if "Symbols:" in fasta_header:
        symbols_part = fasta_header.split("Symbols:")[1].split("|")[0].strip()
        symbols = [s.strip() for s in symbols_part.split(",")]
    
    return {
        'protein_id': protein_data['Proteins'],
        'symbols': ", ".join(symbols) if symbols else "No symbols found",
        'molecular_weight': protein_data['MolecularWeight'],
        'location': protein_data['Subcellular localization'],
        'log2fc': protein_data['log2FC'],
        'pvalue': protein_data['pval'],
        'description': protein_data['Fasta_headers']
    }

def setup_llm():
    return ChatOpenAI(
        model_name=config.DEFAULT_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        streaming=False
    )

def format_chat_history(history):
    formatted = []
    for msg in history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted[-4:])

def get_initial_summary(protein_info):
    llm = setup_llm()
    prompt = ChatPromptTemplate.from_template("""Provide a concise summary of this protein:
    ID: {protein_id}
    Symbols: {symbols}
    Description: {description}
    Location: {location}
    Expression: log2FC={log2fc}, p={pvalue}
    
    Give a 2-3 sentence summary focusing on key features and potential significance.""")
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(**protein_info)

# Main interface
if protein_query:
    protein_data = find_protein(protein_query)
    
    if protein_data is not None:
        protein_info = get_protein_info(protein_data)
        
        # Show protein details in a compact two-column layout
        st.markdown("### Protein Details")
        
        # Create two columns for basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information:**")
            st.write(f"**ID:** {protein_info['protein_id']}")
            st.write(f"**Symbols:** {protein_info['symbols']}")
            st.write(f"**Location:** {protein_info['location']}")
        
        with col2:
            st.write("**Expression Data:**")
            st.write(f"**MW:** {protein_info['molecular_weight']} Da")
            st.write(f"**Log2FC:** {protein_info['log2fc']}")
            st.write(f"**p-value:** {protein_info['pvalue']}")
        
        # Description below the columns
        st.write("**Description:**")
        st.write(protein_info['description'])
        
        st.markdown("---")

        # Chat interface
        if not api_key:
            st.warning("Please enter your OpenAI API key in the sidebar to enable AI analysis")
        else:
            # Generate initial summary if not done
            if not st.session_state.summary_generated:
                st.markdown("### AI Summary")
                with st.spinner("Generating protein summary..."):
                    summary = get_initial_summary(protein_info)
                    st.markdown(f"🧬 {summary}")
                st.session_state.summary_generated = True
                st.markdown("---")
            
            # Display previous QA pairs
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.write("💭:", message["content"])
                else:
                    st.markdown(f"🧬: {message['content']}")
            
            # Only show input if waiting for question
            if st.session_state.waiting_for_question:
                user_input = st.text_input(
                    "Ask a question:",
                    key="qa_input",
                    placeholder="What would you like to know about this protein?"
                )
                
                if user_input:
                    st.session_state.waiting_for_question = False
                    # Add user message to history
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
                    try:
                        # Setup LLM if not exists
                        if 'llm' not in st.session_state:
                            st.session_state.llm = setup_llm()
                        
                        # Format chat history for context
                        chat_context = format_chat_history(st.session_state.chat_history[:-1])
                        
                        # Create prompt template and chain
                        prompt = ChatPromptTemplate.from_template(config.CHAT_TEMPLATE)
                        chain = LLMChain(llm=st.session_state.llm, prompt=prompt)
                        
                        # Get response
                        response = chain.run(
                            question=user_input,
                            chat_history=chat_context,
                            **protein_info
                        )
                        
                        # Add response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # Rerun to update chat display
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Show continue button only after an answer
            if not st.session_state.waiting_for_question and st.session_state.chat_history:
                if st.button("Ask Another Question"):
                    st.session_state.waiting_for_question = True
                    st.rerun()
                
                if st.button("Start New Conversation"):
                    st.session_state.chat_history = []
                    st.session_state.summary_generated = False
                    st.session_state.waiting_for_question = True
                    st.rerun()
    else:
        st.error(f"No protein found matching '{protein_query}'. Please try another name or ID.") 