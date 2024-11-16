# ToDo List for Local LLM-based DM Assistant App

## 1. Setup Infrastructure
- [x] Configure Docker environment to run PostgreSQL as a vector store.
- [x] Ensure Postgres container is accessible and connected to app.
- [x] Install and configure necessary libraries for Llama model and LlamaIndex in environment.
- [x] Verify basic connectivity between Llama model, PostgreSQL, and the app.

## 2. Implement Vector Store (Postgres)
- [ ] Set up tables in PostgreSQL for vector storage, optimized for retrieval.
- [ ] Configure schema for documents and metadata (e.g., DnD rules, campaign notes, character info).
- [ ] Integrate basic RAG setup to retrieve and store vectors in Postgres.
- [ ] Test vector storage and retrieval from PostgreSQL.

## 3. PDF & Markdown Parsing
- [ ] Set up a system to parse PDFs and Markdown files for storing rules, notes, and campaign content.
- [ ] Implement a parser to convert documents into vector embeddings.
- [ ] Store parsed content in the PostgreSQL vector store.
- [ ] Test retrieval of parsed content and verify the accuracy of vector embeddings.

## 4. Develop Agent Modules
### Agent 1: Rules, Mechanics, etc.
- [ ] Develop an agent to answer questions about DnD rules and mechanics.
- [ ] Connect this agent to the PostgreSQL vector store for retrieving relevant information.
- [ ] Test rule-related question responses.

### Agent 2: Storytelling & Notes
- [ ] Create an agent focused on storytelling and campaign note retrieval.
- [ ] Feed it with background lore, world-building info, and session notes.
- [ ] Ensure it retrieves relevant notes based on context and user queries.

### Agent 3: Campaign-Specific Questions
- [ ] Build an agent for handling campaign-specific questions (e.g., NPC backgrounds, plot points).
- [ ] Link this agent to query  campaign content in the vector store.
- [ ] Test responses to ensure they stay relevant to the campaign.

## 5. LLMOps & Quality Control
- [ ] Define an evaluation process to ensure the quality of model responses.
- [ ] Set up testing and monitoring for LLM responses to tune and improve model accuracy over time.
- [ ] Plan for regular model retraining or fine-tuning based on feedback.

## 6. Streamlit UI Interface
- [ ] Develop a basic Streamlit interface for interacting with the app.
- [ ] Add input fields to submit queries and display responses from different agents.
- [ ] Implement a feedback system in the UI for quality assurance.
- [ ] Test UI for usability and smooth integration with back-end agents.

## 7. Final Testing & Deployment
- [ ] Test the entire system end-to-end, from query input to LLM response.
- [ ] Debug and optimize components as necessary.
- [ ] Document the setup, usage instructions, and future improvement ideas.
- [ ] Deploy the app locally or in a controlled environment for practical use during DnD sessions.
