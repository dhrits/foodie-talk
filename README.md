---
title: Foodie-Talk
emoji: 🍔
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: cc-by-nc-nd-4.0
---

# A LangGraph Based Agentic Application for your inner foodie

## Introduction
This is a educational project made as a part of *AI Engineering Bootcamp* from [*AI Makerspace*](https://aimakerspace.io/). The code and the data is strictly for educational use. In particular, for any subset of Yelp data, please find the terms of use [here](https://business.yelp.com/external-assets/files/Yelp-JSON.zip). For 10000 restaurant reviews, please find the license [here](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews).

## Installation
1. Install uv using the instructions [here](https://docs.astral.sh/uv/getting-started/installation/). Eg, on Mac run `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Create a virtual environment by running the command `uv venv`
3. Activate the virtual environment by running `source .venv/bin/activate`
4. Install dependencies by running `uv sync`
5. Run the app by running the command `uv run chainlit run app.py`


## Task 1: The Problem

Foodie-Talk *solves the problem of finding the perfect food for your tastes when faced with an overwhelming number of options.*

Consumers today have more choices than ever when it comes to prepped food. Restaurants tend to cater to every niche cuisine imaginable. Additionally, services like Grubhub, Doordash, Seamless and Uber Eats have made it trivially easy to get this food delivered to nearby locations. **Herein lies the problem.**. Today's consumer faces an overwhelming amount of choice. Every option is available, but there's *no guidance or advice on what to choose and no trusted expert friend to answer questions.*

Additionally, while there is a plethora of information available online in the form of user-reviews and ratings, there are few resources to summarize this data and answer questions from the consumer. While algorithmic engines on food delivery portals can recommend restaurants based on past history, no resource exists to help the user make those initial decisions in the first place. 

What if there was a better way? 

## Task 2: The Solution - Foodie-Talk Restaurant-Expert Agent
Foodie-Talk is an AI Agent that gathers and summarizes publicly available information on restaurants and presents it to users in a fun conversational way. Feeling like having Mexican cuisine? Just ask the Foodie Talk agent for the best Mexican restaurants in the area. Still unsure? Ask for a summary of user-reviews. Feeling like eating healthy? Ask for the healthiest eateries. Ask follow-up questions and get appropriate answers from the best online sources. 

Foodie-Talk presents restaurant and food information to you in a summarized manner saving your valuable time for the things that matter most: enjoying your meal. 

### Technical Design
Foodie-Talk leverages LangChain and LangGraph to build an agentic application which relies on web-search results and an available database of restaurant reviews from Yelp and Kaggle to help answer any questions a user may have about food and restaurants in their location. Each aspect of the application is described below. 

1. LLM - At the top level, I make use of **gpt-4.1** as an orchestrator and decision-maker (using LangGraph) for this application. In particular, this LLM **gets the user's queries/chats and decides whether it needs information from the Internet or its available database of user-reviews to answer the question**. Should Additional information be needed, this LLM routes the flow-of-control of the application to a RAG (Retrieval Augmented Generation) pipeline to get appropriate context to respond to the user. This RAG sub-graph is also powered gpt-4.1. 
2. Embedding Model - The RAG part of this application retrieves user-review information from a vector database where it is indexed using an embedding model. The embedding model I use for this application is [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l). I additionally finetune this model on restaurant review data from Kaggle. 
3. Orchestration - The overall orchestration of the application is via LangGraph. The top level conditional edge (from the START node) relies on GPT-4.1 to decide whether to let the LLM directly answer the user's questions, or retrieve additional information from the Internet using **Tavily** and RAG subgraphs. The entire application is then made available to the user using **Chainlit**.
4. Vector Database - For hosting an indexed version of user-reviews, I make use of **Qdrant**, a scalable and open-source vector database. For this project, I used a hosted version of Qdrant for maximum scalability. 
5. Monitoring - Application monitoring is handled by **LangSmith**.
6. Evaluation - Overall evaluation of the application is performed using RAGAS. In particular, RAGAS helps develop a synthetic dataset for evaluating the retrieval performance of the base vs finetuned snowflake-arctic-embed-l model.  
7. User-Interface - The application is delivered to the user by making use of **Chainlit**.
8. Serving and Inference - Serving and inference is handled by a combination of Huggingface Spaces (for the base chainlit application) and Huggingface Inference Endpoints (for inference of the finetuned model).

![LangGraph Agentic Graph Diagram](foodie-talk-graph.png)

## Task 3: Dealing with Data
This application makes use of two *non-commercial datasets*. The first one is the **[Yelp Business Reviews Dataset](https://business.yelp.com/external-assets/files/Yelp-JSON.zip)** and the second is **[Kaggle's 10000 Restaurant Reviews Dataset](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews)**. The ways in which this data is used is outlined below.

1. The first step is to explore the data and create subsets of it which can be of use to the application. In particular, the *Yelp dataset is prohibitively expensive to index in its entirety given the time and cost constraints for this project.* Thus the [data_gathering.ipynb notebook](https://github.com/dhrits/foodie-talk-nbs/blob/main/nbs/data_gathering/data_gathering.ipynb) creates a subset of this data which can be used as part of an **Agentic RAG System**. 
2. Once the Yelp data subset is prepared in step 1, it is *indexed into an instance of Qdrant cloud*. This is done in the [index_data.ipynb notebook](https://github.com/dhrits/foodie-talk-nbs/blob/main/nbs/data_gathering/index_data.ipynb). The embeddings used are [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l). The final version of this application makes use of a finetuned version of these embeddings as outlined below.
    - The chunking strategy used for this RAG indexing was just the `RecursiveCharacterTextSplitter` with a `chunk_size` of 700 and a chunk overlap of 250. This was based on the 75th percentile length of the reviews in the Yelp reviews dataset. 
3. While the Yelp dataset is used as a source of "realtime" indexed RAG data for answering the user's food related questions, the Kaggle 10000 Restaurant Reviews dataset (which is smaller and more manageable given project cost and time constraints) is used to **generate synthetic data to actually finetune Snowflake-Arctic-Embed-l.** Additionally, another subset of this data is used to generate the Golden Test Dataset.
4. I also make use of `TavilySearch` **langchain community tool** to get realtime restaurant data from the Internet.

## Task 4: Building a quick end to end prototype

A prototype application is included at [Foodie-Talk](https://huggingface.co/spaces/deman539/foodie-talk) (Please note that **I may disable this application because it costs a lot to keep the huggingface inference endpoints and Qdrant Cloud subscription ongoing. Please message me to re-enable it**).

## Task 5: Creating a Golden Test Data Set

In order to have a well-tuned **Agentic RAG Application**, it is important to have a Golden Test Data Set which can be used to benchmark and test the performance of the RAG chain. This Golden Test Data Set is generated using the Kaggle 10000 Restaurant Review dataset. I generated [training](https://github.com/dhrits/foodie-talk-nbs/blob/main/nbs/finetuning/training_dataset.jsonl), [validation](https://github.com/dhrits/foodie-talk-nbs/blob/main/nbs/finetuning/val_dataset.jsonl) and [test](https://github.com/dhrits/foodie-talk-nbs/blob/main/nbs/finetuning/val_dataset.jsonl) datasets to both finetune the performance of [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l) model, as well as to evaluate the performance of the model against the baseline using RAGAS. These experiments are outlined in the [finetuning.ipynb notebook](https://github.com/dhrits/foodie-talk-nbs/blob/main/nbs/finetuning/finetuning.ipynb).

Additionally, the[finetuning.ipynb notebook](https://github.com/dhrits/foodie-talk-nbs/blob/main/nbs/finetuning/finetuning.ipynb) also assesses the performance of the baseline and finetuned RAG chain on key metrics like faithfulness, response relevance, context precision, and context recall. **A table is included at the bottom of the notebook comparing the performance of the two pipelines**. **It is clear that the finetuned pipeline performs much better in comparison to the baseline on every metric.**

## Task 6: Fine-Tuning Open Source Embeddings
The code for finetuning [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l) is also included in the [finetuning.ipynb notebook](https://github.com/dhrits/foodie-talk-nbs/blob/main/nbs/finetuning/finetuning.ipynb) notebook. A *comparative performance between the finetuned embeddings and the original embeddings is included at the bottom of the notebook.*

Additionally, the fintuned embeddings are available on huggingface hub [here](https://huggingface.co/deman539/food-review-ft-snowflake-l-f18eeff6-7504-48c7-af10-1d2d85ca8caa).

## Task 7: Assessing Performance
An analysis of the performance of the finetuned RAG chain is included in [finetuning.ipynb notebook](https://github.com/dhrits/foodie-talk-nbs/blob/main/nbs/finetuning/finetuning.ipynb) (this notebook also does a comparative analysis of the performance of the baseline and finetuned pipelines). Additionally, an assessment of the entire Agentic RAG pipeline is included in the [agentic_rag.ipynb notebook](https://github.com/dhrits/foodie-talk-nbs/blob/main/nbs/agents/agentic_rag.ipynb). 

### Future Enhancements
Going forward, there's a few features I'd like to be able to support:
1. Multi-User support to allow the application to build a taste profile for each user and personalize the experience using chat memories.
2. Support for MCP servers for Grubhub, Uber Eats and Seamless allowing the user the flexibility to order food from within the application.
3. Better/more-realtime review support by relying on Yelp's official API.
