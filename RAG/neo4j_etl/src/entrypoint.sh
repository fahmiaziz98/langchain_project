#!/bin/bash

# Run any setup steps or pre-processing tasks here
echo "Running ETL to move hospital data from csvs to Neo4j..."
python3 load_csv_to_db.py

# Create Embedding Neo4j Graph
echo "Create Embedding from data reviews..."
python3 create_embedding_review.py