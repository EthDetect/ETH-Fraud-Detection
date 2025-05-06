import os
import csv
import time
from neo4j import GraphDatabase

# Neo4j configuration
uri = ""
user = ""
password = ""

# Folder paths 
folder_paths = {
    'nodes': '',
    'relationships': '',
    'labels': ''
}

def load_nodes(session, folder_path):
    start_time = time.time()
    print(f"Starting to load nodes from {folder_path}...")
    
    file_count = 0
    row_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                row_index = 0
                for row in reader:
                    row_index += 1
                    if len(row) < 2:  
                        print(f"Skipping row with insufficient columns in {file_path}: {row}")
                        continue
                    id = row[0]  
                    feature = row[1] if len(row) > 1 else None  
                    try:
                        session.run(
                            "MERGE (n:Node {id: $id}) SET n.feature = $feature",
                            {"id": id, "feature": feature}
                        )
                        if row_index % 1000 == 0:  
                            print(f"Processed {row_index} rows in {file_path}")
                    except Exception as e:
                        print(f"Error inserting node {id} at row {row_index}: {e}")
                    row_count += 1
            file_count += 1
            print(f"Nodes loaded from {file_path}")
    
    end_time = time.time()
    print(f"Finished loading {file_count} node files with {row_count} rows in {end_time - start_time:.2f} seconds.")

def load_relationships(session, folder_path):
    start_time = time.time()
    print(f"Starting to load relationships from {folder_path}...")
    
    file_count = 0
    row_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                row_index = 0
                for row in reader:
                    row_index += 1
                    if len(row) < 2:  
                        print(f"Skipping row with insufficient columns in {file_path}: {row}")
                        continue
                    start_node = row[0] 
                    end_node = row[1] 
                    try:
                        session.run(
                            """
                            MATCH (a:Node {id: $start_node}), (b:Node {id: $end_node})
                            MERGE (a)-[:RELATIONSHIP]->(b)
                            """,
                            {"start_node": start_node, "end_node": end_node}
                        )
                        if row_index % 1000 == 0: 
                            print(f"Processed {row_index} rows in {file_path}")
                    except Exception as e:
                        print(f"Error creating relationship between {start_node} and {end_node} at row {row_index}: {e}")
                    row_count += 1
            file_count += 1
            print(f"Relationships loaded from {file_path}")
    
    end_time = time.time()
    print(f"Finished loading {file_count} relationship files with {row_count} rows in {end_time - start_time:.2f} seconds.")

def load_labels(session, file_path):
    start_time = time.time()
    print(f"Starting to load labels from {file_path}...")
    
    row_count = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        row_index = 0
        for row in reader:
            row_index += 1
            if len(row) < 2:  
                print(f"Skipping row with insufficient columns in {file_path}: {row}")
                continue
            id = row[0]  
            label = row[1] 
            try:
                session.run(
                    """
                    MATCH (n:Node {id: $id})
                    SET n:`$label`
                    """,
                    {"id": id, "label": label}
                )
                if row_index % 1000 == 0: 
                    print(f"Processed {row_index} rows in {file_path}")
            except Exception as e:
                print(f"Error setting label {label} for node {id} at row {row_index}: {e}")
            row_count += 1
    
    end_time = time.time()
    print(f"Finished loading labels with {row_count} rows in {end_time - start_time:.2f} seconds.")

def main():
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            load_nodes(session, folder_paths['nodes'])
            load_relationships(session, folder_paths['relationships'])
            load_labels(session, os.path.join(folder_paths['labels'], 'label_all_data.csv'))
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()