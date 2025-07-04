import json
import os
import glob
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

# BERTopic and related imports
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.graph_objects as go
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dynamic_topic_processing.log'),
        logging.StreamHandler()
    ]
)

class DynamicTopicModeling:
    def __init__(self, output_dir="dynamic_topic_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize models (same as multibertopic.py)
        self.embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        self.hdbscan_model = HDBSCAN(min_cluster_size=40, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        
        logging.info("DynamicTopicModeling initialized with models")
    
    def load_saved_embeddings(self, embedding_dir="embedding_model"):
        """Load pre-saved embedding model and embeddings"""
        try:
            embedding_path = Path(embedding_dir)
            if not embedding_path.exists():
                logging.warning(f"Embedding directory {embedding_dir} not found. Will use default model.")
                return None, None
            
            # Load the saved embedding model
            logging.info(f"Loading saved embedding model from {embedding_dir}...")
            self.embedding_model = SentenceTransformer(embedding_dir)
            
            # Check if saved embeddings exist
            saved_embeddings_path = Path("saved_embeddings")
            if saved_embeddings_path.exists():
                try:
                    # Load saved embeddings
                    with open(saved_embeddings_path / "documents.json", 'r') as f:
                        saved_documents = json.load(f)
                    with open(saved_embeddings_path / "metadata.json", 'r') as f:
                        saved_metadata = json.load(f)
                    
                    logging.info(f"Loaded {len(saved_documents)} pre-computed embeddings")
                    return saved_documents, saved_metadata
                except Exception as e:
                    logging.warning(f"Could not load saved embeddings: {str(e)}")
                    return None, None
            else:
                logging.info("No saved embeddings found. Will compute new embeddings.")
                return None, None
                
        except Exception as e:
            logging.error(f"Error loading saved embedding model: {str(e)}")
            return None, None
    
    def load_all_datasets(self, data_dir="nmsince2000"):
        """Load and combine all yearly datasets with timestamps"""
        all_documents = []
        all_timestamps = []
        all_titles = []
        all_abstracts = []
        all_journals = []
        all_dois = []
        
        # Get all JSON files sorted by year
        json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        
        logging.info(f"Found {len(json_files)} yearly datasets to process")
        
        for file_path in json_files:
            year = Path(file_path).stem
            logging.info(f"Processing year: {year}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Filter out entries with empty titles
                data = [item for item in data if item.get('title')]
                
                # Extract data for this year
                for item in data:
                    title = item.get('title', '')
                    abstract = item.get('abstract', '')
                    journal = item.get('journal', '')
                    doi = item.get('doi', '')
                    
                    # Combine title and abstract
                    title_abstract = f"{title} {abstract}"
                    
                    all_documents.append(title_abstract)
                    all_timestamps.append(year)  # Use year as string timestamp
                    all_titles.append(title)
                    all_abstracts.append(abstract)
                    all_journals.append(journal)
                    all_dois.append(doi)
                
                logging.info(f"Loaded {len(data)} documents from {year}")
                
            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}")
                continue
        
        logging.info(f"Total documents loaded: {len(all_documents)}")
        logging.info(f"Year range: {min(all_timestamps)} - {max(all_timestamps)}")
        
        return all_documents, all_timestamps, all_titles, all_abstracts, all_journals, all_dois
    
    def train_dynamic_model(self, documents, timestamps, titles, abstracts, journals, dois, use_saved_embeddings=True):
        """Train BERTopic model and then apply dynamic topic modeling"""
        try:
            logging.info("Training base BERTopic model...")
            
            # Try to load saved embeddings first
            embeddings = None
            if use_saved_embeddings:
                saved_documents, saved_metadata = self.load_saved_embeddings()
                if saved_documents is not None and saved_metadata is not None:
                    # Check if the documents match
                    if len(saved_documents) == len(documents):
                        logging.info("Using pre-computed embeddings...")
                        embeddings = np.array(saved_documents)
                    else:
                        logging.warning("Saved embeddings count doesn't match current documents. Computing new embeddings.")
            
            # If no saved embeddings available or they don't match, compute new ones
            if embeddings is None:
                logging.info("Generating new embeddings...")
                embeddings = self.embedding_model.encode(documents, show_progress_bar=False)
            
            # Create representation model
            representation_model = KeyBERTInspired()
            ctfidf_model = ClassTfidfTransformer()

            # Initialize BERTopic model
            topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model,
                top_n_words=20,
                verbose=False,
                calculate_probabilities=True,
                representation_model=representation_model,
                ctfidf_model=ctfidf_model
            )
            
            # Train base model
            logging.info("Training topic model...")
            topics, probs = topic_model.fit_transform(documents, embeddings)
            
            # Reduce outliers
            logging.info("Reducing outliers...")
            new_topics = topic_model.reduce_outliers(documents, topics, probabilities=probs, strategy="probabilities")
            topic_model.update_topics(documents, topics=new_topics)
            
            # Fine-tune topic representations
            vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=3)
            topic_model.update_topics(documents, vectorizer_model=vectorizer_model, top_n_words=20)
            
            # Apply dynamic topic modeling
            logging.info("Applying dynamic topic modeling...")
            topics_over_time = topic_model.topics_over_time(
                documents, 
                timestamps, 
                nr_bins=len(set(timestamps)),  # One bin per year
                global_tuning=True, 
                evolution_tuning=True,
                datetime_format="%Y"  # Specify 4-digit year format
            )
            
            # Save model
            model_save_path = self.output_dir / "model"
            topic_model.save(str(model_save_path), serialization="safetensors", save_ctfidf=True, save_embedding_model=self.embedding_model)
            
            # Save topics over time
            topics_over_time.to_csv(self.output_dir / "topics_over_time.csv", index=False)
            
            # Generate and save topic info
            topic_info = topic_model.get_topic_info()
            # Clean the data to avoid CSV issues with quotes, newlines, and special characters
            topic_info_clean = topic_info.copy()
            for col in topic_info_clean.columns:
                if topic_info_clean[col].dtype == 'object':
                    # Truncate very long text and clean problematic characters
                    topic_info_clean[col] = (topic_info_clean[col]
                        .astype(str)
                        .str.replace('"', '""')  # Escape quotes
                        .str.replace('\n', ' ')  # Replace newlines
                        .str.replace('\r', ' ')  # Replace carriage returns
                        .str.replace('\t', ' ')  # Replace tabs
                        .str[:1000]  # Truncate to 1000 characters to avoid massive CSV files
                    )
            topic_info_clean.to_csv(self.output_dir / "topic_info.csv", index=False, quoting=1)  # quoting=1 uses QUOTE_ALL
            
            # Save topics as JSON
            topics_dict = topic_model.get_topics()
            with open(self.output_dir / "topics.json", 'w') as f:
                json.dump(topics_dict, f, indent=2)            
            
            # Generate visualizations
            self.generate_dynamic_visualizations(topic_model, topics_over_time, titles, embeddings)
            
            # Export topic documents to spreadsheets
            self.export_topic_documents(topic_model, documents, timestamps, titles, abstracts, journals, dois)
            
            # Export topic frequency by year to spreadsheets
            self.export_topic_frequency_by_year(topics_over_time)
            
            # Create summary statistics
            summary = {
                'total_documents': len(documents),
                'total_topics': len(topic_info),
                'year_range': f"{min(timestamps)} - {max(timestamps)}",
                'total_years': len(set(timestamps)),
                'outlier_documents': len(topic_info[topic_info['Topic'] == -1]),
                'largest_topic_size': int(topic_info['Count'].max()),
                'smallest_topic_size': int(topic_info['Count'].min()),
                'avg_topic_size': float(topic_info['Count'].mean()),
                'processing_time': datetime.now().isoformat()
            }
            
            with open(self.output_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

            topic_model.embedding_model.embedding_model.save_pretrained("embedding_model")
            
            logging.info(f"Dynamic topic modeling completed: {len(topic_info)} topics found")
            return topic_model, topics_over_time, summary
        
        
            
        except Exception as e:
            logging.error(f"Error in dynamic topic modeling: {str(e)}")
            return None, None, None
    

    def generate_dynamic_visualizations(self, topic_model, topics_over_time, titles, embeddings):
        """Generate dynamic topic modeling visualizations"""
        try:
            logging.info("Generating dynamic visualizations...")
            
            # Reduce embeddings for visualization
            reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
            
            # 1. Topics over time visualization (core BERTopic functionality)
            fig_over_time = topic_model.visualize_topics_over_time(
                topics_over_time, 
                width=1200,
                height=800
            )
            fig_over_time.write_html(str(self.output_dir / "topics_over_time_visualization.html"))
            
            # 2. Document visualization
            fig_docs = topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings)
            fig_docs.update_layout(
                font=dict(family="Arial, sans-serif", size=12),
                title_font=dict(family="Arial, sans-serif", size=16),
                title="Document Clusters Over Time"
            )
            fig_docs.write_html(str(self.output_dir / "documents_visualization.html"))
            
            # 3. Topic hierarchy
            hierarchy_fig = topic_model.visualize_hierarchy()
            hierarchy_fig.write_html(str(self.output_dir / "topic_hierarchy.html"))
            
            # 4. Topics visualization
            topics_fig = topic_model.visualize_topics()
            topics_fig.write_html(str(self.output_dir / "topics_visualization.html"))
            
            logging.info(f"Dynamic visualizations saved to {self.output_dir}")
            
        except Exception as e:
            logging.error(f"Error generating dynamic visualizations: {str(e)}")
    
    def export_topic_documents(self, topic_model, documents, timestamps, titles, abstracts, journals, dois):
        """Export documents for each topic to separate CSV files"""
        try:
            logging.info("Exporting topic documents to spreadsheets...")
            
            # Get topics for each document
            topics, _ = topic_model.transform(documents)
            
            # Create topic documents directory
            topic_docs_dir = self.output_dir / "topic_documents"
            topic_docs_dir.mkdir(exist_ok=True)
            
            # Group documents by topic
            topic_groups = defaultdict(list)
            for i, topic in enumerate(topics):
                topic_groups[topic].append({
                    'year': timestamps[i],
                    'title': titles[i],
                    'abstract': abstracts[i],
                    'journal': journals[i],
                    'doi': dois[i]
                })
            
            # Export each topic to a CSV file
            for topic_id, docs in topic_groups.items():
                if topic_id == -1:  # Skip outlier documents
                    continue
                    
                # Create DataFrame
                df = pd.DataFrame(docs)
                
                # Clean the data to avoid CSV issues
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = (df[col]
                            .astype(str)
                            .str.replace('"', '""')  # Escape quotes
                            .str.replace('\n', ' ')  # Replace newlines
                            .str.replace('\r', ' ')  # Replace carriage returns
                            .str.replace('\t', ' ')  # Replace tabs
                        )
                
                # Save to CSV
                filename = f"topic_{topic_id}_documents.csv"
                filepath = topic_docs_dir / filename
                df.to_csv(filepath, index=False, quoting=1)
                
                logging.info(f"Exported {len(docs)} documents for topic {topic_id} to {filename}")
            
            logging.info(f"Topic documents exported to {topic_docs_dir}")
            
        except Exception as e:
            logging.error(f"Error exporting topic documents: {str(e)}")
    
    def export_topic_frequency_by_year(self, topics_over_time):
        """Export topic frequency by year for each topic to separate CSV files"""
        try:
            logging.info("Exporting topic frequency by year to spreadsheets...")
            
            # Create topic frequency directory
            topic_freq_dir = self.output_dir / "topic_frequency_by_year"
            topic_freq_dir.mkdir(exist_ok=True)
            
            # Get unique topics (excluding -1 for outliers)
            unique_topics = topics_over_time['Topic'].unique()
            unique_topics = [t for t in unique_topics if t != -1]
            
            # Export each topic's frequency over time
            for topic_id in unique_topics:
                # Filter data for this topic
                topic_data = topics_over_time[topics_over_time['Topic'] == topic_id].copy()
                
                # Sort by year
                topic_data = topic_data.sort_values('Timestamp')
                
                # Select relevant columns and rename for clarity
                frequency_data = topic_data[['Timestamp', 'Count']].copy()
                frequency_data.columns = ['Year', 'Frequency']
                
                # Clean the data to avoid CSV issues
                for col in frequency_data.columns:
                    if frequency_data[col].dtype == 'object':
                        frequency_data[col] = (frequency_data[col]
                            .astype(str)
                            .str.replace('"', '""')  # Escape quotes
                            .str.replace('\n', ' ')  # Replace newlines
                            .str.replace('\r', ' ')  # Replace carriage returns
                            .str.replace('\t', ' ')  # Replace tabs
                        )
                
                # Save to CSV
                filename = f"topic_{topic_id}_frequency_by_year.csv"
                filepath = topic_freq_dir / filename
                frequency_data.to_csv(filepath, index=False, quoting=1)
                
                logging.info(f"Exported frequency data for topic {topic_id} to {filename}")
            
            logging.info(f"Topic frequency by year exported to {topic_freq_dir}")
            
        except Exception as e:
            logging.error(f"Error exporting topic frequency by year: {str(e)}")
    

    
    def run_dynamic_analysis(self, data_dir="nmsince2000", use_saved_embeddings=True):
        """Main function to run the complete dynamic topic analysis"""
        logging.info("Starting dynamic topic modeling analysis...")
        
        # Load all datasets
        documents, timestamps, titles, abstracts, journals, dois = self.load_all_datasets(data_dir)
        
        if not documents:
            logging.error("No documents loaded. Exiting.")
            return None
        
        # Train dynamic model
        topic_model, topics_over_time, summary = self.train_dynamic_model(documents, timestamps, titles, abstracts, journals, dois, use_saved_embeddings)
        
        if topic_model is None:
            logging.error("Failed to train dynamic topic model. Exiting.")
            return None
        
        logging.info("Dynamic topic modeling analysis completed successfully!")
        logging.info(f"Results saved to: {self.output_dir}")
        
        return {
            'topic_model': topic_model,
            'topics_over_time': topics_over_time,
            'summary': summary
        }

def main():
    """Main function to run the dynamic topic modeling"""
    dynamic_analyzer = DynamicTopicModeling()
    
    # Run the complete analysis with saved embeddings
    print("Starting dynamic topic modeling with saved embeddings...")
    results = dynamic_analyzer.run_dynamic_analysis(use_saved_embeddings=True)
    
    if results:
        print(f"\nDynamic topic modeling completed successfully!")
        print(f"Check the 'dynamic_topic_results' directory for outputs.")
        print(f"Summary: {results['summary']}")
    else:
        print("Dynamic topic modeling failed. Check logs for details.")

if __name__ == "__main__":
    main()
