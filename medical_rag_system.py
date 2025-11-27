# medical_rag_system.py
import pandas as pd
import numpy as np
import re
import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from typing import List, Dict
import sys
import os

class MedicalDataPreprocessor:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text):
        """Clean medical text data"""
        if pd.isna(text):
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        return text.strip()
    
    def smart_chunking(self, text, transcription_id, medical_specialty):
        """Split text into chunks with medical context preservation"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                if current_chunk:
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "transcription_id": transcription_id,
                            "medical_specialty": medical_specialty,
                            "chunk_length": len(chunk_text),
                            "is_medical": True,
                            "source": "mtsamples"
                        }
                    })
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "transcription_id": transcription_id,
                    "medical_specialty": medical_specialty,
                    "chunk_length": len(chunk_text),
                    "is_medical": True,
                    "source": "mtsamples"
                }
            })
        return chunks

class MedicalVectorStore:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.metadata = []
    
    def create_embeddings(self, chunks: List[Dict]):
        """Create embeddings for medical chunks"""
        texts = [chunk["text"] for chunk in chunks]
        self.metadata = [chunk["metadata"] for chunk in chunks]
        
        print("Creating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks
        print(f"Created vector store with {len(chunks)} documents")
    
    def similarity_search(self, query: str, k: int = 5):
        """Search for similar medical documents"""
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'text': self.chunks[idx]['text'],
                    'metadata': self.metadata[idx],
                    'score': distances[0][i]
                })
        return results

class MedicalRAGPipeline:
    def __init__(self, vector_store, gemini_api_key):
        self.vector_store = vector_store
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents into context"""
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"Document {i+1} (Specialty: {doc['metadata']['medical_specialty']}):\n{doc['text']}")
        return "\n\n".join(context_parts)
    
    def query(self, question: str, max_docs: int = 5) -> Dict:
        """Query the RAG pipeline"""
        documents = self.vector_store.similarity_search(question, max_docs)
        context = self.format_context(documents)
        
        prompt = f"""You are a safe medical assistant. Use the following medical context to answer the question. 
If the context doesn't contain relevant information, say so and don't make up information.

Medical Context:
{context}

Question: {question}

Provide a safe, evidence-based answer. If unsure, recommend consulting a healthcare professional.
Answer:"""
        
        response = self.model.generate_content(prompt)
        
        return {
            "question": question,
            "answer": response.text,
            "source_documents": documents,
            "context": context
        }

def create_sample_medical_data():
    """Create comprehensive sample medical data"""
    sample_data = [
        {
            'medical_specialty': 'Cardiology',
            'transcription': """
            The patient presents with chest pain radiating to left arm and shortness of breath. 
            ECG shows ST elevation in anterior leads. Cardiac enzymes are elevated including troponin.
            Diagnosis of acute myocardial infarction. Treatment includes aspirin, nitroglycerin, 
            oxygen therapy, and consideration for percutaneous coronary intervention. 
            Patient should be monitored in cardiac care unit.
            """
        },
        {
            'medical_specialty': 'Pulmonology',
            'transcription': """
            Patient presents with productive cough, fever, chills, and shortness of breath. 
            Physical exam reveals crackles on lung auscultation in right lower lobe.
            Chest X-ray shows consolidation consistent with pneumonia. 
            Treatment includes antibiotics like azithromycin, bronchodilators, 
            and supportive care with hydration and rest.
            """
        },
        {
            'medical_specialty': 'Endocrinology',
            'transcription': """
            Patient reports polyuria, polydipsia, unexplained weight loss, and fatigue.
            Laboratory results show fasting blood glucose 250 mg/dL, HbA1c 10.2%.
            Diagnosis of type 2 diabetes mellitus. Management includes metformin initiation,
            dietary modification focusing on carbohydrate counting, regular exercise,
            and blood glucose monitoring. Patient education on diabetes self-management.
            """
        },
        {
            'medical_specialty': 'Neurology',
            'transcription': """
            Patient presents with unilateral throbbing headache, photophobia, phonophobia, and nausea.
            Headache lasts 4-72 hours and is aggravated by physical activity.
            Diagnosis of migraine without aura. Acute treatment includes triptans like sumatriptan,
            NSAIDs for pain relief, and antiemetics for nausea. Preventive measures discussed
            including trigger avoidance and lifestyle modifications.
            """
        },
        {
            'medical_specialty': 'Gastroenterology',
            'transcription': """
            Patient complains of epigastric pain, heartburn, and acid regurgitation.
            Symptoms worsen after meals and at night. Endoscopy shows erosive esophagitis.
            Diagnosis of gastroesophageal reflux disease. Treatment includes proton pump inhibitors,
            lifestyle modifications like weight loss, avoiding trigger foods, and elevating head during sleep.
            """
        },
        {
            'medical_specialty': 'Orthopedics',
            'transcription': """
            Patient presents with joint pain, stiffness, and decreased range of motion in knees.
            X-rays show joint space narrowing and osteophyte formation.
            Diagnosis of osteoarthritis. Management includes acetaminophen for pain,
            NSAIDs for inflammation, physical therapy for strengthening, weight management,
            and consideration of joint injection for severe cases.
            """
        }
    ]
    return sample_data

def initialize_system(gemini_api_key):
    """Initialize the medical RAG system"""
    
    # Create sample data
    sample_data = create_sample_medical_data()
    
    # Preprocess data
    preprocessor = MedicalDataPreprocessor()
    all_chunks = []
    
    for idx, record in enumerate(sample_data):
        transcription_id = f"sample_{idx}"
        medical_specialty = record['medical_specialty']
        text = record['transcription']
        
        cleaned_text = preprocessor.clean_text(text)
        chunks = preprocessor.smart_chunking(cleaned_text, transcription_id, medical_specialty)
        all_chunks.extend(chunks)
    
    # Create vector store
    vector_store = MedicalVectorStore()
    vector_store.create_embeddings(all_chunks)
    
    # Create RAG pipeline
    rag_pipeline = MedicalRAGPipeline(vector_store, gemini_api_key)
    
    return rag_pipeline
