"""
Document Processor Module
Handles PDF parsing, text/table/image extraction using Unstructured
"""

import os
import io
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import htmltabletomd
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from pypdf import PdfReader
from PIL import Image


class DocumentProcessor:
    """Process PDF documents and extract text, tables, and images"""
    
    def __init__(self, dataset_path: Path):
        """
        Initialize document processor
        
        Args:
            dataset_path: Path to folder containing PDF files
        """
        self.dataset_path = Path(dataset_path)
        self.figures_path = self.dataset_path / "figures"
        
        # Create figures directory if it doesn't exist
        self.figures_path.mkdir(exist_ok=True)
        
    def _clear_figures_directory(self):
        """Clear the figures directory before processing"""
        if self.figures_path.exists():
            shutil.rmtree(self.figures_path)
        self.figures_path.mkdir(exist_ok=True)
    
    def _extract_images_from_pdf(self, pdf_path: Path, output_dir: Path) -> List[str]:
        """
        Extract images from PDF using PyPDF
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            List of image file paths
        """
        image_paths = []
        
        try:
            reader = PdfReader(str(pdf_path))
            image_count = 0
            
            for page_num, page in enumerate(reader.pages):
                # Extract images from page
                if '/XObject' in page['/Resources']:
                    xObject = page['/Resources']['/XObject'].get_object()
                    
                    for obj_name in xObject:
                        obj = xObject[obj_name]
                        
                        if obj['/Subtype'] == '/Image':
                            try:
                                # Get image data
                                size = (obj['/Width'], obj['/Height'])
                                data = obj.get_data()
                                
                                # Determine image format
                                if obj['/ColorSpace'] == '/DeviceRGB':
                                    mode = "RGB"
                                elif obj['/ColorSpace'] == '/DeviceGray':
                                    mode = "L"
                                else:
                                    mode = "RGB"
                                
                                # Create PIL Image
                                if '/Filter' in obj:
                                    filter_type = obj['/Filter']
                                    if filter_type == '/FlateDecode':
                                        img = Image.frombytes(mode, size, data)
                                    elif filter_type == '/DCTDecode':
                                        img = Image.open(io.BytesIO(data))
                                    else:
                                        continue
                                else:
                                    img = Image.frombytes(mode, size, data)
                                
                                # Save image
                                image_count += 1
                                img_filename = f"page_{page_num + 1}_img_{image_count}.jpg"
                                img_path = output_dir / img_filename
                                img.save(str(img_path), 'JPEG')
                                image_paths.append(str(img_path))
                                
                            except Exception as e:
                                print(f"    ⚠️  Could not extract image {obj_name} from page {page_num + 1}: {e}")
                                continue
            
            if image_count > 0:
                print(f"  🖼️  Extracted {image_count} images using PyPDF")
                
        except Exception as e:
            print(f"    ⚠️  PyPDF image extraction failed: {e}")
        
        return image_paths
    
    def process_single_pdf(self, pdf_path: Path) -> Tuple[List[str], List[str], List[str]]:
        """
        Process a single PDF file and extract text, tables, and images
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (text_chunks, tables, image_paths)
        """
        print(f"📄 Processing PDF: {pdf_path.name}")
        
        # Create a subdirectory for this PDF's figures
        pdf_figures_path = self.figures_path / pdf_path.stem
        pdf_figures_path.mkdir(exist_ok=True)
        
        try:
            # Step 1: Extract tables first (without chunking to preserve table structure)
            print(f"  ⏳ Extracting tables and images from {pdf_path.name}...")
            loader = UnstructuredPDFLoader(
                file_path=str(pdf_path),
                strategy='hi_res',
                extract_images_in_pdf=True,
                infer_table_structure=True,
                mode='elements',
                image_output_dir_path=str(pdf_figures_path)
            )
            data = loader.load()
            print(f"  ✅ Extracted {len(data)} elements from {pdf_path.name}")
            
            # Extract tables
            tables_raw = [doc for doc in data if doc.metadata.get('category') == 'Table']
            print(f"  📊 Found {len(tables_raw)} tables")
            
            # Step 2: Extract text with chunking
            print(f"  ⏳ Chunking text from {pdf_path.name}...")
            loader = UnstructuredPDFLoader(
                file_path=str(pdf_path),
                strategy='hi_res',
                extract_images_in_pdf=True,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=4000,
                new_after_n_chars=4000,
                combine_text_under_n_chars=2000,
                mode='elements',
                image_output_dir_path=str(pdf_figures_path)
            )
            texts_raw = loader.load()
            print(f"  ✅ Created {len(texts_raw)} text chunks")
            
            # Combine data
            all_elements = texts_raw + tables_raw
            
            # Separate text and table elements
            text_docs = []
            table_docs = []
            
            for doc in all_elements:
                category = doc.metadata.get('category', '')
                if category == 'Table':
                    table_docs.append(doc)
                elif category in ['CompositeElement', 'NarrativeText', 'Title', 'ListItem']:
                    text_docs.append(doc)
            
            # Convert tables from HTML to Markdown
            tables_md = []
            for table in table_docs:
                if 'text_as_html' in table.metadata and table.metadata['text_as_html']:
                    try:
                        md_table = htmltabletomd.convert_table(table.metadata['text_as_html'])
                        tables_md.append(md_table)
                    except Exception as e:
                        print(f"Warning: Could not convert table to markdown: {e}")
                        tables_md.append(table.page_content)
                else:
                    tables_md.append(table.page_content)
            
            # Extract text content
            texts = [doc.page_content for doc in text_docs]
            
            # Get image paths from unstructured output
            image_paths = []
            if pdf_figures_path.exists():
                for img_file in sorted(pdf_figures_path.glob("*.jpg")):
                    image_paths.append(str(img_file))
            
            # If no images found, try extracting with PyPDF
            if len(image_paths) == 0:
                print(f"  🔍 No images found by unstructured, trying PyPDF extraction...")
                image_paths = self._extract_images_from_pdf(pdf_path, pdf_figures_path)
            else:
                print(f"  🖼️  Extracted {len(image_paths)} images via unstructured")
            
            print(f"  ✅ Completed {pdf_path.name}: {len(texts)} texts, {len(tables_md)} tables, {len(image_paths)} images")
            
            return texts, tables_md, image_paths
            
        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {str(e)}")
            return [], [], []
    
    def process_all_documents(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Process all PDF files in the dataset directory
        
        Returns:
            Tuple of (all_texts, all_tables, all_image_paths)
        """
        # Clear figures directory
        self._clear_figures_directory()
        
        all_texts = []
        all_tables = []
        all_images = []
        
        # Find all PDF files
        pdf_files = list(self.dataset_path.glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.dataset_path}")
        
        print(f"\n{'='*60}")
        print(f"📚 Found {len(pdf_files)} PDF files to process")
        print(f"{'='*60}\n")
        
        # Process each PDF
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"\n📖 Processing file {idx}/{len(pdf_files)}: {pdf_path.name}")
            print(f"{'-'*60}")
            texts, tables, images = self.process_single_pdf(pdf_path)
            all_texts.extend(texts)
            all_tables.extend(tables)
            all_images.extend(images)
            print(f"{'-'*60}\n")
        
        print(f"\n{'='*60}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📊 Total Results:")
        print(f"  • Text chunks: {len(all_texts)}")
        print(f"  • Tables: {len(all_tables)}")
        print(f"  • Images: {len(all_images)}")
        print(f"{'='*60}\n")
        
        return all_texts, all_tables, all_images


def get_mock_documents() -> Tuple[List[str], List[str], List[str]]:
    """
    Generate mock documents for testing without processing PDFs
    
    Returns:
        Tuple of (mock_texts, mock_tables, mock_images)
    """
    mock_texts = [
        """The Transformer model architecture is based on self-attention mechanisms. 
        It consists of an encoder and decoder, each composed of multiple identical layers. 
        The encoder maps an input sequence to a sequence of continuous representations, 
        which the decoder then uses to generate an output sequence one element at a time.""",
        
        """Attention mechanisms allow the model to focus on different parts of the input 
        when producing each part of the output. The multi-head attention mechanism runs 
        multiple attention operations in parallel, allowing the model to attend to 
        information from different representation subspaces.""",
        
        """Retrieval-Augmented Generation (RAG) combines the benefits of retrieval-based 
        and generation-based approaches. It uses a retriever to find relevant documents 
        and a generator to produce answers based on the retrieved context. This approach 
        enables the model to access external knowledge while generating responses.""",
        
        """GPT-3 demonstrates strong few-shot learning capabilities. By providing a few 
        examples in the prompt, the model can perform tasks without explicit fine-tuning. 
        This in-context learning ability emerges from training on diverse internet text.""",
        
        """Positional encoding is crucial in Transformers because the self-attention 
        mechanism doesn't inherently capture position information. Sinusoidal functions 
        are used to inject position information, allowing the model to understand the 
        order of tokens in the sequence."""
    ]
    
    mock_tables = [
        """| Model | Parameters | Performance |
|-------|------------|-------------|
| GPT-3 | 175B | 95.3% |
| BERT | 340M | 88.5% |
| T5 | 11B | 92.1% |""",
        
        """| Component | Description |
|-----------|-------------|
| Encoder | Processes input sequence |
| Decoder | Generates output sequence |
| Attention | Focus mechanism |"""
    ]
    
    # Mock images (empty list for debug mode)
    mock_images = []
    
    return mock_texts, mock_tables, mock_images
