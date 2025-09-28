import PyPDF2
import re
from typing import List, Dict
import logging

class DocumentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        # Fix common OCR issues
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\.\s+\.', '.', text)
        
        return text.strip()
    
    def split_into_sections(self, text: str, min_section_length: int = 100) -> List[Dict]:
        """
        Split text into logical sections based on headings
        """
        # Common HR policy headings pattern
        heading_pattern = r'\n\s*([A-Z][A-Z\s]{5,50}:?)\s*\n'
        
        sections = []
        matches = list(re.finditer(heading_pattern, text))
        
        if not matches:
            # If no clear headings, split by paragraphs
            paragraphs = text.split('\n\n')
            for i, para in enumerate(paragraphs):
                if len(para.strip()) > min_section_length:
                    sections.append({
                        'heading': f"Section {i+1}",
                        'content': para.strip(),
                        'section_id': i
                    })
            return sections
        
        # Split by headings
        for i, match in enumerate(matches):
            heading = match.group(1).strip()
            start_pos = match.end()
            
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            content = text[start_pos:end_pos].strip()
            
            if len(content) > min_section_length:
                sections.append({
                    'heading': heading,
                    'content': content,
                    'section_id': i
                })
        
        return sections

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    text = processor.extract_text_from_path("hr_policy.pdf")
    cleaned_text = processor.clean_text(text)
    sections = processor.split_into_sections(cleaned_text)
    print(f"Extracted {len(sections)} sections")