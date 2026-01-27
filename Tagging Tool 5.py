import pandas as pd
import requests
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import openai
from pydantic import BaseModel
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import traceback
import streamlit as st
from pathlib import Path
import yaml
import pickle
from dataclasses import dataclass
import numpy as np
import io
import re


def sanitize_for_excel(value):
    """
    Remove illegal characters that cannot be used in Excel worksheets.

    openpyxl raises IllegalCharacterError for control characters (ASCII 0-31)
    except for tab (9), newline (10), and carriage return (13).
    """
    if value is None:
        return value
    if not isinstance(value, str):
        return value

    # Remove illegal control characters (ASCII 0-8, 11-12, 14-31)
    # Keep tab (9), newline (10), carriage return (13)
    illegal_chars_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
    cleaned = illegal_chars_pattern.sub('', value)

    return cleaned


def sanitize_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize all string columns in a DataFrame to remove illegal Excel characters.
    Returns a copy of the DataFrame with sanitized values.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(sanitize_for_excel)
    return df_clean

@dataclass
class TaxonomyConfig:
    """Configuration for taxonomy structure"""
    categories: Dict[str, List[str]]
    descriptions: Dict[str, str]
    hierarchical: bool = False
    hierarchy_mapping: Dict[str, Dict[str, List[str]]] = None


class GenericTagger:
    def __init__(self, perplexity_api_key: str = None, openai_api_key: str = None, checkpoint_dir: str = "tagging_checkpoints"):
        """Initialize the GenericTagger with optional API keys"""
        self.perplexity_api_key = perplexity_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.taxonomy = None
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.file_lock = threading.Lock()
        
    def load_taxonomy_from_dict(self, taxonomy_dict: Dict[str, Any]) -> TaxonomyConfig:
        """Load taxonomy from a dictionary structure"""
        categories = taxonomy_dict.get('categories', {})
        descriptions = taxonomy_dict.get('descriptions', {})
        hierarchical = taxonomy_dict.get('hierarchical', False)
        hierarchy_mapping = taxonomy_dict.get('hierarchy_mapping', {})
        
        return TaxonomyConfig(
            categories=categories,
            descriptions=descriptions,
            hierarchical=hierarchical,
            hierarchy_mapping=hierarchy_mapping
        )
    
    def load_taxonomy_from_excel(self, file_path: str, sheet_name: str = None) -> TaxonomyConfig:
        """Load taxonomy from Excel file"""
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        
        categories = {}
        descriptions = {}
        
        if 'Category' in df.columns and 'Tag' in df.columns:
            for category in df['Category'].unique():
                if pd.notna(category):
                    category_tags = df[df['Category'] == category]['Tag'].dropna().tolist()
                    categories[category] = category_tags
                    
                    if 'Description' in df.columns:
                        for _, row in df[df['Category'] == category].iterrows():
                            if pd.notna(row.get('Description')):
                                descriptions[row['Tag']] = row['Description']
        else:
            categories['default'] = df.iloc[:, 0].dropna().tolist()
            if df.shape[1] > 1:
                for i, tag in enumerate(categories['default']):
                    if i < len(df) and pd.notna(df.iloc[i, 1]):
                        descriptions[tag] = df.iloc[i, 1]
        
        return TaxonomyConfig(categories=categories, descriptions=descriptions)
    
    def save_checkpoint(self, results: List[Dict], checkpoint_name: str):
        """Save checkpoint and automatically clean up old ones"""
        with self.file_lock:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Always save pickle (can handle empty data)
            checkpoint_path_pkl = self.checkpoint_dir / f"{checkpoint_name}_{timestamp}.pkl"
            with open(checkpoint_path_pkl, 'wb') as f:
                pickle.dump(results, f)

            # Only create Excel if there's data to write
            checkpoint_path_xlsx = self.checkpoint_dir / f"{checkpoint_name}_{timestamp}.xlsx"
            has_data = False

            try:
                if isinstance(results, dict):
                    # Check if any job has results
                    sheets_to_write = {}
                    for key, data in results.items():
                        if data and len(data) > 0:
                            df = pd.DataFrame(data)
                            if len(df) > 0:
                                sheet_name = f"{key[0]}_{key[1]}"[:31]
                                # Sanitize the dataframe to remove illegal characters
                                sheets_to_write[sheet_name] = sanitize_dataframe_for_excel(df)
                                has_data = True

                    # Only create Excel file if we have data
                    if has_data:
                        with pd.ExcelWriter(checkpoint_path_xlsx, engine='openpyxl') as writer:
                            for sheet_name, df in sheets_to_write.items():
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # Handle list of results
                    if results and len(results) > 0:
                        df = pd.DataFrame(results)
                        if len(df) > 0:
                            # Sanitize the dataframe to remove illegal characters
                            sanitized_df = sanitize_dataframe_for_excel(df)
                            sanitized_df.to_excel(checkpoint_path_xlsx, index=False)
                            has_data = True
            except Exception as e:
                # Log Excel checkpoint error but don't fail - pickle is the primary backup
                print(f"Warning: Excel checkpoint save failed: {e}")

            # Clean up old files
            all_pkl_files = sorted(self.checkpoint_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime)
            all_xlsx_files = sorted(self.checkpoint_dir.glob("*.xlsx"), key=lambda x: x.stat().st_mtime)

            for old_file in all_pkl_files[:-2]:
                try:
                    old_file.unlink()
                except Exception:
                    pass
            for old_file in all_xlsx_files[:-2]:
                try:
                    old_file.unlink()
                except Exception:
                    pass

            return checkpoint_path_pkl
    
    def load_checkpoint(self, checkpoint_path: str) -> List[Dict]:
        """Load results from a checkpoint file"""
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    
    def retry_with_exponential_backoff(self, func, max_retries: int = 3, base_delay: float = 1.0, 
                                     entity_name: str = None, progress_callback=None):
        """Retry a function with exponential backoff"""
        last_exception = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt)
                
                retry_msg = f"Retry {attempt + 1}/{max_retries} for {entity_name if entity_name else 'operation'} in {delay:.1f}s..."
                if progress_callback:
                    progress_callback(retry_msg)
                else:
                    print(retry_msg)
                
                time.sleep(delay)
        raise last_exception
    
    def search_entity_info(self, entity_name: str, entity_url: str = None, 
                        additional_context: str = "", max_retries: int = 3,
                        progress_callback=None, custom_prompt: str = None,
                        include_sources: bool = True, taxonomy_instructions: str = "") -> Tuple[str, bool]:
        """Search for entity information using Perplexity API"""
        if not self.perplexity_api_key:
            return f"No search performed for {entity_name} - Perplexity API key not provided", False
        
        def _search():
            if custom_prompt:
                query = f"""You are searching for information about {entity_name} to help answer the following analysis prompt:

ANALYSIS PROMPT: {custom_prompt}

Please find and provide information about {entity_name} that would be most relevant for answering the above prompt. Focus on:
1. Information directly relevant to the analysis prompt
2. Key facts and data points needed for the classification/analysis
3. Any specific aspects mentioned in the prompt

{additional_context}"""
            else:
                query = f"""Analyze {entity_name} and provide:
1. What they do or offer
2. Key characteristics and attributes
3. Industry or domain they operate in
{additional_context}"""
            
            if taxonomy_instructions:
                query += f"\n\nAdditional search guidance:\n{taxonomy_instructions}"
            
            if include_sources:
                query += "\n\nIMPORTANT: Please cite your sources by including [Source: URL or source name] after each key fact or piece of information."
            
            url = "https://api.perplexity.ai/chat/completions"
            
            headers = {
                'Authorization': f'Bearer {self.perplexity_api_key}',
                'Content-Type': 'application/json'
            }
            
            system_content = "You are an analyst. Provide detailed, factual descriptions based on available information. Focus on information that would be useful for the specific analysis requested."
            if include_sources:
                system_content += " Always cite your sources by including [Source: URL or source name] after each fact or claim."
            
            payload = {
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.2,
                "max_tokens": 700
            }
            
            if entity_url:
                clean_domain = entity_url.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
                payload["search_domain_filter"] = [clean_domain]
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 429:
                raise Exception("Perplexity API rate limit reached (429)")
            
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        try:
            description = self.retry_with_exponential_backoff(
                _search, 
                max_retries=max_retries,
                entity_name=entity_name,
                progress_callback=progress_callback
            )
            time.sleep(0.05)
            return description, True
        except Exception as e:
            error_msg = f"Error retrieving information for {entity_name}: {str(e)}"
            return error_msg, False
    
    def select_tags_with_ai(self, description: str, entity_name: str, 
                           available_tags: List[str], tag_descriptions: Dict[str, str],
                           multi_select: bool = False, existing_data: Dict = None,
                           custom_prompt: str = None, taxonomy_instructions: str = "") -> Dict:
        """Use AI to select appropriate tags or classify based on prompt using Responses API"""
        if not self.openai_client:
            return {'status': 'error', 'error': 'OpenAI client not initialized'}
        
        def _select_tags():
            context = ""
            if existing_data:
                context = "\n\nAdditional context from data:\n"
                context += "\n".join([f"{k}: {v}" for k, v in existing_data.items() if v])
            
            if custom_prompt and not available_tags:
                if multi_select:
                    class CustomPromptMultiOutput(BaseModel):
                        primary_result: str
                        secondary_results: List[str]
                        confidence: float
                        reasoning: str
                    
                    system_content = f"""{custom_prompt}

IMPORTANT: Focus only on the textual information provided. Ignore any URLs, website references, or source citations. Do not attempt to visit or access any websites.

Since multiple results are allowed, provide:
- primary_result: The main/most important classification or result
- secondary_results: A list of additional relevant classifications or results (can be empty list if only one result applies)

For the reasoning field, provide a brief explanation (3 sentences max) that includes:
1. Why you chose the primary result (and secondary results if any)
2. What specific information from the description influenced your decision
3. How any additional context factored into your decision"""
                    
                    user_content = f"Entity: {entity_name}\nDescription: {description}{context}"
                    
                    response = self.openai_client.responses.parse(
                        model="gpt-5.2",
                        input=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ],
                        text_format=CustomPromptMultiOutput
                    )
                    
                    parsed = response.output_parsed
                    
                    return {
                        'status': 'success',
                        'primary_tag': parsed.primary_result,
                        'secondary_tags': parsed.secondary_results,
                        'confidence': parsed.confidence,
                        'reasoning': parsed.reasoning
                    }
                else:
                    class CustomPromptOutput(BaseModel):
                        result: str
                        confidence: float
                        reasoning: str
                    
                    system_content = f"""{custom_prompt}

IMPORTANT: Focus only on the textual information provided. Ignore any URLs, website references, or source citations. Do not attempt to visit or access any websites.

For the reasoning field, provide a brief explanation (3 sentences max) that includes:
1. Why you chose this particular classification
2. What specific information from the description influenced your decision
3. How any additional context factored into your decision"""
                    
                    user_content = f"Entity: {entity_name}\nDescription: {description}{context}"
                    
                    response = self.openai_client.responses.parse(
                        model="gpt-5.2",
                        input=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ],
                        text_format=CustomPromptOutput
                    )
                    
                    parsed = response.output_parsed
                    
                    return {
                        'status': 'success',
                        'result': parsed.result,
                        'confidence': parsed.confidence,
                        'reasoning': parsed.reasoning
                    }
            
            tags_desc = "\n".join([
                f"- {tag}: {tag_descriptions.get(tag, 'No description available')}"
                for tag in available_tags
            ])
            
            if multi_select:
                class MultiTagOutput(BaseModel):
                    primary_tag: str
                    secondary_tags: List[str]
                    confidence: float
                    reasoning: str
                
                system_content = f"""You are an expert at classifying entities based on the following taxonomy.
Select multiple tags if appropriate, with one primary and optional secondary tags.

IMPORTANT: Focus only on the textual information provided. Ignore any URLs, website references, or source citations. Do not attempt to visit or access any websites.

Available tags:
{tags_desc}

Ensure your primary_tag and all secondary_tags are from the available tags list above.

For the reasoning field, provide a brief explanation (3 sentences max) that includes:
1. Why you chose the specific primary tag (and secondary tags if any)
2. What key information from the entity description influenced your decision
3. How any additional context data factored into your classification"""

                if taxonomy_instructions:
                    system_content += f"\n\nADDITIONAL CUSTOM INSTRUCTIONS:\n{taxonomy_instructions}\n\nPlease follow these custom instructions while still selecting from the available taxonomy tags."
                
                user_content = f"Entity: {entity_name}\nDescription: {description}{context}"
                
                response = self.openai_client.responses.parse(
                    model="gpt-5.2",
                    input=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    text_format=MultiTagOutput
                )
                
                parsed = response.output_parsed
                
                return {
                    'status': 'success',
                    'primary_tag': parsed.primary_tag.strip().rstrip(','),
                    'secondary_tags': [tag.strip().rstrip(',') for tag in parsed.secondary_tags],
                    'confidence': parsed.confidence,
                    'reasoning': parsed.reasoning
                }
            else:
                class SingleTagOutput(BaseModel):
                    selected_tag: str
                    confidence: float
                    reasoning: str
                
                system_content = f"""You are an expert at classifying entities based on the following taxonomy.
Select the single most appropriate tag.

IMPORTANT: Focus only on the textual information provided. Ignore any URLs, website references, or source citations. Do not attempt to visit or access any websites.

Available tags:
{tags_desc}

Ensure your selected_tag is from the available tags list above.

For the reasoning field, provide a brief explanation (3 sentences max) that includes:
1. Why you chose this specific tag over others
2. What key information from the entity description influenced your decision
3. How any additional context data factored into your classification"""

                if taxonomy_instructions:
                    system_content += f"\n\nADDITIONAL CUSTOM INSTRUCTIONS:\n{taxonomy_instructions}\n\nPlease follow these custom instructions while still selecting from the available taxonomy tags."
                
                user_content = f"Entity: {entity_name}\nDescription: {description}{context}"
                
                response = self.openai_client.responses.parse(
                    model="gpt-5.2",
                    input=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    text_format=SingleTagOutput
                )
                
                parsed = response.output_parsed
                
                return {
                    'status': 'success',
                    'tag': parsed.selected_tag.strip().rstrip(','),
                    'confidence': parsed.confidence,
                    'reasoning': parsed.reasoning
                }
        
        try:
            return self.retry_with_exponential_backoff(_select_tags)
        except Exception as e:
            error_msg = str(e)
            if "refusal" in error_msg.lower():
                return {'status': 'error', 'error': 'Model refused to respond for safety reasons'}
            return {'status': 'error', 'error': error_msg}
    
    def process_single_entity(self, row_data: Dict, config: Dict, progress_callback=None) -> Dict:
        """Process a single entity based on configuration"""
        try:
            entity_name = row_data.get(config['name_column'], 'Unknown')
            
            context_columns = config.get('context_columns', [])
            context_data = {col: row_data.get(col) for col in context_columns if col in row_data}
            
            taxonomy_instructions = ""
            if config.get('use_taxonomy', True):
                taxonomy_instructions = config.get('taxonomy_custom_instructions', '')
            
            context_parts = [f"{k}: {v}" for k, v in context_data.items() if v]
            additional_context = ""
            if context_parts:
                additional_context = "Additional context:\n" + "\n".join(context_parts)
            
            custom_prompt = config.get('custom_prompt', None)
            
            search_success = True
            if config['use_search'] and self.perplexity_api_key:
                url_column = config.get('url_column')
                entity_url = row_data.get(url_column) if url_column else None
                
                max_retries = config.get('search_max_retries', 3)
                
                description, search_success = self.search_entity_info(
                    entity_name, 
                    entity_url,
                    additional_context=additional_context,
                    max_retries=max_retries,
                    progress_callback=progress_callback,
                    custom_prompt=custom_prompt,
                    include_sources=True,
                    taxonomy_instructions=taxonomy_instructions
                )
                
                if not search_success:
                    result = row_data.copy()
                    result.update({
                        'Search_Description': description,
                        'Tagged_Result': 'Search Failed',
                        'Confidence': '0%',
                        'Reasoning': 'Perplexity search failed - cannot tag without description',
                        'Status': 'Search Error'
                    })
                    return result
            else:
                desc_columns = config.get('description_columns', [])
                description_parts = []
                for col in desc_columns:
                    if col in row_data and row_data[col]:
                        description_parts.append(f"{col}: {row_data[col]}")
                description = "\n".join(description_parts) if description_parts else f"No description available for {entity_name}"
            
            use_taxonomy = config.get('use_taxonomy', True)
            custom_prompt = config.get('custom_prompt', None)
            
            # Get taxonomy from config instead of self.taxonomy
            taxonomy = config.get('taxonomy')
            
            if use_taxonomy and taxonomy:
                if config.get('category_column') and taxonomy.categories:
                    category = row_data.get(config['category_column'], 'default')
                    available_tags = taxonomy.categories.get(category, 
                                                                taxonomy.categories.get('default', []))
                else:
                    all_tags = []
                    for tags in taxonomy.categories.values():
                        all_tags.extend(tags)
                    available_tags = list(set(all_tags))
                
                tag_descriptions = taxonomy.descriptions
            else:
                available_tags = []
                tag_descriptions = {}
            
            tag_result = self.select_tags_with_ai(
                description=description,
                entity_name=entity_name,
                available_tags=available_tags,
                tag_descriptions=tag_descriptions,
                multi_select=config.get('multi_select', False),
                existing_data=context_data,
                custom_prompt=custom_prompt,
                taxonomy_instructions=taxonomy_instructions
            )
            
            result = row_data.copy()
            
            if config['use_search']:
                result['Search_Description'] = description[:500]
            
            if tag_result['status'] == 'error':
                result.update({
                    'Tagged_Result': 'Error',
                    'Confidence': '0%',
                    'Reasoning': tag_result['error'],
                    'Status': 'Error'
                })
            else:
                if 'result' in tag_result:
                    result.update({
                        'Tagged_Result': tag_result['result'],
                        'Confidence': f"{tag_result['confidence']:.0%}",
                        'Reasoning': tag_result['reasoning'],
                        'Status': 'Success'
                    })
                elif config.get('multi_select'):
                    secondary_tags = tag_result.get('secondary_tags', [])
                    if isinstance(secondary_tags, list):
                        secondary_tags_str = '; '.join(secondary_tags) if secondary_tags else ''
                    else:
                        secondary_tags_str = str(secondary_tags)
                    
                    result.update({
                        'Primary_Tag': tag_result['primary_tag'],
                        'Secondary_Tags': secondary_tags_str,
                        'Confidence': f"{tag_result['confidence']:.0%}",
                        'Reasoning': tag_result['reasoning'],
                        'Status': 'Success'
                    })
                else:
                    tag_value = tag_result.get('tag', '')
                    if isinstance(tag_value, str):
                        tag_value = tag_value.strip().rstrip(',')
                    
                    result.update({
                        'Tagged_Result': tag_value,
                        'Confidence': f"{tag_result['confidence']:.0%}",
                        'Reasoning': tag_result['reasoning'],
                        'Status': 'Success'
                    })
            
            return result
            
        except Exception as e:
            result = row_data.copy()
            result.update({
                'Tagged_Result': 'Error',
                'Confidence': '0%',
                'Reasoning': str(e),
                'Status': 'Error'
            })
            return result


def validate_custom_instructions(instructions: str) -> Optional[str]:
    """Validate custom instructions for taxonomy mode"""
    if len(instructions) > 1000:
        return "Instructions too long (maximum 1000 characters)"
    
    forbidden_phrases = [
        "ignore taxonomy", "ignore the taxonomy", "create new tags", "make up tags",
        "add new tags", "invent tags", "don't use taxonomy", "bypass taxonomy"
    ]
    
    instructions_lower = instructions.lower()
    for phrase in forbidden_phrases:
        if phrase in instructions_lower:
            return f"Instructions cannot override core taxonomy functionality (found: '{phrase}')"
    
    return None


PRESET_PROMPTS = {
    "Industry Classification": """You are an expert at classifying companies by industry. Analyze the provided information and classify the entity into one of these industries: Technology, Healthcare, Finance, Retail, Manufacturing, Services, Education, Real Estate, Energy, Transportation, Media/Entertainment, or Other. Provide your answer in the format: "Industry: [classification]" followed by a brief explanation.""",
    
    "Company Size": """Based on the provided information, classify this company's size as: Startup (1-50 employees), Small (51-200), Medium (201-1000), Large (1001-5000), or Enterprise (5000+). Provide your answer in the format: "Size: [classification]" with reasoning.""",
    
    "B2B vs B2C": """Analyze the entity and determine if it primarily serves Business customers (B2B), Consumer customers (B2C), or Both (B2B/B2C). Provide your answer in the format: "Type: [classification]" with explanation.""",
    
    "Technology Level": """Assess the technology sophistication of this entity. Classify as: High-Tech (cutting-edge technology focus), Tech-Enabled (uses technology significantly), Traditional Tech (basic technology use), or Non-Tech. Format: "Tech Level: [classification]" with reasoning.""",
    
    "Market Maturity": """Evaluate the entity's market position and maturity. Classify as: Early Stage (new/unproven), Growth Stage (expanding rapidly), Mature (established market position), or Declining. Format: "Maturity: [classification]" with explanation.""",
    
    "Custom Analysis": """[This is a placeholder - users should write their own prompt for custom analysis]"""
}


def create_tag_categories_helper(key_suffix=""):
    """Create the 'Select this if...' helper text and popup functionality"""
    popup_key = f"show_categories_popup_{key_suffix}"
    if popup_key not in st.session_state:
        st.session_state[popup_key] = False
    
    if st.button("*Select this if...*", key=f"categories_help_{key_suffix}", 
                help="Click for explanation of tag categories",
                type="secondary"):
        st.session_state[popup_key] = True
    
    if st.session_state[popup_key]:
        with st.expander("Tag Categories Explanation", expanded=True):
            col1, col2 = st.columns([10, 1])
            with col2:
                if st.button("✕", key=f"close_popup_{key_suffix}", help="Close"):
                    st.session_state[popup_key] = False
                    st.rerun()
            
            with col1:
                st.markdown("""
                <div style="font-size: 16px; color: #333; margin-bottom: 15px;">
                Select this if your tags are grouped by category. Each entity will be restricted to the tags in its category. E.g.
                </div>
                """, unsafe_allow_html=True)
                
                categorized_data = {
                    "Category": ["Technology", "Technology", "Healthcare", "Healthcare", "Finance", "Finance"],
                    "Tag": ["Software", "Hardware", "Pharmaceutical", "Medical Devices", "Banking", "Fintech"],
                    "Description": [
                        "Companies that develop software products",
                        "Companies that manufacture hardware", 
                        "Drug development and manufacturing",
                        "Medical equipment manufacturers",
                        "Traditional banking services",
                        "Financial technology companies"
                    ]
                }
                categorized_df = pd.DataFrame(categorized_data)
                
                st.markdown("""
                <style>
                .small-table table {
                    font-size: 12px !important;
                }
                .small-table td, .small-table th {
                    font-size: 12px !important;
                    padding: 4px 8px !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="small-table">', unsafe_allow_html=True)
                st.table(categorized_df)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div style="font-size: 16px; color: #333; margin: 15px 0;">
                As opposed to the simpler:
                </div>
                """, unsafe_allow_html=True)
                
                simple_data = {
                    "Tag": ["Software", "Hardware", "Consulting"],
                    "Description": [
                        "Companies that develop software",
                        "Companies that manufacture hardware",
                        "Professional services companies"
                    ]
                }
                simple_df = pd.DataFrame(simple_data)
                
                st.markdown('<div class="small-table">', unsafe_allow_html=True)
                st.table(simple_df)
                st.markdown('</div>', unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables"""
    if 'tagger' not in st.session_state:
        st.session_state.tagger = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'sheet_data' not in st.session_state:
        st.session_state.sheet_data = {}
    if 'sheet_configs' not in st.session_state:
        st.session_state.sheet_configs = {}
    if 'tagging_configs' not in st.session_state:
        st.session_state.tagging_configs = {}


def create_column_config(df, sheet_key):
    """Create column configuration UI for a single sheet"""
    columns = df.columns.tolist()
    
    use_search = st.checkbox(
        "Use web search (Perplexity)", 
        value=False,
        key=f"use_search_{sheet_key}"
    )
    
    name_column = st.selectbox(
        "Entity name column", 
        columns,
        help="Column containing the names to tag",
        key=f"name_col_{sheet_key}"
    )
    
    url_column = None
    description_columns = []
    
    if use_search:
        url_column = st.selectbox(
            "URL column (optional)", 
            ['None'] + columns,
            key=f"url_col_{sheet_key}"
        )
        url_column = None if url_column == 'None' else url_column
    else:
        description_columns = st.multiselect(
            "Description columns",
            columns,
            help="Columns containing descriptions",
            key=f"desc_cols_{sheet_key}"
        )
    
    context_columns = st.multiselect(
        "Context columns (optional)",
        [c for c in columns if c != name_column],
        help="Additional context for tagging",
        key=f"context_cols_{sheet_key}"
    )
    
    category_column = st.selectbox(
        "Category column (optional)",
        ['None'] + columns,
        help="If your taxonomy has categories",
        key=f"category_col_{sheet_key}"
    )
    category_column = None if category_column == 'None' else category_column
    
    return {
        'name_column': name_column,
        'use_search': use_search,
        'url_column': url_column,
        'description_columns': description_columns,
        'context_columns': context_columns,
        'category_column': category_column,
        'multi_select': False
    }


def create_data_input_tab():
    """Create the Data Input tab with multi-file/sheet support
    
    Note: This function handles cleanup of removed sheets:
    - When files are removed/cleared, all associated configs are deleted
    - When sheets are deselected, their configs are removed from tagging_configs and sheet_configs
    """
    st.header("📁 Data Input")
    
    st.markdown("""
    <div style="font-size: 16px; color: #333; margin-bottom: 20px;">
    <strong>Input one Excel file containing your entities to be tagged (and any other additional columns of information). Each sheet of input data should be formatted with column titles starting in cell A1. Input multiple Excel files or select multiple sheets if you want to tag more than one set of data at once.</strong>
    <br><br>
    The tool will output your original sheet with an added column containing tags (will add multiple columns containing tags if you set up and run multiple tags at once).
    <br><br>
    <strong>Data Input columns:</strong>
    <ul>
        <li><strong>Entity to be tagged</strong> (required)</li>
        <li><strong>'Description' column:</strong> Only applicable if not using Web Search</li>
        <li><strong>'URL' column:</strong> Optional, only applicable if using Web Search</li>
        <li><strong>'Context' column(s):</strong> Optional, extra fields that help improve search and tagging</li>
        <li><strong>Category column:</strong> Only if using tag categories</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload Excel/CSV file(s)", 
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if not uploaded_files:
        # Clear all data if no files uploaded
        if st.session_state.sheet_data or st.session_state.sheet_configs or st.session_state.tagging_configs:
            st.session_state.sheet_data = {}
            st.session_state.sheet_configs = {}
            st.session_state.tagging_configs = {}
            st.info("All data cleared. Please upload files to continue.")
        return
    
    if uploaded_files:
        st.session_state.uploaded_files = {f.name: f for f in uploaded_files}
        
        file_sheet_options = {}
        for filename, file in st.session_state.uploaded_files.items():
            if filename.endswith('.csv'):
                file_sheet_options[filename] = ['main']
            else:
                excel_file = pd.ExcelFile(file)
                file_sheet_options[filename] = excel_file.sheet_names
        
        st.subheader("Select Sheet(s)")
        selected_sheets = {}
        
        for filename, sheets in file_sheet_options.items():
            if len(sheets) == 1:
                selected_sheets[filename] = sheets
                st.info(f"📄 **{filename}**: {sheets[0]} (auto-selected)")
            else:
                selected = st.multiselect(
                    f"Select sheet(s) from **{filename}**",
                    sheets,
                    default=sheets[:1],
                    key=f"sheet_select_{filename}"
                )
                if selected:
                    selected_sheets[filename] = selected
        
        total_sheets = sum(len(sheets) for sheets in selected_sheets.values())
        
        if total_sheets == 0:
            st.warning("Please select at least one sheet to proceed.")
            return
        
        st.success(f"✅ {total_sheets} sheet(s) selected from {len(selected_sheets)} file(s)")
        
        # Clear sheet_data and rebuild based on current selection
        st.session_state.sheet_data = {}
        
        for filename, sheets in selected_sheets.items():
            file = st.session_state.uploaded_files[filename]
            for sheet_name in sheets:
                key = (filename, sheet_name)
                
                if filename.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file, sheet_name=sheet_name)
                
                st.session_state.sheet_data[key] = df
        
        # Clean up removed sheets from all session state
        current_keys = set(st.session_state.sheet_data.keys())
        all_keys = set()
        all_keys.update(st.session_state.sheet_configs.keys())
        all_keys.update(st.session_state.tagging_configs.keys())
        
        removed_keys = all_keys - current_keys
        if removed_keys:
            for removed_key in removed_keys:
                if removed_key in st.session_state.sheet_configs:
                    del st.session_state.sheet_configs[removed_key]
                if removed_key in st.session_state.tagging_configs:
                    del st.session_state.tagging_configs[removed_key]
            
            removed_sheet_names = [f"{filename}/{sheet_name}" for filename, sheet_name in removed_keys]
            st.info(f"🗑️ Cleaned up configurations for removed sheets: {', '.join(removed_sheet_names)}")
        
        st.header("📊 Column Configuration")
        
        if total_sheets == 1:
            key = list(st.session_state.sheet_data.keys())[0]
            filename, sheet_name = key
            df = st.session_state.sheet_data[key]
            
            st.subheader(f"Configure: {filename} - {sheet_name}")
            config = create_column_config(df, key)
            st.session_state.sheet_configs[key] = config
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.info(f"Showing first 10 rows of {len(df)} total rows")
            
        else:
            for key in st.session_state.sheet_data.keys():
                filename, sheet_name = key
                df = st.session_state.sheet_data[key]
                
                with st.expander(f"Configure input data: {sheet_name}", expanded=False):
                    config = create_column_config(df, key)
                    st.session_state.sheet_configs[key] = config
                    
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    st.info(f"Showing first 10 rows of {len(df)} total rows")


def create_taxonomy_config(sheet_key, config_idx, config):
    """Create taxonomy configuration UI"""
    
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        st.subheader("📋 Taxonomy Setup")
    with col2:
        use_categories = st.checkbox(
            "Use tag categories", 
            key=f"use_cat_{sheet_key}_{config_idx}"
        )
    with col3:
        create_tag_categories_helper(f"taxonomy_{sheet_key}_{config_idx}")
    
    taxonomy_method = st.selectbox(
        "How would you like to provide the taxonomy?",
        ["Upload Excel file", "Paste as text", "Enter manually"],
        key=f"tax_method_{sheet_key}_{config_idx}"
    )
    
    if taxonomy_method == "Upload Excel file":
        taxonomy_file = st.file_uploader(
            "Upload taxonomy file", 
            type=['xlsx', 'xls'],
            key=f"tax_file_{sheet_key}_{config_idx}"
        )
        if taxonomy_file:
            tax_sheet = None
            if taxonomy_file.name.endswith(('.xlsx', '.xls')):
                tax_excel = pd.ExcelFile(taxonomy_file)
                if len(tax_excel.sheet_names) > 1:
                    tax_sheet = st.selectbox("Select taxonomy sheet", 
                                            tax_excel.sheet_names,
                                            key=f"tax_sheet_{sheet_key}_{config_idx}")
            
            taxonomy = st.session_state.tagger.load_taxonomy_from_excel(taxonomy_file, tax_sheet)
            config['taxonomy'] = taxonomy
            st.success("✅ Taxonomy loaded!")
            
            with st.expander("View loaded taxonomy"):
                for category, tags in taxonomy.categories.items():
                    st.write(f"**{category}**")
                    for tag in tags:
                        desc = taxonomy.descriptions.get(tag, "")
                        st.write(f"- {tag}: {desc}" if desc else f"- {tag}")
        
        st.subheader("📋 Excel Taxonomy Formatting")
        
        if use_categories:
            st.markdown("Starting in cell A1, three columns: **Category**, **Tag**, and **Description** of tag.")
            st.markdown("**Excel Taxonomy Formatting - Example**")
            
            categorized_data = {
                "Category": ["Technology", "Technology", "Healthcare", "Healthcare", "Finance", "Finance"],
                "Tag": ["Software", "Hardware", "Pharmaceutical", "Medical Devices", "Banking", "Fintech"],
                "Description": [
                    "Companies that develop software products",
                    "Companies that manufacture hardware", 
                    "Drug development and manufacturing",
                    "Medical equipment manufacturers",
                    "Traditional banking services",
                    "Financial technology companies"
                ]
            }
            categorized_df = pd.DataFrame(categorized_data)
            
            st.markdown("""
            <style>
            .small-table table {
                font-size: 12px !important;
            }
            .small-table td, .small-table th {
                font-size: 12px !important;
                padding: 4px 8px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="small-table">', unsafe_allow_html=True)
            st.table(categorized_df)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown("Starting in cell A1, two columns: **Tag** and **Description** of tag.")
            st.markdown("**Excel Taxonomy Formatting - Example**")
            
            simple_data = {
                "Tag": ["Software", "Hardware", "Consulting"],
                "Description": [
                    "Companies that develop software",
                    "Companies that manufacture hardware",
                    "Professional services companies"
                ]
            }
            simple_df = pd.DataFrame(simple_data)
            
            st.markdown("""
            <style>
            .small-table table {
                font-size: 12px !important;
            }
            .small-table td, .small-table th {
                font-size: 12px !important;
                padding: 4px 8px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="small-table">', unsafe_allow_html=True)
            st.table(simple_df)
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif taxonomy_method == "Paste as text":
        st.markdown("Paste your taxonomy in YAML format:")
        
        if use_categories:
            taxonomy_label = "Taxonomy"
            taxonomy_placeholder = """categories:
  Category1:
    - Tag1
    - Tag2
  Category2:
    - Tag3
    - Tag4
descriptions:
  Tag1: "Description of Tag1"
  Tag2: "Description of Tag2"
  Tag3: "Description of Tag3"
  Tag4: "Description of Tag4"
"""
        else:
            taxonomy_label = "Taxonomy\n\nSince tags are not categorized, there is one category of tags called 'default'."
            taxonomy_placeholder = """categories:
  default:
    - Tag1
    - Tag2
    - Tag3
descriptions:
  Tag1: "Description of Tag1"
  Tag2: "Description of Tag2"
  Tag3: "Description of Tag3"
"""
        
        taxonomy_text = st.text_area(taxonomy_label, height=300, 
                                placeholder=taxonomy_placeholder,
                                key=f"tax_text_{sheet_key}_{config_idx}")
        
        if st.button("Parse Taxonomy", key=f"parse_tax_{sheet_key}_{config_idx}"):
            try:
                taxonomy_dict = yaml.safe_load(taxonomy_text)
                taxonomy = st.session_state.tagger.load_taxonomy_from_dict(taxonomy_dict)
                config['taxonomy'] = taxonomy
                st.success("✅ Taxonomy parsed successfully!")
            except Exception as e:
                st.error(f"Failed to parse YAML taxonomy: {str(e)}")
    
    else:
        st.markdown("Enter tags manually:")
        
        categories = {}
        descriptions = {}
        
        if use_categories:
            num_categories = st.number_input("Number of categories", 1, 10, 1, 
                                           key=f"num_cat_{sheet_key}_{config_idx}")
            
            for i in range(num_categories):
                with st.expander(f"Category {i+1}", expanded=i==0):
                    cat_name = st.text_input(f"Category name", "default", 
                                           key=f"cat_{sheet_key}_{config_idx}_{i}")
                    tags_text = st.text_area(
                        "Tags (one per line)", 
                        key=f"tags_{sheet_key}_{config_idx}_{i}",
                        help="Enter each tag on a new line"
                    )
                    st.markdown("*You will enter descriptions for each tag after entering the list of tags themselves and clicking \"Create Taxonomy\".*")
                    
                    if tags_text:
                        tags = [t.strip() for t in tags_text.split('\n') if t.strip()]
                        categories[cat_name] = tags
                        
                        if st.checkbox("Add descriptions", key=f"desc_check_{sheet_key}_{config_idx}_{i}"):
                            for tag in tags:
                                desc = st.text_input(f"Description for '{tag}'", 
                                                key=f"desc_{sheet_key}_{config_idx}_{i}_{tag}")
                                if desc:
                                    descriptions[tag] = desc
        else:
            tags_text = st.text_area(
                "Tags (one per line)", 
                key=f"tags_simple_{sheet_key}_{config_idx}",
                help="Enter each tag on a new line"
            )
            st.markdown("*You will enter descriptions for each tag after entering the list of tags themselves and clicking \"Create Taxonomy\".*")
            
            if tags_text:
                tags = [t.strip() for t in tags_text.split('\n') if t.strip()]
                categories["default"] = tags
                
                if st.checkbox("Add descriptions", key=f"desc_check_simple_{sheet_key}_{config_idx}"):
                    for tag in tags:
                        desc = st.text_input(f"Description for '{tag}'", 
                                        key=f"desc_simple_{sheet_key}_{config_idx}_{tag}")
                        if desc:
                            descriptions[tag] = desc
        
        if st.button("Create Taxonomy", key=f"create_tax_{sheet_key}_{config_idx}"):
            taxonomy = TaxonomyConfig(
                categories=categories,
                descriptions=descriptions
            )
            config['taxonomy'] = taxonomy
            st.success("✅ Taxonomy created!")
    
    st.subheader("Add custom instructions (optional)")
    st.markdown("<small>Instructions for search behavior and tagging decisions</small>", unsafe_allow_html=True)
    
    custom_instructions = st.text_area(
        "Custom instructions",
        value=config.get('custom_instructions', ''),
        height=150,
        key=f"custom_inst_{sheet_key}_{config_idx}",
        placeholder="""Examples:
- Prioritize more specific manufacturing tags over general "manufacturing" tag
- Base classification on company's end-customers rather than what the company does""",
        help="Optional instructions to guide search and tagging"
    )
    
    if custom_instructions:
        validation_error = validate_custom_instructions(custom_instructions)
        if validation_error:
            st.error(validation_error)
            config['custom_instructions'] = ""
        else:
            config['custom_instructions'] = custom_instructions
            st.info(f"✅ Custom instructions added ({len(custom_instructions)} characters)")
    else:
        config['custom_instructions'] = ""


def create_custom_prompt_config(sheet_key, config_idx, config):
    """Create custom prompt configuration UI"""
    
    st.subheader("✍️ Prompt Configuration")
    st.markdown("<small>Custom queries method is recommended only if required, i.e. if you do not have a complete list of tags for your task. If you have a list of tags, Taxonomy method is recommended for optimal results.</small>", unsafe_allow_html=True)
    
    preset_choice = st.selectbox(
        "Choose a preset prompt or create custom:",
        list(PRESET_PROMPTS.keys()),
        key=f"preset_{sheet_key}_{config_idx}"
    )
    
    if preset_choice == "Custom Analysis":
        custom_prompt = st.text_area(
            "Enter your custom prompt:",
            value=config.get('custom_prompt', ''),
            height=200,
            key=f"prompt_{sheet_key}_{config_idx}",
            placeholder="Example: Analyze and classify this entity..."
        )
    else:
        custom_prompt = st.text_area(
            "Prompt (you can edit):",
            value=PRESET_PROMPTS[preset_choice],
            height=200,
            key=f"prompt_{sheet_key}_{config_idx}"
        )
    
    config['custom_prompt'] = custom_prompt
    
    with st.expander("💡 Prompt Tips"):
        st.markdown("""
        **Tips for writing effective prompts:**
        - Be specific about what you want to classify or analyze
        - Define clear categories or output format
        - Include any specific criteria for classification
        - Ask for reasoning to understand the AI's decision
        
        **Example format:**
        "You are an expert at [domain]. Analyze the entity and classify it as [categories]. 
        Consider [specific factors]. Provide your answer as '[Output format]' with explanation."
        """)


def create_tagging_method_tab():
    """Create the Tagging Method tab with multi-config support"""
    st.header("🎯 Tagging Method")
    
    st.info("ℹ️ For each sheet, tool will output the original sheet plus an additional column with tagged results. Multiple configurations for one sheet will output one sheet plus an additional column with tagged results for each configuration.")
    
    if not st.session_state.sheet_data:
        st.warning("Please upload and configure data in the Data Input tab first.")
        return
    
    # Clean up configs for sheets that no longer exist
    current_sheet_keys = set(st.session_state.sheet_data.keys())
    config_keys = set(st.session_state.tagging_configs.keys())
    orphaned_keys = config_keys - current_sheet_keys
    
    if orphaned_keys:
        for orphaned_key in orphaned_keys:
            del st.session_state.tagging_configs[orphaned_key]
        st.info(f"🗑️ Cleaned up {len(orphaned_keys)} configuration(s) for removed sheets")
    
    for sheet_key in st.session_state.sheet_data.keys():
        if sheet_key not in st.session_state.tagging_configs:
            st.session_state.tagging_configs[sheet_key] = []
    
    all_configs = []
    for sheet_key in st.session_state.sheet_data.keys():
        filename, sheet_name = sheet_key
        configs = st.session_state.tagging_configs[sheet_key]
        
        if len(configs) == 0:
            st.session_state.tagging_configs[sheet_key].append({
                'config_num': 1,
                'method': 'Use Taxonomy',
                'taxonomy': None,
                'custom_prompt': None,
                'custom_instructions': ''
            })
            configs = st.session_state.tagging_configs[sheet_key]
        
        for idx, config in enumerate(configs):
            all_configs.append({
                'sheet_key': sheet_key,
                'sheet_name': sheet_name,
                'config_idx': idx,
                'config': config
            })
    
    for item in all_configs:
        sheet_key = item['sheet_key']
        sheet_name = item['sheet_name']
        config_idx = item['config_idx']
        config = item['config']
        config_num = config['config_num']
        
        title = f"Configure tagging method {config_num} for {sheet_name}" if config_num > 1 else f"Configure tagging method for {sheet_name}"
        
        with st.expander(title, expanded=(len(all_configs) == 1)):
            
            method = st.selectbox(
                "Choose tagging method:",
                ["Use Taxonomy", "Use Custom Prompt"],
                key=f"method_{sheet_key}_{config_idx}",
                index=0 if config['method'] == 'Use Taxonomy' else 1
            )
            config['method'] = method
            
            if method == "Use Taxonomy":
                create_taxonomy_config(sheet_key, config_idx, config)
            else:
                create_custom_prompt_config(sheet_key, config_idx, config)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"➕ Add another tagging configuration for {sheet_name}", 
                           key=f"add_config_{sheet_key}_{config_idx}"):
                    configs = st.session_state.tagging_configs[sheet_key]
                    st.session_state.tagging_configs[sheet_key].append({
                        'config_num': len(configs) + 1,
                        'method': 'Use Taxonomy',
                        'taxonomy': None,
                        'custom_prompt': None,
                        'custom_instructions': ''
                    })
                    st.rerun()
            
            with col2:
                if config_num > 1:
                    if st.button(f"🗑️ Remove this configuration", 
                               key=f"remove_{sheet_key}_{config_idx}"):
                        st.session_state.tagging_configs[sheet_key].pop(config_idx)
                        st.rerun()


def create_row_selection_ui(job, key_suffix):
    """Create row selection UI for a tagging job"""
    df = job['df']
    sheet_config = job['sheet_config']
    tagging_config = job.get('tagging_config', {})
    
    # Add multi-select checkbox for all configurations
    st.subheader("Tagging Options")
    multi_select = st.checkbox(
        "Allow multiple tags per entity",
        value=job.get('multi_select', False),
        key=f"multi_select_{key_suffix}",
        help="When enabled, the model can assign multiple tags (one primary and multiple secondary tags) to each entity"
    )
    job['multi_select'] = multi_select
    
    st.subheader("Select rows to process")
    
    run_test = st.checkbox("Run a Test", key=f"test_{key_suffix}")
    st.markdown("<small>Tags first 10 rows only, to preview output (recommended before processing full dataset)</small>", unsafe_allow_html=True)
    
    if run_test:
        job['selected_rows'] = df.head(10)
        st.info(f"Test mode: Will process first {len(job['selected_rows'])} rows")
        return
    
    choose_rows = st.checkbox("Choose rows to process", value=False, key=f"choose_{key_suffix}")
    
    if not choose_rows:
        job['selected_rows'] = df
        return
    
    has_category_column = sheet_config.get('category_column')
    if has_category_column:
        st.markdown("<small>Tool will process all rows you select using \"Select range\", \"Select rows\", and \"Select categories\".</small>", unsafe_allow_html=True)
    else:
        st.markdown("<small>Tool will process all rows you select using \"Select range\" and \"Select rows\". (Optionally, configure a Category column in the Data Input tab in order to select rows by category)</small>", unsafe_allow_html=True)
    
    selected_indices = set()
    
    select_range = st.checkbox("Select range", key=f"range_{key_suffix}", help="Select a continuous range of rows")
    if select_range:
        col1, col2 = st.columns(2)
        with col1:
            start_row = st.number_input("Starting row", 1, len(df), 1, key=f"start_{key_suffix}")
        with col2:
            end_row = st.number_input("End row", start_row, len(df), min(10, len(df)), key=f"end_{key_suffix}")
        
        if start_row <= end_row:
            selected_indices.update(range(start_row - 1, end_row))
    
    select_manual = st.checkbox("Select rows", key=f"manual_{key_suffix}", help="Manually select specific row indices")
    if select_manual:
        st.markdown("""
        <small>
        • Supports individual numbers: 1,3,5,10<br>
        • Supports ranges: 5-8 (includes rows 5,6,7,8)<br>
        • Supports combinations: 1,3,5-8,10
        </small>
        """, unsafe_allow_html=True)
        
        manual_rows = st.text_input("Row numbers (e.g., 1,3,5-8,10)", key=f"manual_input_{key_suffix}")
        if manual_rows:
            try:
                for part in manual_rows.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        if 1 <= start <= end <= len(df):
                            selected_indices.update(range(start - 1, end))
                    else:
                        row_num = int(part)
                        if 1 <= row_num <= len(df):
                            selected_indices.add(row_num - 1)
            except:
                st.error("Invalid format. Use comma-separated numbers or ranges (e.g., 1,3,5-8,10)")
    
    if has_category_column:
        select_categories = st.checkbox("Select categories", key=f"cat_{key_suffix}", help="Select rows by category values")
        if select_categories:
            categories = df[sheet_config['category_column']].unique()
            selected_cats = st.multiselect("Categories", categories, key=f"cats_{key_suffix}", help="Select categories to include")
            if selected_cats:
                mask = df[sheet_config['category_column']].isin(selected_cats)
                selected_indices.update(df[mask].index)
    
    if selected_indices:
        job['selected_rows'] = df.loc[sorted(selected_indices)]
    else:
        job['selected_rows'] = df.iloc[0:0]
    
    st.info(f"Selected {len(job.get('selected_rows', df))} rows")


def run_concurrent_tagging(tagging_jobs, max_workers, batch_size, search_retries):
    """Run all tagging jobs concurrently"""
    st.session_state.processing = True
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    all_tasks = []
    for job_idx, job in enumerate(tagging_jobs):
        df = job['selected_rows']
        sheet_config = job['sheet_config']
        tagging_config = job['tagging_config']
        
        tagger_config = {
            **sheet_config,
            'use_taxonomy': tagging_config['method'] == 'Use Taxonomy',
            'custom_prompt': tagging_config.get('custom_prompt'),
            'taxonomy_custom_instructions': tagging_config.get('custom_instructions', ''),
            'search_max_retries': search_retries,
            'multi_select': job.get('multi_select', False),
            'taxonomy': tagging_config.get('taxonomy')  # Pass taxonomy in config
        }
        
        for idx, row in df.iterrows():
            all_tasks.append({
                'job_idx': job_idx,
                'job': job,
                'idx': idx,
                'row': row,
                'config': tagger_config
            })
    
    results_by_job = {i: [] for i in range(len(tagging_jobs))}
    completed = 0
    total_tasks = len(all_tasks)
    
    if total_tasks == 0:
        st.error("No tasks to process. Please select rows to tag.")
        st.session_state.processing = False
        return
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                st.session_state.tagger.process_single_entity,
                task['row'].to_dict(),
                task['config']
            ): task
            for task in all_tasks
        }
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            job_idx = task['job_idx']
            
            try:
                result = future.result()
                result['_original_index'] = task['idx']
                results_by_job[job_idx].append(result)
                
                completed += 1
                progress = min(completed / total_tasks, 1.0)  # Clamp to 1.0 to avoid exceeding bounds
                progress_bar.progress(progress)
                
                entity_name = result.get(task['config']['name_column'], 'Unknown')
                status_text.text(f"Processing {completed}/{total_tasks}: {entity_name}")
                
                if completed % batch_size == 0:
                    try:
                        checkpoint_path = st.session_state.tagger.save_checkpoint(
                            results_by_job, f"checkpoint_{completed}"
                        )
                        status_text.text(f"Checkpoint saved: {checkpoint_path.name}")
                    except Exception as checkpoint_error:
                        # Log checkpoint error but continue processing
                        print(f"Warning: Failed to save checkpoint at {completed}/{total_tasks}: {checkpoint_error}")
                        status_text.text(f"Warning: Checkpoint save failed, continuing processing...")
                
            except Exception as e:
                st.error(f"Error processing entity: {str(e)}")
                completed += 1
                progress = min(completed / total_tasks, 1.0)  # Clamp to 1.0 to avoid exceeding bounds
                progress_bar.progress(progress)
    
    final_results_by_sheet = {}
    for job_idx, job in enumerate(tagging_jobs):
        sheet_key = job['sheet_key']
        config_num = job['config_num']
        
        if sheet_key not in final_results_by_sheet:
            final_results_by_sheet[sheet_key] = job['df'].copy()
        
        job_results = results_by_job[job_idx]
        jobs_for_sheet = [j for j in tagging_jobs if j['sheet_key'] == sheet_key]
        suffix = f"_{config_num}" if len(jobs_for_sheet) > 1 else ""
        
        if job_results:
            job_results.sort(key=lambda x: x['_original_index'])
            
            for result in job_results:
                idx = result['_original_index']
                for key, value in result.items():
                    if key != '_original_index' and key in result:
                        if key not in job['df'].columns:
                            col_name = f"{key}{suffix}"
                            final_results_by_sheet[sheet_key].loc[idx, col_name] = value
    
    # Ensure progress bar shows 100% completion
    progress_bar.progress(1.0)
    status_text.text(f"Processing complete! {completed}/{total_tasks} entities processed")
    
    st.session_state.results = final_results_by_sheet
    st.session_state.processing = False
    
    st.success(f"✅ Processing complete! Processed {completed} entities across {len(tagging_jobs)} configurations")
    
    if len(tagging_jobs) > 1:
        with st.expander("📊 Per-Configuration Statistics"):
            for job_idx, job in enumerate(tagging_jobs):
                sheet_name = job['sheet_name']
                config_num = job['config_num']
                job_results = results_by_job[job_idx]
                
                if job_results:
                    status_counts = {}
                    for result in job_results:
                        status = result.get('Status', 'Unknown')
                        status_counts[status] = status_counts.get(status, 0) + 1
                    
                    st.write(f"**{sheet_name} - Configuration {config_num}:**")
                    for status, count in status_counts.items():
                        st.write(f"- {status}: {count}")
    
    create_download_buttons(final_results_by_sheet)


def create_download_buttons(results_by_sheet):
    """Create download buttons for results"""
    st.header("📥 Download Results")

    # Prepare tagged dataframes for all sheets
    prepared_dfs = {}
    for (filename, sheet_name), df in results_by_sheet.items():
        if 'Status' in df.columns:
            tagged_df = df[df['Status'].notna()].copy()
        else:
            result_cols = [col for col in df.columns if any(
                keyword in col for keyword in ['Tagged_Result', 'Primary_Tag', 'Confidence', 'Status']
            )]
            if result_cols:
                mask = df[result_cols].notna().any(axis=1)
                tagged_df = df[mask].copy()
            else:
                tagged_df = df.copy()

        if len(tagged_df) > 0:
            prepared_dfs[(filename, sheet_name)] = tagged_df
        else:
            st.warning(f"⚠️ No results to export for sheet: {sheet_name}")

    if not prepared_dfs:
        st.error("No results available to download. All sheets are empty.")
        return

    col1, col2, col3 = st.columns([2, 2, 1])

    # Excel download with error handling
    with col1:
        excel_error = None
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for (filename, sheet_name), tagged_df in prepared_dfs.items():
                    safe_sheet_name = sheet_name[:31]
                    # Sanitize the dataframe to remove illegal characters
                    sanitized_df = sanitize_dataframe_for_excel(tagged_df)
                    sanitized_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

            st.download_button(
                label="📥 Download All Results (Excel)",
                data=output.getvalue(),
                file_name=f"tagged_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            excel_error = str(e)
            st.error(f"⚠️ Excel export failed: {excel_error}")
            st.info("Please use the CSV download option instead.")

    # CSV download as alternative/fallback
    with col2:
        try:
            # For multiple sheets, create a zip file with CSVs
            if len(prepared_dfs) == 1:
                # Single sheet - direct CSV download
                (filename, sheet_name), tagged_df = list(prepared_dfs.items())[0]
                csv_output = io.StringIO()
                tagged_df.to_csv(csv_output, index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_output.getvalue(),
                    file_name=f"tagged_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                # Multiple sheets - create zip with multiple CSVs
                import zipfile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for (filename, sheet_name), tagged_df in prepared_dfs.items():
                        csv_output = io.StringIO()
                        tagged_df.to_csv(csv_output, index=False)
                        safe_name = sheet_name[:31].replace('/', '_').replace('\\', '_')
                        zip_file.writestr(f"{safe_name}.csv", csv_output.getvalue())

                st.download_button(
                    label="📥 Download as CSV (zip)",
                    data=zip_buffer.getvalue(),
                    file_name=f"tagged_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        except Exception as e:
            st.error(f"CSV export failed: {str(e)}")

    with col3:
        if st.button("🗑️ Clear Results"):
            st.session_state.results = []
            st.success("Results cleared!")
            st.rerun()
        st.caption("💡 Clear after downloading")
    
    st.header("📊 Results Preview")
    for (filename, sheet_name), df in results_by_sheet.items():
        if 'Status' in df.columns:
            display_df = df[df['Status'].notna()].copy()
        else:
            result_cols = [col for col in df.columns if any(
                keyword in col for keyword in ['Tagged_Result', 'Primary_Tag', 'Confidence', 'Status']
            )]
            if result_cols:
                mask = df[result_cols].notna().any(axis=1)
                display_df = df[mask].copy()
            else:
                display_df = df.copy()
        
        if len(display_df) == 0:
            st.warning(f"⚠️ No results to display for sheet: {sheet_name}")
            continue
            
        with st.expander(f"Results: {sheet_name} ({len(display_df)} tagged rows)", expanded=True):
            show_filter = st.checkbox(f"Show filter options for {sheet_name}", key=f"filter_{sheet_name}")
            if show_filter:
                if 'Status' in display_df.columns:
                    status_filter = st.multiselect(
                        "Filter by status",
                        display_df['Status'].unique(),
                        default=display_df['Status'].unique(),
                        key=f"status_filter_{sheet_name}"
                    )
                    filtered_df = display_df[display_df['Status'].isin(status_filter)]
                else:
                    filtered_df = display_df
            else:
                filtered_df = display_df
            
            st.dataframe(filtered_df, use_container_width=True)
            st.info(f"Showing {len(filtered_df)} tagged rows")


def create_start_tagging_tab():
    """Create the Start Tagging tab with concurrent processing"""
    st.header("🚀 Start Tagging")
    
    if not st.session_state.sheet_data or not st.session_state.tagging_configs:
        st.warning("Please complete configuration in previous tabs first.")
        return
    
    # Clean up configs for sheets that no longer exist
    current_sheet_keys = set(st.session_state.sheet_data.keys())
    config_keys = set(st.session_state.tagging_configs.keys())
    orphaned_keys = config_keys - current_sheet_keys
    
    if orphaned_keys:
        for orphaned_key in orphaned_keys:
            del st.session_state.tagging_configs[orphaned_key]
    
    tagging_jobs = []
    for sheet_key, configs in st.session_state.tagging_configs.items():
        # Only process if sheet still exists in sheet_data
        if sheet_key not in st.session_state.sheet_data:
            continue
            
        if configs:
            filename, sheet_name = sheet_key
            df = st.session_state.sheet_data[sheet_key]
            sheet_config = st.session_state.sheet_configs[sheet_key]
            
            for config in configs:
                tagging_jobs.append({
                    'sheet_key': sheet_key,
                    'filename': filename,
                    'sheet_name': sheet_name,
                    'df': df,
                    'sheet_config': sheet_config,
                    'tagging_config': config,
                    'config_num': config['config_num']
                })
    
    if not tagging_jobs:
        st.warning("No tagging configurations found. Please configure at least one tagging method.")
        return
    
    st.success(f"✅ {len(tagging_jobs)} tagging configuration(s) ready to process")
    
    if len(tagging_jobs) > 1:
        st.subheader("Configuration Summary")
        
        jobs_by_sheet = {}
        for job in tagging_jobs:
            key = job['sheet_key']
            if key not in jobs_by_sheet:
                jobs_by_sheet[key] = []
            jobs_by_sheet[key].append(job)
        
        for sheet_key, sheet_jobs in jobs_by_sheet.items():
            filename, sheet_name = sheet_key
            
            with st.expander(f"📋 {sheet_name} ({len(sheet_jobs)} configuration(s))", expanded=False):
                for job in sheet_jobs:
                    method = job['tagging_config']['method']
                    config_num = job['config_num']
                    st.write(f"**Configuration {config_num}:** {method}")
                    
                    create_row_selection_ui(job, f"{sheet_key}_{config_num}")
    else:
        job = tagging_jobs[0]
        create_row_selection_ui(job, "single")
    
    st.header("⚙️ Processing Options")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_workers = st.number_input("Parallel threads", 1, 100, 5)
    with col2:
        batch_size = st.number_input("Checkpoint batch size", 10, 1000, 100)
    with col3:
        search_retries = st.number_input("Search max retries", 0, 10, 3)
    
    total_rows = sum(
        len(job.get('selected_rows', job['df'])) 
        for job in tagging_jobs
    )
    
    st.info(f"**Will process {total_rows} total rows across {len(tagging_jobs)} configuration(s)**")
    
    if st.button("🏁 Start Tagging", type="primary", disabled=st.session_state.processing):
        run_concurrent_tagging(tagging_jobs, max_workers, batch_size, search_retries)


def create_streamlit_app():
    """Create the updated Streamlit user interface with multi-config support"""
    st.set_page_config(page_title="Generic Tagging Tool", layout="wide")
    
    st.title("🏷️ Tagging Tool")
    st.markdown("Tag any entities using AI with customizable taxonomies or prompts")
    
    initialize_session_state()
    
    with st.expander("📚 Quick Start Guide (4 steps)", expanded=False):
        st.markdown("""
        ### 1. **Initialize** in the sidebar
        Enter API keys. Enter Perplexity key only if using Web Search.
        
        ### 2. **Upload input data and map columns**
        - Upload one or more Excel/CSV files
        - Select one or more sheets per file
        - Configure columns for each sheet
        - **Note:** Removing files or deselecting sheets will automatically clean up their configurations
        
        ### 3. **Configure tagging methods**
        - For each sheet, set up one or more tagging configurations
        - Choose between Taxonomy or Custom Prompt for each configuration
        
        ### 4. **Start tagging**
        - All configurations will run concurrently
        - Multiple configurations for the same sheet will output to one sheet with multiple result columns
        - Download results as Excel with separate sheets
        """)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        with st.expander("🔑 API Keys", expanded=True):
            openai_key = st.text_input("OpenAI API Key", type="password", 
                                      help="Required for AI tagging")
            perplexity_key = st.text_input("Perplexity API Key", type="password",
                                         help="Optional - only needed if using web search")
            
            st.markdown("---")
            checkpoint_path = st.text_input(
                "Checkpoint Directory Path",
                value="tagging_checkpoints",
                help="Directory where checkpoint files will be saved"
            )
            
            if st.button("Initialize Tagger"):
                if openai_key:
                    st.session_state.tagger = GenericTagger(
                        perplexity_api_key=perplexity_key if perplexity_key else None,
                        openai_api_key=openai_key,
                        checkpoint_dir=checkpoint_path
                    )
                    st.success("✅ Tagger initialized!")
                else:
                    st.error("OpenAI API key is required")
        
        if st.session_state.tagger:
            st.header("💾 Checkpoints")
            
            with st.expander("💾 Manage Checkpoints"):
                checkpoint_dir = st.session_state.tagger.checkpoint_dir.absolute()
                st.text(f"Directory: {checkpoint_dir}")
                
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    checkpoint_pkl_files = sorted(
                        checkpoint_dir.glob("*.pkl"),
                        key=lambda x: x.stat().st_mtime,
                        reverse=True
                    )
                except Exception as e:
                    st.error(f"Error reading checkpoint files: {e}")
                    checkpoint_pkl_files = []
                
                if checkpoint_pkl_files:
                    st.write(f"Found {len(checkpoint_pkl_files)} checkpoint(s)")
                    
                    selected_checkpoint = st.selectbox(
                        "Load checkpoint",
                        ["None"] + [f.name for f in checkpoint_pkl_files]
                    )
                    
                    if selected_checkpoint != "None":
                        if st.button("Load Checkpoint"):
                            checkpoint_path = checkpoint_dir / selected_checkpoint
                            loaded_results = st.session_state.tagger.load_checkpoint(checkpoint_path)
                            st.session_state.results = loaded_results
                            st.success(f"Loaded checkpoint")
                    
                    if st.button("Clear All Checkpoints"):
                        for f in checkpoint_pkl_files:
                            f.unlink()
                        for f in checkpoint_dir.glob("*.xlsx"):
                            f.unlink()
                        st.success("All checkpoints cleared")
                        st.rerun()
                else:
                    st.info("No checkpoints found yet. Checkpoints are created automatically during processing.")
    
    if st.session_state.tagger:
        tab1, tab2, tab3 = st.tabs([
            "Data Input and Column Configuration",
            "Tagging Method and Configuration", 
            "Start Tagging"
        ])
        
        with tab1:
            create_data_input_tab()
        
        with tab2:
            create_tagging_method_tab()
        
        with tab3:
            create_start_tagging_tab()
    
    else:
        st.warning("Please initialize the tagger with your API keys in the sidebar")


if __name__ == "__main__":
    create_streamlit_app()