import pandas as pd
import requests
import json
import time
import re
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


def sanitize_tag_value(value: str) -> str:
    """
    Sanitize tag values by removing illegal and problematic characters.

    Handles:
    - Control characters (0x00-0x1F except tab/newline which are normalized)
    - Null bytes
    - Zero-width characters (ZWSP, ZWNJ, ZWJ, BOM)
    - Excessive whitespace (collapses to single space)
    - Leading/trailing whitespace and commas
    - Carriage returns and newlines (normalized to space)
    - Tabs (normalized to space)
    """
    if not isinstance(value, str):
        return str(value) if value is not None else ""

    # Remove null bytes
    value = value.replace('\x00', '')

    # Remove zero-width characters
    zero_width_chars = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # BOM / Zero-width no-break space
        '\u200e',  # Left-to-right mark
        '\u200f',  # Right-to-left mark
        '\u2060',  # Word joiner
        '\u2061',  # Function application
        '\u2062',  # Invisible times
        '\u2063',  # Invisible separator
        '\u2064',  # Invisible plus
    ]
    for char in zero_width_chars:
        value = value.replace(char, '')

    # Remove control characters (0x00-0x1F) except we handle \t, \n, \r separately
    value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', value)

    # Normalize newlines, carriage returns, and tabs to spaces
    value = value.replace('\r\n', ' ')
    value = value.replace('\r', ' ')
    value = value.replace('\n', ' ')
    value = value.replace('\t', ' ')

    # Collapse multiple spaces into single space
    value = re.sub(r' +', ' ', value)

    # Strip leading/trailing whitespace and trailing commas
    value = value.strip().rstrip(',').strip()

    return value

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
        """Save checkpoint as both pickle and CSV files, and clean up old ones"""
        with self.file_lock:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Always save pickle (can handle empty data and preserves all data types)
            checkpoint_path_pkl = self.checkpoint_dir / f"{checkpoint_name}_{timestamp}.pkl"
            with open(checkpoint_path_pkl, 'wb') as f:
                pickle.dump(results, f)

            # Save CSV checkpoint (more accessible than Excel, no character issues)
            try:
                if isinstance(results, dict):
                    # Multiple jobs - save each as separate CSV or combine
                    all_rows = []
                    for key, data in results.items():
                        if data and len(data) > 0:
                            for row in data:
                                row_copy = row.copy()
                                row_copy['_source_file'] = key[0] if isinstance(key, tuple) else str(key)
                                row_copy['_source_sheet'] = key[1] if isinstance(key, tuple) and len(key) > 1 else ''
                                all_rows.append(row_copy)

                    if all_rows:
                        df = pd.DataFrame(all_rows)
                        checkpoint_path_csv = self.checkpoint_dir / f"{checkpoint_name}_{timestamp}.csv"
                        df.to_csv(checkpoint_path_csv, index=False)
                else:
                    # Handle list of results
                    if results and len(results) > 0:
                        df = pd.DataFrame(results)
                        if len(df) > 0:
                            checkpoint_path_csv = self.checkpoint_dir / f"{checkpoint_name}_{timestamp}.csv"
                            df.to_csv(checkpoint_path_csv, index=False)
            except Exception as e:
                # Log CSV checkpoint error but don't fail - pickle is the primary backup
                print(f"Warning: CSV checkpoint save failed: {e}")

            # Clean up old files (keep last 2 of each type)
            all_pkl_files = sorted(self.checkpoint_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime)
            all_csv_files = sorted(self.checkpoint_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime)

            for old_file in all_pkl_files[:-2]:
                try:
                    old_file.unlink()
                except Exception:
                    pass
            for old_file in all_csv_files[:-2]:
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
                    'primary_tag': sanitize_tag_value(parsed.primary_tag),
                    'secondary_tags': [sanitize_tag_value(tag) for tag in parsed.secondary_tags],
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
                    'tag': sanitize_tag_value(parsed.selected_tag),
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

    def search_and_tag_with_openai(self, entity_name: str, entity_url: str = None,
                                    available_tags: List[str] = None, tag_descriptions: Dict[str, str] = None,
                                    multi_select: bool = False, existing_data: Dict = None,
                                    custom_prompt: str = None, taxonomy_instructions: str = "",
                                    additional_context: str = "", max_retries: int = 3,
                                    progress_callback=None) -> Dict:
        """
        Combined web search and tagging using OpenAI's built-in web_search tool.
        This performs search and classification in a single API call, eliminating
        the need for a separate Perplexity search step.
        """
        if not self.openai_client:
            return {'status': 'error', 'error': 'OpenAI client not initialized'}

        def _search_and_tag():
            context = ""
            if existing_data:
                context = "\n\nAdditional context from data:\n"
                context += "\n".join([f"{k}: {v}" for k, v in existing_data.items() if v])

            # Build the web search tool configuration
            web_search_tool = {"type": "web_search"}

            # Add domain filtering if URL provided
            if entity_url:
                clean_domain = entity_url.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
                web_search_tool["filters"] = {"allowed_domains": [clean_domain]}

            # Build system prompt based on mode
            if custom_prompt and not available_tags:
                # Custom prompt mode (no taxonomy)
                if multi_select:
                    class OpenAISearchMultiOutput(BaseModel):
                        search_summary: str
                        primary_result: str
                        secondary_results: List[str]
                        confidence: float
                        reasoning: str

                    system_content = f"""You have access to web search. First, search the web for information about the entity, then analyze and classify based on the following prompt:

{custom_prompt}

IMPORTANT: Use web search to find current, accurate information about the entity. Focus on factual information from reliable sources.

Since multiple results are allowed, provide:
- search_summary: A brief summary (2-3 sentences) of key information found about the entity
- primary_result: The main/most important classification or result
- secondary_results: A list of additional relevant classifications or results (can be empty list if only one result applies)

For the reasoning field, provide a brief explanation (3 sentences max) that includes:
1. Why you chose the primary result (and secondary results if any)
2. What specific information from your search influenced your decision
3. How any additional context factored into your decision"""

                    user_content = f"Search for and classify: {entity_name}"
                    if additional_context:
                        user_content += f"\n\n{additional_context}"
                    user_content += context

                    response = self.openai_client.responses.parse(
                        model="gpt-5.2",
                        tools=[web_search_tool],
                        input=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ],
                        text_format=OpenAISearchMultiOutput
                    )

                    parsed = response.output_parsed

                    return {
                        'status': 'success',
                        'search_description': parsed.search_summary,
                        'primary_tag': sanitize_tag_value(parsed.primary_result),
                        'secondary_tags': [sanitize_tag_value(tag) for tag in parsed.secondary_results],
                        'confidence': parsed.confidence,
                        'reasoning': parsed.reasoning
                    }
                else:
                    class OpenAISearchSingleOutput(BaseModel):
                        search_summary: str
                        result: str
                        confidence: float
                        reasoning: str

                    system_content = f"""You have access to web search. First, search the web for information about the entity, then analyze and classify based on the following prompt:

{custom_prompt}

IMPORTANT: Use web search to find current, accurate information about the entity. Focus on factual information from reliable sources.

Provide:
- search_summary: A brief summary (2-3 sentences) of key information found about the entity
- result: Your classification result

For the reasoning field, provide a brief explanation (3 sentences max) that includes:
1. Why you chose this particular classification
2. What specific information from your search influenced your decision
3. How any additional context factored into your decision"""

                    user_content = f"Search for and classify: {entity_name}"
                    if additional_context:
                        user_content += f"\n\n{additional_context}"
                    user_content += context

                    response = self.openai_client.responses.parse(
                        model="gpt-5.2",
                        tools=[web_search_tool],
                        input=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ],
                        text_format=OpenAISearchSingleOutput
                    )

                    parsed = response.output_parsed

                    return {
                        'status': 'success',
                        'search_description': parsed.search_summary,
                        'result': sanitize_tag_value(parsed.result),
                        'confidence': parsed.confidence,
                        'reasoning': parsed.reasoning
                    }

            # Taxonomy mode
            tags_desc = "\n".join([
                f"- {tag}: {tag_descriptions.get(tag, 'No description available')}"
                for tag in (available_tags or [])
            ])

            if multi_select:
                class OpenAISearchTaxonomyMultiOutput(BaseModel):
                    search_summary: str
                    primary_tag: str
                    secondary_tags: List[str]
                    confidence: float
                    reasoning: str

                system_content = f"""You have access to web search. First, search the web for information about the entity, then classify it using the taxonomy below.

Select multiple tags if appropriate, with one primary and optional secondary tags.

IMPORTANT: Use web search to find current, accurate information about the entity. Focus on factual information from reliable sources.

Available tags:
{tags_desc}

Ensure your primary_tag and all secondary_tags are from the available tags list above.

Provide:
- search_summary: A brief summary (2-3 sentences) of key information found about the entity
- primary_tag: The most appropriate tag from the taxonomy
- secondary_tags: Additional relevant tags (can be empty)

For the reasoning field, provide a brief explanation (3 sentences max) that includes:
1. Why you chose the specific primary tag (and secondary tags if any)
2. What key information from your search influenced your decision
3. How any additional context data factored into your classification"""

                if taxonomy_instructions:
                    system_content += f"\n\nADDITIONAL CUSTOM INSTRUCTIONS:\n{taxonomy_instructions}\n\nPlease follow these custom instructions while still selecting from the available taxonomy tags."

                user_content = f"Search for and classify: {entity_name}"
                if additional_context:
                    user_content += f"\n\n{additional_context}"
                user_content += context

                response = self.openai_client.responses.parse(
                    model="gpt-5.2",
                    tools=[web_search_tool],
                    input=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    text_format=OpenAISearchTaxonomyMultiOutput
                )

                parsed = response.output_parsed

                return {
                    'status': 'success',
                    'search_description': parsed.search_summary,
                    'primary_tag': sanitize_tag_value(parsed.primary_tag),
                    'secondary_tags': [sanitize_tag_value(tag) for tag in parsed.secondary_tags],
                    'confidence': parsed.confidence,
                    'reasoning': parsed.reasoning
                }
            else:
                class OpenAISearchTaxonomySingleOutput(BaseModel):
                    search_summary: str
                    selected_tag: str
                    confidence: float
                    reasoning: str

                system_content = f"""You have access to web search. First, search the web for information about the entity, then classify it using the taxonomy below.

Select the single most appropriate tag.

IMPORTANT: Use web search to find current, accurate information about the entity. Focus on factual information from reliable sources.

Available tags:
{tags_desc}

Ensure your selected_tag is from the available tags list above.

Provide:
- search_summary: A brief summary (2-3 sentences) of key information found about the entity
- selected_tag: The most appropriate tag from the taxonomy

For the reasoning field, provide a brief explanation (3 sentences max) that includes:
1. Why you chose this specific tag over others
2. What key information from your search influenced your decision
3. How any additional context data factored into your classification"""

                if taxonomy_instructions:
                    system_content += f"\n\nADDITIONAL CUSTOM INSTRUCTIONS:\n{taxonomy_instructions}\n\nPlease follow these custom instructions while still selecting from the available taxonomy tags."

                user_content = f"Search for and classify: {entity_name}"
                if additional_context:
                    user_content += f"\n\n{additional_context}"
                user_content += context

                response = self.openai_client.responses.parse(
                    model="gpt-5.2",
                    tools=[web_search_tool],
                    input=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    text_format=OpenAISearchTaxonomySingleOutput
                )

                parsed = response.output_parsed

                return {
                    'status': 'success',
                    'search_description': parsed.search_summary,
                    'tag': sanitize_tag_value(parsed.selected_tag),
                    'confidence': parsed.confidence,
                    'reasoning': parsed.reasoning
                }

        try:
            return self.retry_with_exponential_backoff(
                _search_and_tag,
                max_retries=max_retries,
                entity_name=entity_name,
                progress_callback=progress_callback
            )
        except Exception as e:
            error_msg = str(e)
            if "refusal" in error_msg.lower():
                return {'status': 'error', 'error': 'Model refused to respond for safety reasons'}
            return {'status': 'error', 'error': error_msg}

    def process_single_entity(self, row_data: Dict, config: Dict, progress_callback=None) -> Dict:
        """Process a single entity based on configuration

        Supports two search providers:
        - 'perplexity': Uses Perplexity API for search, then OpenAI for tagging (2 API calls)
        - 'openai': Uses OpenAI's built-in web_search tool for combined search+tag (1 API call)
        """
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
            search_provider = config.get('search_provider', 'perplexity')

            # Get taxonomy info (needed for both search paths)
            use_taxonomy = config.get('use_taxonomy', True)
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

            # OpenAI combined search+tag path (single API call)
            if config['use_search'] and search_provider == 'openai':
                url_column = config.get('url_column')
                entity_url = row_data.get(url_column) if url_column else None
                max_retries = config.get('search_max_retries', 3)

                tag_result = self.search_and_tag_with_openai(
                    entity_name=entity_name,
                    entity_url=entity_url,
                    available_tags=available_tags if use_taxonomy else None,
                    tag_descriptions=tag_descriptions if use_taxonomy else None,
                    multi_select=config.get('multi_select', False),
                    existing_data=context_data,
                    custom_prompt=custom_prompt,
                    taxonomy_instructions=taxonomy_instructions,
                    additional_context=additional_context,
                    max_retries=max_retries,
                    progress_callback=progress_callback
                )

                result = row_data.copy()

                if tag_result['status'] == 'error':
                    result.update({
                        'Search_Description': tag_result.get('error', 'Unknown error'),
                        'Tagged_Result': 'Error',
                        'Confidence': '0%',
                        'Reasoning': tag_result['error'],
                        'Status': 'Error'
                    })
                else:
                    # Add search description from OpenAI's web search
                    if 'search_description' in tag_result:
                        result['Search_Description'] = tag_result['search_description'][:500]

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
                        tag_value = sanitize_tag_value(tag_value)

                        result.update({
                            'Tagged_Result': tag_value,
                            'Confidence': f"{tag_result['confidence']:.0%}",
                            'Reasoning': tag_result['reasoning'],
                            'Status': 'Success'
                        })

                return result

            # Perplexity search + OpenAI tag path (original 2-step process)
            search_success = True
            if config['use_search'] and search_provider == 'perplexity':
                # Check if Perplexity API key is available
                if not self.perplexity_api_key:
                    result = row_data.copy()
                    result.update({
                        'Search_Description': 'Perplexity API key not provided',
                        'Tagged_Result': 'Configuration Error',
                        'Confidence': '0%',
                        'Reasoning': 'Perplexity search provider selected but no Perplexity API key was provided. Please add your Perplexity API key in the sidebar or switch to OpenAI search provider.',
                        'Status': 'Config Error'
                    })
                    return result
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
                    tag_value = sanitize_tag_value(tag_value)

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
    if 'last_tagging_jobs' not in st.session_state:
        st.session_state.last_tagging_jobs = None


def create_column_config(df, sheet_key):
    """Create column configuration UI for a single sheet with smart defaults"""
    columns = df.columns.tolist()

    # Auto-detect common column patterns
    auto_name = auto_detect_column(columns, ['name', 'company', 'entity', 'title', 'organization'])
    auto_url = auto_detect_column(columns, ['url', 'website', 'link', 'domain', 'site'])
    auto_desc = auto_detect_column(columns, ['description', 'desc', 'about', 'summary', 'overview'])
    auto_category = auto_detect_column(columns, ['category', 'type', 'segment', 'group', 'class'])

    # Entity name column (required) - default to auto-detected or first column
    default_name_idx = columns.index(auto_name) if auto_name else 0
    name_column = st.selectbox(
        "Entity name column",
        columns,
        index=default_name_idx,
        key=f"name_col_{sheet_key}"
    )

    # Web search toggle with OpenAI as default provider
    use_search = st.checkbox(
        "Use web search to gather information",
        value=False,
        key=f"use_search_{sheet_key}",
        help="Search the web for each entity before tagging"
    )

    search_provider = 'openai'  # Default to OpenAI (simpler, one API call)
    url_column = None
    description_columns = []

    if use_search:
        # Show provider choice in expander for advanced users
        with st.expander("Search options", expanded=False):
            search_provider = st.radio(
                "Search provider",
                options=['openai', 'perplexity'],
                format_func=lambda x: 'OpenAI (recommended)' if x == 'openai' else 'Perplexity',
                key=f"search_provider_{sheet_key}",
                horizontal=True
            )
            if search_provider == 'perplexity':
                st.caption("Requires Perplexity API key")

            # URL column for domain filtering
            default_url_idx = columns.index(auto_url) + 1 if auto_url else 0
            url_column = st.selectbox(
                "URL column (for domain filtering)",
                ['None'] + columns,
                index=default_url_idx,
                key=f"url_col_{sheet_key}"
            )
            url_column = None if url_column == 'None' else url_column
    else:
        # Description columns when not using search
        default_desc = [auto_desc] if auto_desc else []
        description_columns = st.multiselect(
            "Description columns",
            columns,
            default=default_desc,
            key=f"desc_cols_{sheet_key}",
            help="Columns with information about each entity"
        )

    # Optional columns in expander
    with st.expander("Additional options", expanded=False):
        context_columns = st.multiselect(
            "Context columns",
            [c for c in columns if c != name_column],
            key=f"context_cols_{sheet_key}",
            help="Extra columns to provide context"
        )

        default_cat_idx = columns.index(auto_category) + 1 if auto_category else 0
        category_column = st.selectbox(
            "Category column",
            ['None'] + columns,
            index=default_cat_idx,
            key=f"category_col_{sheet_key}",
            help="For category-based taxonomy filtering"
        )
        category_column = None if category_column == 'None' else category_column

    return {
        'name_column': name_column,
        'use_search': use_search,
        'search_provider': search_provider,
        'url_column': url_column,
        'description_columns': description_columns,
        'context_columns': context_columns,
        'category_column': category_column,
        'multi_select': False
    }


def auto_detect_column(columns, patterns):
    """Auto-detect column based on common naming patterns"""
    columns_lower = [c.lower() for c in columns]
    for pattern in patterns:
        for i, col_lower in enumerate(columns_lower):
            if pattern in col_lower:
                return columns[i]
    return None


def create_data_input_tab():
    """Create the Data Input tab with multi-file/sheet support"""
    st.header("Data Input")

    st.caption("Upload your data file with entities to tag. The tool adds result columns to your data.")

    uploaded_files = st.file_uploader(
        "Upload file(s)",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed"
    )
    
    if not uploaded_files:
        if st.session_state.sheet_data or st.session_state.sheet_configs or st.session_state.tagging_configs:
            st.session_state.sheet_data = {}
            st.session_state.sheet_configs = {}
            st.session_state.tagging_configs = {}
        return

    st.session_state.uploaded_files = {f.name: f for f in uploaded_files}

    file_sheet_options = {}
    for filename, file in st.session_state.uploaded_files.items():
        if filename.endswith('.csv'):
            file_sheet_options[filename] = ['main']
        else:
            excel_file = pd.ExcelFile(file)
            file_sheet_options[filename] = excel_file.sheet_names

    # Sheet selection - only show if multiple sheets exist
    selected_sheets = {}
    has_multi_sheet = any(len(sheets) > 1 for sheets in file_sheet_options.values())

    if has_multi_sheet:
        st.subheader("Select Sheets")

    for filename, sheets in file_sheet_options.items():
        if len(sheets) == 1:
            selected_sheets[filename] = sheets
        else:
            selected = st.multiselect(
                f"{filename}",
                sheets,
                default=sheets[:1],
                key=f"sheet_select_{filename}"
            )
            if selected:
                selected_sheets[filename] = selected

    total_sheets = sum(len(sheets) for sheets in selected_sheets.values())

    if total_sheets == 0:
        st.warning("Select at least one sheet to proceed.")
        return

    # Build sheet data
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

    # Clean up removed sheets
    current_keys = set(st.session_state.sheet_data.keys())
    all_keys = set()
    all_keys.update(st.session_state.sheet_configs.keys())
    all_keys.update(st.session_state.tagging_configs.keys())

    for removed_key in (all_keys - current_keys):
        st.session_state.sheet_configs.pop(removed_key, None)
        st.session_state.tagging_configs.pop(removed_key, None)

    # Column Configuration
    st.subheader("Column Mapping")

    if total_sheets == 1:
        key = list(st.session_state.sheet_data.keys())[0]
        df = st.session_state.sheet_data[key]

        config = create_column_config(df, key)
        st.session_state.sheet_configs[key] = config

        # Data preview
        with st.expander(f"Preview ({len(df)} rows)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
    else:
        for key in st.session_state.sheet_data.keys():
            filename, sheet_name = key
            df = st.session_state.sheet_data[key]

            with st.expander(f"{sheet_name} ({len(df)} rows)", expanded=False):
                config = create_column_config(df, key)
                st.session_state.sheet_configs[key] = config

                st.dataframe(df.head(5), use_container_width=True)


def create_taxonomy_config(sheet_key, config_idx, config):
    """Create simplified taxonomy configuration UI"""

    # Taxonomy file upload (primary method)
    taxonomy_file = st.file_uploader(
        "Upload taxonomy file (Excel)",
        type=['xlsx', 'xls'],
        key=f"tax_file_{sheet_key}_{config_idx}",
        help="Excel with columns: Tag, Description (optionally: Category)"
    )

    if taxonomy_file:
        tax_sheet = None
        tax_excel = pd.ExcelFile(taxonomy_file)
        if len(tax_excel.sheet_names) > 1:
            tax_sheet = st.selectbox("Sheet", tax_excel.sheet_names,
                                    key=f"tax_sheet_{sheet_key}_{config_idx}")

        taxonomy = st.session_state.tagger.load_taxonomy_from_excel(taxonomy_file, tax_sheet)
        config['taxonomy'] = taxonomy

        # Show loaded tags compactly
        total_tags = sum(len(tags) for tags in taxonomy.categories.values())
        st.success(f"Loaded {total_tags} tags")

        with st.expander("View taxonomy", expanded=False):
            for category, tags in taxonomy.categories.items():
                if category != 'default':
                    st.caption(category)
                for tag in tags:
                    desc = taxonomy.descriptions.get(tag, "")
                    st.write(f"• {tag}" + (f" - {desc}" if desc else ""))

    # Show format help in collapsed expander
    with st.expander("Format guide", expanded=False):
        st.caption("Excel format: Tag and Description columns. Add Category column for grouped tags.")
        example_data = {
            "Tag": ["Software", "Hardware", "Consulting"],
            "Description": ["Software development", "Hardware manufacturing", "Professional services"]
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True)

    # Alternative input methods in collapsed expander
    with st.expander("Other input methods", expanded=False):
        alt_method = st.radio(
            "Method",
            ["YAML text", "Manual entry"],
            key=f"alt_method_{sheet_key}_{config_idx}",
            horizontal=True,
            label_visibility="collapsed"
        )

        if alt_method == "YAML text":
            taxonomy_text = st.text_area(
                "YAML",
                height=150,
                key=f"tax_text_{sheet_key}_{config_idx}",
                placeholder="categories:\n  default:\n    - Tag1\n    - Tag2\ndescriptions:\n  Tag1: \"Description\"",
                label_visibility="collapsed"
            )

            if st.button("Parse", key=f"parse_tax_{sheet_key}_{config_idx}"):
                try:
                    taxonomy_dict = yaml.safe_load(taxonomy_text)
                    taxonomy = st.session_state.tagger.load_taxonomy_from_dict(taxonomy_dict)
                    config['taxonomy'] = taxonomy
                    st.success("Taxonomy loaded")
                except Exception as e:
                    st.error(f"Parse error: {str(e)}")

        else:  # Manual entry
            tags_text = st.text_area(
                "Tags (one per line)",
                key=f"tags_simple_{sheet_key}_{config_idx}",
                height=100,
                label_visibility="collapsed",
                placeholder="Tag1\nTag2\nTag3"
            )

            if tags_text and st.button("Create", key=f"create_tax_{sheet_key}_{config_idx}"):
                tags = [t.strip() for t in tags_text.split('\n') if t.strip()]
                taxonomy = TaxonomyConfig(categories={"default": tags}, descriptions={})
                config['taxonomy'] = taxonomy
                st.success(f"Created {len(tags)} tags")

    # Custom instructions (collapsed by default)
    with st.expander("Custom instructions", expanded=False):
        custom_instructions = st.text_area(
            "Additional guidance for tagging",
            value=config.get('custom_instructions', ''),
            height=80,
            key=f"custom_inst_{sheet_key}_{config_idx}",
            placeholder="E.g., Prioritize specific tags over general ones",
            label_visibility="collapsed"
        )

        if custom_instructions:
            validation_error = validate_custom_instructions(custom_instructions)
            if validation_error:
                st.error(validation_error)
                config['custom_instructions'] = ""
            else:
                config['custom_instructions'] = custom_instructions
        else:
            config['custom_instructions'] = ""


def create_custom_prompt_config(sheet_key, config_idx, config):
    """Create simplified custom prompt configuration UI"""

    st.caption("Use a preset or write your own analysis prompt")

    # Simplified preset selection
    preset_options = ["Custom"] + [k for k in PRESET_PROMPTS.keys() if k != "Custom Analysis"]
    preset_choice = st.selectbox(
        "Start from preset",
        preset_options,
        key=f"preset_{sheet_key}_{config_idx}",
        label_visibility="collapsed"
    )

    # Get initial value
    if preset_choice == "Custom":
        initial_value = config.get('custom_prompt', '')
    else:
        initial_value = PRESET_PROMPTS.get(preset_choice, '')

    custom_prompt = st.text_area(
        "Prompt",
        value=initial_value,
        height=150,
        key=f"prompt_{sheet_key}_{config_idx}",
        placeholder="Analyze and classify this entity...",
        label_visibility="collapsed"
    )

    config['custom_prompt'] = custom_prompt

    with st.expander("Tips", expanded=False):
        st.caption("""
• Be specific about classification categories
• Define the expected output format
• Include criteria for decision-making
        """)


def create_tagging_method_tab():
    """Create the Tagging Method tab with multi-config support"""
    st.header("Tagging Method")

    if not st.session_state.sheet_data:
        st.warning("Upload data in the Data Input tab first.")
        return

    # Clean up orphaned configs
    current_sheet_keys = set(st.session_state.sheet_data.keys())
    for orphaned_key in (set(st.session_state.tagging_configs.keys()) - current_sheet_keys):
        del st.session_state.tagging_configs[orphaned_key]

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

        title = f"{sheet_name} - Config {config_num}" if config_num > 1 else sheet_name

        with st.expander(title, expanded=(len(all_configs) == 1)):

            method = st.radio(
                "Tagging method",
                ["Use Taxonomy", "Use Custom Prompt"],
                key=f"method_{sheet_key}_{config_idx}",
                index=0 if config['method'] == 'Use Taxonomy' else 1,
                horizontal=True,
                help="Taxonomy: classify into predefined tags. Custom Prompt: free-form analysis."
            )
            config['method'] = method

            if method == "Use Taxonomy":
                create_taxonomy_config(sheet_key, config_idx, config)
            else:
                create_custom_prompt_config(sheet_key, config_idx, config)

            # Add/remove config buttons - simplified
            st.markdown("---")
            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button(f"Add configuration", key=f"add_config_{sheet_key}_{config_idx}",
                           use_container_width=True):
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
                    if st.button("Remove", key=f"remove_{sheet_key}_{config_idx}",
                               use_container_width=True):
                        st.session_state.tagging_configs[sheet_key].pop(config_idx)
                        st.rerun()


def create_row_selection_ui(job, key_suffix):
    """Create simplified row selection UI for a tagging job"""
    df = job['df']
    sheet_config = job['sheet_config']

    col1, col2 = st.columns([1, 1])

    with col1:
        # Simplified row selection
        row_option = st.radio(
            "Rows to process",
            ["All rows", "Test (first 10)", "Custom selection"],
            key=f"row_option_{key_suffix}",
            horizontal=True,
            label_visibility="collapsed"
        )

    with col2:
        multi_select = st.checkbox(
            "Allow multiple tags",
            value=job.get('multi_select', False),
            key=f"multi_select_{key_suffix}",
            help="Assign primary and secondary tags"
        )
        job['multi_select'] = multi_select

    if row_option == "Test (first 10)":
        job['selected_rows'] = df.head(10)
        st.caption(f"Will process first 10 of {len(df)} rows")
        return

    if row_option == "All rows":
        job['selected_rows'] = df
        return

    # Custom selection options (collapsed by default)
    selected_indices = set()

    selection_method = st.radio(
        "Selection method",
        ["Row range", "Specific rows"] + (["By category"] if sheet_config.get('category_column') else []),
        key=f"sel_method_{key_suffix}",
        horizontal=True,
        label_visibility="collapsed"
    )

    if selection_method == "Row range":
        col1, col2 = st.columns(2)
        with col1:
            start_row = st.number_input("From", 1, len(df), 1, key=f"start_{key_suffix}")
        with col2:
            end_row = st.number_input("To", start_row, len(df), min(10, len(df)), key=f"end_{key_suffix}")
        if start_row <= end_row:
            selected_indices.update(range(start_row - 1, end_row))

    elif selection_method == "Specific rows":
        manual_rows = st.text_input("Rows (e.g., 1,3,5-8)", key=f"manual_input_{key_suffix}")
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
                st.error("Invalid format")

    elif selection_method == "By category" and sheet_config.get('category_column'):
        categories = df[sheet_config['category_column']].unique()
        selected_cats = st.multiselect("Categories", categories, key=f"cats_{key_suffix}")
        if selected_cats:
            mask = df[sheet_config['category_column']].isin(selected_cats)
            selected_indices.update(df[mask].index)

    if selected_indices:
        job['selected_rows'] = df.loc[sorted(selected_indices)]
    else:
        job['selected_rows'] = df.iloc[0:0]

    st.caption(f"Selected {len(job.get('selected_rows', df))} rows")


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
    
    st.success(f"Complete! {completed} entities processed")

    # Store tagging jobs in session state for retry functionality
    st.session_state.last_tagging_jobs = tagging_jobs

    create_download_buttons(final_results_by_sheet, tagging_jobs)


def get_failed_rows_info(results_by_sheet):
    """Identify rows with errors across all result sheets and status columns.

    Returns dict mapping sheet_key -> list of (row_index, status_column) tuples
    """
    failed_info = {}
    error_statuses = ['Error', 'Search Error', 'Config Error', 'Search Failed']

    for sheet_key, df in results_by_sheet.items():
        # Find all status columns (Status, Status_1, Status_2, etc.)
        status_cols = [col for col in df.columns if col == 'Status' or col.startswith('Status_')]

        failed_rows = []
        for status_col in status_cols:
            if status_col in df.columns:
                # Find rows where this status column indicates an error
                error_mask = df[status_col].isin(error_statuses)
                for idx in df[error_mask].index:
                    failed_rows.append((idx, status_col))

        if failed_rows:
            failed_info[sheet_key] = failed_rows

    return failed_info


def retry_failed_rows(tagging_jobs, failed_info, max_workers, batch_size, search_retries):
    """Reprocess only the failed rows, updating existing results."""
    st.session_state.processing = True

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # Build tasks only for failed rows
    all_tasks = []

    for job_idx, job in enumerate(tagging_jobs):
        sheet_key = job['sheet_key']
        if sheet_key not in failed_info:
            continue

        df = job['df']
        sheet_config = job['sheet_config']
        tagging_config = job['tagging_config']
        config_num = job['config_num']

        # Determine which status column this job writes to
        jobs_for_sheet = [j for j in tagging_jobs if j['sheet_key'] == sheet_key]
        suffix = f"_{config_num}" if len(jobs_for_sheet) > 1 else ""
        status_col = f"Status{suffix}"

        # Get failed row indices for this specific config
        failed_indices = set()
        for idx, failed_status_col in failed_info[sheet_key]:
            if failed_status_col == status_col or (failed_status_col == 'Status' and suffix == ''):
                failed_indices.add(idx)

        if not failed_indices:
            continue

        tagger_config = {
            **sheet_config,
            'use_taxonomy': tagging_config['method'] == 'Use Taxonomy',
            'custom_prompt': tagging_config.get('custom_prompt'),
            'taxonomy_custom_instructions': tagging_config.get('custom_instructions', ''),
            'search_max_retries': search_retries,
            'multi_select': job.get('multi_select', False),
            'taxonomy': tagging_config.get('taxonomy')
        }

        for idx in failed_indices:
            if idx in df.index:
                row = df.loc[idx]
                all_tasks.append({
                    'job_idx': job_idx,
                    'job': job,
                    'idx': idx,
                    'row': row,
                    'config': tagger_config,
                    'suffix': suffix
                })

    if not all_tasks:
        st.warning("No failed rows found to retry.")
        st.session_state.processing = False
        return

    completed = 0
    total_tasks = len(all_tasks)

    status_text.text(f"Retrying {total_tasks} failed rows...")

    # Get current results to update
    results_by_sheet = st.session_state.results

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
            sheet_key = task['job']['sheet_key']
            suffix = task['suffix']
            idx = task['idx']

            try:
                result = future.result()

                # Update only the columns for THIS config (using suffix)
                # Keys like 'Status', 'Tagged_Result' become 'Status_2', 'Tagged_Result_2' for config 2
                result_keys = ['Status', 'Tagged_Result', 'Primary_Tag', 'Secondary_Tags',
                              'Confidence', 'Reasoning', 'Search_Description']

                for key in result_keys:
                    if key in result:
                        col_name = f"{key}{suffix}" if suffix else key
                        # Update if column exists, or create it
                        results_by_sheet[sheet_key].loc[idx, col_name] = result[key]

                completed += 1
                progress = min(completed / total_tasks, 1.0)
                progress_bar.progress(progress)

                entity_name = result.get(task['config']['name_column'], 'Unknown')
                new_status = result.get('Status', 'Unknown')
                status_text.text(f"Retry {completed}/{total_tasks}: {entity_name} -> {new_status}")

            except Exception as e:
                st.error(f"Retry error: {str(e)}")
                completed += 1
                progress = min(completed / total_tasks, 1.0)
                progress_bar.progress(progress)

    progress_bar.progress(1.0)

    # Count remaining errors
    new_failed_info = get_failed_rows_info(results_by_sheet)
    remaining_errors = sum(len(rows) for rows in new_failed_info.values())

    st.session_state.results = results_by_sheet
    st.session_state.processing = False

    if remaining_errors > 0:
        st.warning(f"Retry complete. {remaining_errors} errors remain.")
    else:
        st.success("All retries successful!")

    st.rerun()


def create_download_buttons(results_by_sheet, tagging_jobs=None):
    """Create download buttons for results"""
    st.header("Results")

    # Check for failed rows
    failed_info = get_failed_rows_info(results_by_sheet)
    total_errors = sum(len(rows) for rows in failed_info.values())

    # Show error summary and retry option
    if total_errors > 0 and tagging_jobs:
        st.warning(f"{total_errors} rows have errors")

        col_retry, col_info = st.columns([1, 3])
        with col_retry:
            if st.button("Retry Failed", type="primary", use_container_width=True,
                        disabled=st.session_state.processing):
                retry_failed_rows(tagging_jobs, failed_info, max_workers=5,
                                 batch_size=100, search_retries=3)
        with col_info:
            # Show breakdown by sheet
            error_details = []
            for sheet_key, rows in failed_info.items():
                sheet_name = sheet_key[1] if isinstance(sheet_key, tuple) else str(sheet_key)
                error_details.append(f"{sheet_name}: {len(rows)}")
            st.caption(f"Errors by sheet: {', '.join(error_details)}")

    # Prepare tagged dataframes
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

    if not prepared_dfs:
        st.error("No results to download.")
        return

    # Download buttons - compact layout
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for (filename, sheet_name), tagged_df in prepared_dfs.items():
                    safe_sheet_name = sheet_name[:31]
                    sanitized_df = sanitize_dataframe_for_excel(tagged_df)
                    sanitized_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

            st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name=f"tagged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Excel failed: {str(e)}")

    with col2:
        try:
            if len(prepared_dfs) == 1:
                (filename, sheet_name), tagged_df = list(prepared_dfs.items())[0]
                csv_output = io.StringIO()
                tagged_df.to_csv(csv_output, index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_output.getvalue(),
                    file_name=f"tagged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                import zipfile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for (filename, sheet_name), tagged_df in prepared_dfs.items():
                        csv_output = io.StringIO()
                        tagged_df.to_csv(csv_output, index=False)
                        safe_name = sheet_name[:31].replace('/', '_').replace('\\', '_')
                        zip_file.writestr(f"{safe_name}.csv", csv_output.getvalue())

                st.download_button(
                    label="Download CSV (zip)",
                    data=zip_buffer.getvalue(),
                    file_name=f"tagged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"CSV failed: {str(e)}")

    with col3:
        if st.button("Clear", use_container_width=True):
            st.session_state.results = []
            st.rerun()

    # Results preview - simplified
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
            continue

        with st.expander(f"{sheet_name} ({len(display_df)} rows)", expanded=True):
            # Filter option
            if 'Status' in display_df.columns:
                statuses = display_df['Status'].unique().tolist()
                if len(statuses) > 1:
                    filter_status = st.multiselect(
                        "Filter",
                        statuses,
                        default=statuses,
                        key=f"filter_{sheet_name}",
                        label_visibility="collapsed"
                    )
                    display_df = display_df[display_df['Status'].isin(filter_status)]

            st.dataframe(display_df, use_container_width=True, hide_index=True)


def create_start_tagging_tab():
    """Create simplified Start Tagging tab"""
    st.header("Run Tagging")

    # Show existing results with retry option if available
    if st.session_state.results and isinstance(st.session_state.results, dict):
        failed_info = get_failed_rows_info(st.session_state.results)
        total_errors = sum(len(rows) for rows in failed_info.values())

        if total_errors > 0:
            st.info(f"Previous results have {total_errors} errors")

            if hasattr(st.session_state, 'last_tagging_jobs') and st.session_state.last_tagging_jobs:
                if st.button("Retry Failed Rows", type="secondary"):
                    retry_failed_rows(
                        st.session_state.last_tagging_jobs,
                        failed_info,
                        max_workers=5,
                        batch_size=100,
                        search_retries=3
                    )

        # Show results
        create_download_buttons(st.session_state.results,
                               getattr(st.session_state, 'last_tagging_jobs', None))
        st.markdown("---")

    if not st.session_state.sheet_data or not st.session_state.tagging_configs:
        st.warning("Complete setup in previous tabs first.")
        return

    # Clean up orphaned configs
    current_sheet_keys = set(st.session_state.sheet_data.keys())
    for orphaned_key in (set(st.session_state.tagging_configs.keys()) - current_sheet_keys):
        del st.session_state.tagging_configs[orphaned_key]

    tagging_jobs = []
    for sheet_key, configs in st.session_state.tagging_configs.items():
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
        st.warning("Configure at least one tagging method.")
        return

    # Row selection
    if len(tagging_jobs) > 1:
        for job in tagging_jobs:
            sheet_name = job['sheet_name']
            config_num = job['config_num']
            with st.expander(f"{sheet_name}" + (f" - Config {config_num}" if config_num > 1 else ""), expanded=False):
                create_row_selection_ui(job, f"{job['sheet_key']}_{config_num}")
    else:
        job = tagging_jobs[0]
        create_row_selection_ui(job, "single")

    # Processing options in expander (with sensible defaults)
    max_workers, batch_size, search_retries = 5, 100, 3

    with st.expander("Advanced options", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_workers = st.number_input("Parallel threads", 1, 100, 5)
        with col2:
            batch_size = st.number_input("Checkpoint interval", 10, 1000, 100)
        with col3:
            search_retries = st.number_input("Max retries", 0, 10, 3)

    total_rows = sum(len(job.get('selected_rows', job['df'])) for job in tagging_jobs)

    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric("Total rows to process", total_rows)
    with col2:
        if st.button("Start", type="primary", disabled=st.session_state.processing,
                    use_container_width=True):
            run_concurrent_tagging(tagging_jobs, max_workers, batch_size, search_retries)


def create_streamlit_app():
    """Create the Streamlit user interface"""
    st.set_page_config(page_title="Tagging Tool", layout="wide")

    st.title("Tagging Tool")

    initialize_session_state()

    with st.expander("Help", expanded=False):
        st.markdown("""
**1. Setup** - Enter OpenAI API key in sidebar

**2. Data Input** - Upload file, map columns

**3. Tagging Method** - Upload taxonomy or write prompt

**4. Run** - Select rows and start tagging
        """)
    
    with st.sidebar:
        st.header("Setup")

        # Simplified API key inputs
        openai_key = st.text_input("OpenAI API Key", type="password",
                                  help="Required for AI tagging")
        perplexity_key = st.text_input("Perplexity API Key (optional)", type="password",
                                     help="Only needed if using Perplexity search")

        # Auto-initialize when OpenAI key is provided
        if openai_key and not st.session_state.tagger:
            st.session_state.tagger = GenericTagger(
                perplexity_api_key=perplexity_key if perplexity_key else None,
                openai_api_key=openai_key,
                checkpoint_dir="tagging_checkpoints"
            )

        # Update tagger if keys change
        if openai_key and st.session_state.tagger:
            if perplexity_key != st.session_state.tagger.perplexity_api_key:
                st.session_state.tagger.perplexity_api_key = perplexity_key

        if st.session_state.tagger:
            st.success("Ready")

            # Checkpoints in collapsed expander
            with st.expander("Checkpoints", expanded=False):
                checkpoint_dir = st.session_state.tagger.checkpoint_dir.absolute()
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                try:
                    checkpoint_pkl_files = sorted(
                        checkpoint_dir.glob("*.pkl"),
                        key=lambda x: x.stat().st_mtime,
                        reverse=True
                    )
                except Exception as e:
                    checkpoint_pkl_files = []

                if checkpoint_pkl_files:
                    selected_checkpoint = st.selectbox(
                        "Load checkpoint",
                        ["None"] + [f.name for f in checkpoint_pkl_files],
                        label_visibility="collapsed"
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        if selected_checkpoint != "None":
                            if st.button("Load", use_container_width=True):
                                checkpoint_path_load = checkpoint_dir / selected_checkpoint
                                loaded_results = st.session_state.tagger.load_checkpoint(checkpoint_path_load)
                                st.session_state.results = loaded_results
                                st.success("Loaded")
                    with col2:
                        if st.button("Clear All", use_container_width=True):
                            for f in checkpoint_pkl_files:
                                f.unlink()
                            for f in checkpoint_dir.glob("*.csv"):
                                f.unlink()
                            st.rerun()
                else:
                    st.caption("No checkpoints yet")
        else:
            st.info("Enter OpenAI API key to start")
    
    if st.session_state.tagger:
        tab1, tab2, tab3 = st.tabs(["Data Input", "Tagging Method", "Run"])
        
        with tab1:
            create_data_input_tab()
        
        with tab2:
            create_tagging_method_tab()
        
        with tab3:
            create_start_tagging_tab()
    
    else:
        st.info("Enter your OpenAI API key in the sidebar to begin.")


if __name__ == "__main__":
    create_streamlit_app()