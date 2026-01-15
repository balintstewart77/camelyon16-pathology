"""
S3 utilities for accessing CAMELYON16 dataset.

The dataset is publicly available on AWS S3 without authentication.
This module provides simple functions to list and download files.
"""

import os
from typing import List, Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig


def get_s3_client():
    """Create an S3 client configured for public bucket access."""
    return boto3.client('s3', config=BotoConfig(signature_version=UNSIGNED))


def list_s3_files(s3_folder: str, extension: str = '.tif') -> List[str]:
    """
    List files in an S3 folder with a specific extension.
    
    Args:
        s3_folder: S3 path like 's3://bucket/folder/'
        extension: File extension to filter (e.g., '.tif', '.xml')
        
    Returns:
        List of filenames (not full paths)
        
    Example:
        >>> files = list_s3_files('s3://camelyon-dataset/CAMELYON16/images/', '.tif')
        >>> print(files[:3])
        ['normal_001.tif', 'normal_002.tif', 'normal_003.tif']
    """
    # Parse S3 path
    s3_path = s3_folder.replace('s3://', '')
    bucket_name = s3_path.split('/')[0]
    prefix = '/'.join(s3_path.split('/')[1:])
    
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    
    # List objects
    s3_client = get_s3_client()
    files = []
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.lower().endswith(extension.lower()):
                    filename = key.split('/')[-1]
                    files.append(filename)
        
        return files
        
    except Exception as e:
        print(f"Error listing S3 files: {e}")
        return []


def download_file_from_s3(
    s3_folder: str, 
    filename: str, 
    local_dir: str = '/tmp'
) -> Optional[str]:
    """
    Download a single file from S3 to local storage.
    
    Args:
        s3_folder: S3 folder path (e.g., 's3://bucket/folder/')
        filename: Name of file to download
        local_dir: Local directory to save file
        
    Returns:
        Local file path if successful, None if failed
        
    Example:
        >>> path = download_file_from_s3(
        ...     's3://camelyon-dataset/CAMELYON16/images/',
        ...     'tumor_001.tif'
        ... )
        >>> print(path)
        '/tmp/tumor_001.tif'
    """
    os.makedirs(local_dir, exist_ok=True)
    
    # Parse S3 path
    s3_path = f"{s3_folder.rstrip('/')}/{filename}"
    s3_path_clean = s3_path.replace('s3://', '')
    bucket_name = s3_path_clean.split('/')[0]
    s3_key = '/'.join(s3_path_clean.split('/')[1:])
    
    local_path = os.path.join(local_dir, filename)
    
    # Skip if already exists
    if os.path.exists(local_path):
        return local_path
    
    # Download
    s3_client = get_s3_client()
    
    try:
        print(f"Downloading {filename}...")
        s3_client.download_file(bucket_name, s3_key, local_path)
        return local_path
        
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None


def cleanup_file(file_path: Optional[str]) -> None:
    """
    Delete a local file (for cleanup after processing).
    
    Args:
        file_path: Path to file to delete
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to cleanup {file_path}: {e}")
