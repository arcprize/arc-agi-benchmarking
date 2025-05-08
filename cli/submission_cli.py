#!/usr/bin/env python3

import click
import json
import os
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def validate_json_structure(content):
    """Validate the JSON can be parsed."""
    return True  # If we got here, json.load() succeeded

@click.group()
def cli():
    """CLI tool for validating and uploading model submissions."""
    pass

@cli.command()
@click.argument('task_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
def validate(task_dir, output_dir):
    """Validate output files against task directory."""
    click.echo(f"Starting validation...")
    
    task_path = Path(task_dir)
    output_path = Path(output_dir)
    
    click.echo(f"Task directory: {task_path.absolute()}")
    click.echo(f"Output directory: {output_path.absolute()}")
    
    # Count JSON files in task directory
    task_files = list(task_path.glob('*.json'))
    expected_count = len(task_files)
    
    # Count and validate output files
    output_files = list(output_path.glob('*.json'))
    
    click.echo(f"\nValidating {output_path.name}...")
    click.echo(f"Found {len(output_files)} files (expected {expected_count})")
    
    
    if len(output_files) != expected_count:
        click.echo(f"❌ Error: Output directory must contain exactly {expected_count} files (found in {task_dir})")
        is_valid = False
    else:
        click.echo(f"✅ Output directory contains {expected_count} files")
        is_valid = True
    
    invalid_files = []
    for file in output_files:
        try:
            with open(file) as f:
                json.load(f)
        except json.JSONDecodeError:
            invalid_files.append(file.name)
            
    if invalid_files:
        click.echo("\n❌ The following files have invalid JSON:")
        for file in invalid_files:
            click.echo(f"  - {file}")
        is_valid = False
    else:
        click.echo("\n✅ All files contain valid JSON")

    return is_valid

@cli.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--task-set', type=str, required=True, help="Name of task set (e.g. public_eval_v1)")
@click.option('--org', type=str, default="arcprize", help="Hugging Face organization name")
@click.option('--public/--private', default=False, help="Make dataset public (default: private)")
def upload(output_dir, task_set, org, public):
    """Upload model outputs to Hugging Face dataset."""
    click.echo(f"Starting upload...")
    
    output_path = Path(output_dir)
    model_name = output_path.name
    
    # Format repo name based on task set and org
    repo_id = f"{org}/{task_set}"
    
    api = HfApi()
    try:
        # Create or get dataset repository for this task set
        api.create_repo(
            repo_id=repo_id, 
            repo_type="dataset",
            private=not public,  # Set privacy based on flag
            exist_ok=True
        )
        
        click.echo(f"Created/updated repository as {'public' if public else 'private'}")
        
        # Upload files to a model-specific folder within the task set repo
        click.echo(f"Uploading files to {repo_id}/{model_name}...")
        
        api.upload_folder(
            folder_path=str(output_path),
            path_in_repo=model_name,  # Places files in task_set/model_name/
            repo_id=repo_id,
            repo_type="dataset"
        )
            
        click.echo(f"\n✅ Successfully uploaded files to {repo_id}/{model_name}")
        return True
        
    except Exception as e:
        click.echo(f"\n❌ Upload failed: {str(e)}")
        return False

@cli.command(name='bulk-upload')
@click.argument('submissions_dir', type=click.Path(exists=True))
@click.option('--task-set', type=str, required=True, help="Name of task set (e.g. public_eval_v1)")
@click.option('--org', type=str, default="arcprize", help="Hugging Face organization name")
@click.option('--public/--private', default=False, help="Make dataset public (default: private)")
def bulk_upload(submissions_dir, task_set, org, public):
    """Upload all model outputs from a submissions directory to Hugging Face."""
    click.echo(f"Starting bulk upload...")
    
    submissions_path = Path(submissions_dir)
    repo_id = f"{org}/{task_set}"
    
    # Get all model directories
    model_dirs = [d for d in submissions_path.iterdir() if d.is_dir()]
    click.echo(f"Found {len(model_dirs)} model directories")
    
    api = HfApi()
    try:
        # Create or get dataset repository
        api.create_repo(
            repo_id=repo_id, 
            repo_type="dataset",
            private=not public,
            exist_ok=True
        )
        
        click.echo(f"Created/updated repository as {'public' if public else 'private'}")
        
        # Upload each model's files
        success_count = 0
        for model_dir in model_dirs:
            model_name = model_dir.name
            click.echo(f"\nUploading {model_name} to {repo_id}/{model_name}...")
            
            try:
                api.upload_folder(
                    folder_path=str(model_dir),
                    path_in_repo=model_name,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
                click.echo(f"✅ Successfully uploaded {model_name}")
                success_count += 1
            except Exception as e:
                click.echo(f"❌ Failed to upload {model_name}: {str(e)}")
        
        click.echo(f"\nBulk upload complete: {success_count}/{len(model_dirs)} models uploaded successfully")
        return success_count == len(model_dirs)
        
    except Exception as e:
        click.echo(f"\n❌ Bulk upload failed: {str(e)}")
        return False

if __name__ == '__main__':
    cli() 