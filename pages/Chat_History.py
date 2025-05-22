"""
Chat History page for the SilentCodingLegend AI agent.
This page displays the chat history stored in the session state and backs up data to the Chat_History directory.
It combines entries from all sources including Main Chat, Deep Thinking & Reasoning, Vision Analysis, and Coding Assistant.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
from src.utils import apply_dark_theme

# Page configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="Chat History",
    page_icon="📜",
    layout="wide"
)

# Apply dark theme using utility function
apply_dark_theme()

def backup_chat_history(history_data):
    """Backup chat history to the Chat_History directory."""
    # Create the backup directory if it doesn't exist
    backup_dir = Path("/home/silentlegendkali/scl-openrouter/Chat_History")
    backup_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Backup as CSV
    csv_path = backup_dir / f"chat_history_{timestamp}.csv"
    pd.DataFrame(history_data).to_csv(csv_path, index=False)
    
    # Backup as JSON
    json_path = backup_dir / f"chat_history_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    
    return csv_path, json_path

def load_backup_file(file_path):
    """Load a backup file and return the chat history data."""
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path).to_dict('records')
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

def load_all_backup_files():
    """Load and combine all backup files from Chat_History directory."""
    backup_dir = Path("/home/silentlegendkali/scl-openrouter/Chat_History")
    all_backups = []
    
    # Get all JSON backup files
    json_files = list(backup_dir.glob("*.json"))
    
    # Load data from each file
    for file_path in json_files:
        try:
            # Check if it's a coding, vision, or regular history file
            file_type = "main"
            if "coding_history" in file_path.name:
                file_type = "coding"
            elif "vision_history" in file_path.name:
                file_type = "vision"
            elif "reasoning_history" in file_path.name:
                file_type = "reasoning"
                
            # Load the data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Ensure each entry has a source field if it's from a specialized page
            if isinstance(data, list):
                for entry in data:
                    if "source" not in entry:
                        if file_type == "coding":
                            entry["source"] = "Coding Assistant"
                        elif file_type == "vision":
                            entry["source"] = "Vision Analysis"
                        elif file_type == "reasoning":
                            entry["source"] = "Deep Thinking & Reasoning"
                        else:
                            entry["source"] = "Main Chat"
                all_backups.extend(data)
        except Exception as e:
            # Skip problematic files
            continue
    
    # Sort by timestamp if available
    try:
        all_backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    except:
        pass
        
    return all_backups

def app():
    st.title("Chat History")
    
    # Initialize chat history in session state if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Track if we need to back up history
    if "last_history_length" not in st.session_state:
        st.session_state.last_history_length = 0
    
    # Auto-backup if history has changed
    current_length = len(st.session_state.chat_history)
    if current_length > 0 and current_length != st.session_state.last_history_length:
        # Only back up if there's history and it has changed
        history_data = []
        for entry in st.session_state.chat_history:
            history_data.append({
                "timestamp": entry.get("timestamp", "Unknown"),
                "model": entry.get("model", "Unknown"),
                "prompt": entry.get("prompt", ""),
                "response": entry.get("response", ""),
                "source": entry.get("source", "Main Chat")  # Include source information
            })
        
        # Perform backup
        csv_path, json_path = backup_chat_history(history_data)
        st.session_state.last_history_length = current_length
        st.success(f"Auto-backup created: {json_path.name}")
    
    # Load history from session state AND backup files
    combined_history = []
    
    # First, add entries from session state
    if st.session_state.chat_history:
        for entry in st.session_state.chat_history:
            # Create a copy of the entry to avoid modifying the original
            history_item = entry.copy()
            
            # Ensure there's a source field
            if "source" not in history_item:
                history_item["source"] = "Main Chat"
                
            combined_history.append(history_item)
    
    # Then load all backup files
    backup_entries = load_all_backup_files()
    
    # Add unique entries from backups
    seen_entries = set()
    for entry in combined_history:
        # Create a signature for the entry using prompt and timestamp
        signature = f"{entry.get('timestamp', '')}_{entry.get('prompt', '')[:50]}"
        seen_entries.add(signature)
    
    for entry in backup_entries:
        # Create a signature for this entry
        signature = f"{entry.get('timestamp', '')}_{entry.get('prompt', '')[:50]}"
        
        # If this is a new entry, add it
        if signature not in seen_entries:
            combined_history.append(entry)
            seen_entries.add(signature)
    
    # Sort by timestamp
    try:
        combined_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    except:
        pass
    
    # Display chat history if available
    if combined_history:
        # Create a DataFrame for easier display
        history_data = []
        for entry in combined_history:
            history_item = {
                "timestamp": entry.get("timestamp", "Unknown"),
                "model": entry.get("model", "Unknown"),
                "prompt": entry.get("prompt", ""),
                "response": entry.get("response", "")
            }
            
            # Add source if it exists
            if "source" in entry:
                history_item["source"] = entry["source"]
            else:
                history_item["source"] = "Main Chat"
                
            history_data.append(history_item)
        
        # Create tabs for different sections
        view_tab, backup_tab, restore_tab, manage_tab = st.tabs([
            "View History", "Backup Options", "Restore Backup", "Backup Management"
        ])
        
        with view_tab:
            # Source statistics
            sources = [entry.get('source', 'Main Chat') for entry in history_data]
            source_counts = {source: sources.count(source) for source in set(sources)}
            
            # Create a nicer statistics section with progress bars
            st.markdown("### Chat History Statistics")
            
            total_entries = len(history_data)
            st.write(f"Total entries: {total_entries}")
            
            # Add progress bars for each source
            for source, count in source_counts.items():
                percentage = int((count / total_entries) * 100)
                st.write(f"{source}: {count} entries ({percentage}%)")
                st.progress(count / total_entries)
            
            # Add a divider
            st.markdown("---")
            
            # Add filter options - now including Coding Assistant
            filter_options = ["All", "Main Chat", "Deep Thinking & Reasoning", "Coding Assistant", "Vision Analysis"]
            selected_filter = st.radio("Filter by source:", filter_options, horizontal=True)
            
            # Filter the entries based on the selection
            filtered_history = history_data
            if selected_filter != "All":
                filtered_history = [entry for entry in history_data 
                                    if entry.get('source', 'Main Chat') == selected_filter]
            
            if not filtered_history:
                st.info(f"No entries found for {selected_filter}")
            
            # Display each chat entry in an expander
            for i, entry in enumerate(filtered_history):
                # Check if the entry has a source and assign appropriate icon
                source = entry.get('source', 'Main Chat')
                icon = "💬"  # Default icon
                
                # Assign specific icons based on source
                if source == "Deep Thinking & Reasoning":
                    icon = "🧠"
                elif source == "Coding Assistant":
                    icon = "�"
                elif source == "Vision Analysis":
                    icon = "👁️"
                
                with st.expander(f"{icon} Chat {i+1} - {entry['timestamp']} - {entry['model']} ({source})"):
                    st.markdown("**User Prompt:**")
                    st.markdown(entry["prompt"])
                    st.markdown("**AI Response:**")
                    st.markdown(entry["response"])
            
            # Add button to clear history
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        with backup_tab:
            # Convert to DataFrame
            history_df = pd.DataFrame(history_data)
            
            # Backup to Chat_History directory
            csv_path, json_path = backup_chat_history(history_data)
            st.success(f"Chat history backed up to {csv_path.name} and {json_path.name}")
            
            # Download buttons for chat history
            col1, col2 = st.columns(2)
            with col1:
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_str = json.dumps(history_data, indent=2)
                st.download_button(
                    label="Download as JSON",
                    data=json_str,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
        with restore_tab:
            # Display backup files
            backup_dir = Path("/home/silentlegendkali/scl-openrouter/Chat_History")
            
            # Get all backup files
            backup_files = list(backup_dir.glob("*.json"))  # Prefer JSON for restoration
            if not backup_files:
                backup_files = list(backup_dir.glob("*.csv"))
            
            if backup_files:
                # Sort files by modification time (newest first)
                backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Create a list of options with file names and timestamps
                file_options = [f"{file.name} ({datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})" 
                                for file in backup_files]
                
                selected_file_idx = st.selectbox(
                    "Select a backup file to restore:",
                    options=range(len(file_options)),
                    format_func=lambda i: file_options[i],
                    key="restore_file"
                )
                
                if st.button("Restore Selected Backup"):
                    selected_file = backup_files[selected_file_idx]
                    restored_data = load_backup_file(selected_file)
                    if restored_data:
                        st.session_state.chat_history = restored_data
                        st.success(f"Successfully restored chat history from {selected_file.name}")
                        st.rerun()
                    else:
                        st.error(f"Failed to restore from {selected_file.name}")
            else:
                st.info("No backup files available for restoration.")
        
        with manage_tab:
            # Backup management
            # Calculate total size of backup files
            all_backup_files = list(backup_dir.glob("*.csv")) + list(backup_dir.glob("*.json"))
            total_size_kb = sum(file.stat().st_size for file in all_backup_files) / 1024
            
            st.write(f"Total backup files: {len(all_backup_files)}")
            st.write(f"Total size: {total_size_kb:.2f} KB")
            
            # Show list of backup files
            with st.expander("View Available Backup Files"):
                # List CSV files
                csv_files = list(backup_dir.glob("*.csv"))
                if csv_files:
                    st.markdown("### CSV Files")
                    for file in sorted(csv_files, reverse=True):
                        st.write(f"- {file.name} ({file.stat().st_size / 1024:.1f} KB)")
                
                # List JSON files
                json_files = list(backup_dir.glob("*.json"))
                if json_files:
                    st.markdown("### JSON Files")
                    for file in sorted(json_files, reverse=True):
                        st.write(f"- {file.name} ({file.stat().st_size / 1024:.1f} KB)")
            
            # Options to delete old backups
            if all_backup_files:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Delete Oldest Backup"):
                        oldest_file = min(all_backup_files, key=lambda x: x.stat().st_mtime)
                        try:
                            oldest_file.unlink()
                            st.success(f"Deleted {oldest_file.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {e}")
                
                with col2:
                    if st.button("Delete All Backups", type="primary"):
                        confirm = st.checkbox("Confirm deletion of ALL backup files", key="confirm_delete_all")
                        if confirm:
                            try:
                                for file in all_backup_files:
                                    file.unlink()
                                st.success(f"Deleted {len(all_backup_files)} backup files")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting files: {e}")
                        else:
                            st.warning("Please confirm deletion by checking the box")
    else:
        st.info("No chat history available yet. Start a conversation from the main page.")

if __name__ == "__main__":
    app()