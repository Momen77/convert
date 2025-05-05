# Import necessary libraries
import streamlit as st
import pandas as pd
import io # Used for creating the download button data
import re # For normalization function
# import numpy as np # Not strictly needed for this version

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide")
st.title("📊 Multi-CSV Column Structure Converter") # Updated title
st.write("""
Upload **one or more 'Approval Details' CSV files** and **one 'Claim Details' CSV file (Template)**.
The app will convert each Approval file to match the Claim file structure using predefined rules:
- **'TPA Ref.' column generated as 'YY/ApprovalID' (from Accident Date YY and Approval ID).**
- 'Total Amount To Pay' column populated from 'Insurance Share Amount'.
Use the options below to include/exclude the 'Service Chapter' column and manage extra columns from the Approval file.
The results from all processed Approval files will be combined into a single downloadable CSV.
""")

# --- Normalization Function (Not used directly, but kept for reference if needed later) ---
def normalize_column_name_aggr(name):
    """Aggressive normalization used for mapping analysis."""
    if isinstance(name, str):
        name = name.lower() # Lowercase
        name = re.sub(r'\[.*?\]', '', name) # Remove content in brackets
        name = re.sub(r'\(.*?\)', '', name) # Remove content in parentheses
        name = re.sub(r'[_\-.,;:]+', ' ', name) # Replace separators with space
        name = name.strip() # Strip leading/trailing whitespace
        name = re.sub(r'\s+', ' ', name) # Consolidate multiple spaces
    return name

# --- Hardcoded AGGRESSIVE Mapping ---
# Format: { 'Claim Column Name': 'Approval Column Name' }
# 'TPA Ref.' is REMOVED from here as it requires custom logic
AGGRESSIVE_PREDETERMINED_MAPPING = {
    # 'TPA Ref.': 'Approval ID', # Removed - handled by custom logic below
    'Approval ID': 'Approval ID',
    'Insured Name': 'Insured Full Name [Loc.]', # Using local name
    'Card Number': 'Card Number',
    'Work ID': 'Work ID',
    'Alias Policy No': 'Alias Policy No',
    'Coverage Plan Name': 'Coverage Plan',
    'Policy Holder': 'Client Organization Name', # User specified
    'Chronic Form ID': 'Chronic Form ID',
    'Chronic': 'Diagnosis Is Chronic',
    'Preexisting': 'Diagnosis Is Preexisting',
    'Claim Type': 'Claim Type',
    'Claim Status': 'Approval Status', # Aggressive semantic match
    'Claim Form Type': 'Approval Type', # Aggressive semantic match
    'Main Provider No': 'Main Provider No',
    'Main Provider Name': 'Main Provider Name',
    'Treatment Doctor Name': 'Treatment Doctor Name',
    'Prescription Code': 'Prescription Code',
    'Accident Date': 'Accident Date', # Keep original Accident Date mapping if needed elsewhere
    'Service Code': 'Detail Service Code', # Matched 'service code' keyword
    'Service Name': 'Detail Service Name', # Matched 'service name' keyword
    # 'Service Chapter' is handled by the checkbox option below
    'Medical Diagnosis Code': 'Diagnosis Code', # Matched 'diagnosis code' keyword
    'Medical Diagnosis Name': 'Diagnosis Name', # Matched 'diagnosis name' keyword
    'Sub Product': 'Sub Product Detail Name', # Matched 'sub product' keyword
    'Accepted Amount': 'Accepted Amount',
    'Requested Amount': 'Requested Amount',
    'Deductible Amount': 'Deductible Amount',
    'Debit Amount': 'Debit Amount',
    'Total Amount To Pay': 'Insurance Share Amount', # User Rule
}


# --- File Uploaders ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Approval File(s)") # Updated header
    # **** ALLOW MULTIPLE FILES ****
    approval_files = st.file_uploader(
        "Upload one or more 'Approval Details' CSV files",
        type=["csv"],
        key="approval_uploader",
        accept_multiple_files=True # Key change here
    )

with col2:
    st.header("2. Upload Template File")
    claim_file = st.file_uploader(
        "Upload ONE 'Claim Details - Service Level' CSV file", # Emphasize ONE
        type=["csv"],
        key="claim_uploader",
        accept_multiple_files=False # Ensure only one template file
    )

# --- Main Logic ---
# **** Check if list of approval_files is not empty ****
if approval_files and claim_file is not None: # Check if approval_files list has content
    try:
        # Read the TEMPLATE file ONCE
        claim_df = pd.read_csv(claim_file, low_memory=False)
        st.success(f"Template file '{claim_file.name}' loaded successfully.")

        st.divider() # Divider after file uploaders

        # --- Configuration Options (Set ONCE for all files) ---
        st.header("3. Configuration Options")

        include_service_chapter = st.checkbox(
            "Include 'Service Chapter' column in output (if present in Claim file)",
            value=True, # Default to including it
            key="include_svc_chapter"
        )

        st.subheader("Options for Additional Columns")
        st.write("""Select which columns from the **original Approval files** you want to include in the final output, in addition to the columns matching the Claim file structure.""")

        # Determine extra columns config (applied to all files)
        keep_extra_mode = st.radio(
            "Choose how to handle extra columns from the Approval file:",
            (
                "Keep only columns matching Claim template structure",
                "Keep ALL original Approval columns (append extras)",
                "Keep SPECIFIC original Approval columns (select below)"
            ),
            index=0,
            key="keep_extra_mode"
        )
        # Note: Selecting specific columns might be less useful if different files have different extra columns.
        # We'll need the columns from the *first* approval file to populate the multiselect options.
        # This assumes all approval files have a reasonably similar structure for the 'specific' option to work well.
        specific_cols_to_keep = []
        if keep_extra_mode == "Keep SPECIFIC original Approval columns (select below)":
            # Read first approval file just to get column names for multiselect
            try:
                first_approval_df_cols = pd.read_csv(approval_files[0], low_memory=False, nrows=0).columns.tolist()
                approval_cols_used_in_mapping = list(AGGRESSIVE_PREDETERMINED_MAPPING.values())
                approval_cols_used_in_mapping_unique = list(set(approval_cols_used_in_mapping))
                approval_cols_not_mapped = [
                    col for col in first_approval_df_cols
                    if col not in approval_cols_used_in_mapping_unique
                ]
                available_options = [col for col in approval_cols_not_mapped if col in first_approval_df_cols] # Redundant check, but safe
                specific_cols_to_keep = st.multiselect(
                    "Select specific Approval columns (based on first file) to keep:",
                    options=available_options,
                    key="specific_cols_select"
                )
            except Exception as e:
                st.error(f"Could not read columns from the first approval file to populate specific column selection: {e}")
                keep_extra_mode = "Keep only columns matching Claim template structure" # Fallback


        st.divider()
        st.header("4. Convert and Download")

        # Button to trigger the conversion for ALL uploaded files
        if st.button(f"🔄 Convert {len(approval_files)} Approval File(s)"):

            all_converted_dfs = [] # List to store processed dataframes
            files_processed_count = 0
            files_failed_count = 0

            # --- Determine Target Columns based on Checkbox (Done ONCE) ---
            all_claim_columns = claim_df.columns.tolist()
            if include_service_chapter:
                target_columns = all_claim_columns
            else:
                target_columns = [col for col in all_claim_columns if col != 'Service Chapter']
                if 'Service Chapter' in all_claim_columns:
                     st.info("Note: 'Service Chapter' column explicitly excluded based on selection.")

            # --- Loop through each uploaded approval file ---
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, current_approval_file in enumerate(approval_files):
                status_text.text(f"Processing file {i+1}/{len(approval_files)}: {current_approval_file.name}...")
                try:
                    # Read the current approval file
                    approval_df = pd.read_csv(current_approval_file, low_memory=False)

                    # --- Core Conversion based on Template ---
                    converted_df = pd.DataFrame()
                    # Initialize all target columns to NA first
                    for col in target_columns:
                         converted_df[col] = pd.NA

                    # Apply mappings from the dictionary
                    for target_col in target_columns:
                        if target_col == 'TPA Ref.': continue # Skip TPA Ref here

                        if target_col in AGGRESSIVE_PREDETERMINED_MAPPING:
                            source_col = AGGRESSIVE_PREDETERMINED_MAPPING[target_col]
                            if source_col in approval_df.columns:
                                converted_df[target_col] = approval_df[source_col].copy()
                            # else: keep NA

                    # --- Custom Logic for 'TPA Ref.' ---
                    if 'TPA Ref.' in target_columns:
                        if 'Accident Date' in approval_df.columns and 'Approval ID' in approval_df.columns:
                            try:
                                accident_dates = pd.to_datetime(approval_df['Accident Date'].astype(str), errors='coerce')
                                approval_ids = approval_df['Approval ID']
                                tpa_ref_series = pd.Series([pd.NA] * len(approval_df), index=approval_df.index).astype(object)
                                valid_mask = accident_dates.notna() & approval_ids.notna()
                                if valid_mask.any():
                                    years = accident_dates[valid_mask].dt.strftime('%y')
                                    ids_str = approval_ids[valid_mask].astype(str)
                                    tpa_ref_series.loc[valid_mask] = years + '/' + ids_str
                                converted_df['TPA Ref.'] = tpa_ref_series
                            except Exception as e:
                                st.warning(f"File '{current_approval_file.name}': Could not generate 'TPA Ref.' due to an error: {e}")
                        else:
                            st.warning(f"File '{current_approval_file.name}': Could not generate 'TPA Ref.' (missing 'Accident Date' or 'Approval ID').")


                    # --- Handle Additional Columns ---
                    added_original_cols = []
                    approval_cols_original_current = approval_df.columns.tolist() # Use columns from *current* file
                    approval_cols_used_in_mapping_unique = list(set(AGGRESSIVE_PREDETERMINED_MAPPING.values()))
                    approval_cols_not_mapped_current = [
                         col for col in approval_cols_original_current
                         if col not in approval_cols_used_in_mapping_unique
                    ]

                    if keep_extra_mode == "Keep ALL original Approval columns (append extras)":
                        for col in approval_cols_not_mapped_current:
                             if col in approval_df.columns and col not in converted_df.columns:
                               converted_df[col] = approval_df[col].copy()
                               added_original_cols.append(col)

                    elif keep_extra_mode == "Keep SPECIFIC original Approval columns (select below)":
                        for col in specific_cols_to_keep: # Use selection based on first file
                            if col in approval_df.columns and col not in converted_df.columns: # Check if col exists in *current* file
                                converted_df[col] = approval_df[col].copy()
                                added_original_cols.append(col)
                    # else: only template structure kept

                    # --- Final Column Ordering for this file ---
                    final_target_order = target_columns
                    desired_column_order = final_target_order + added_original_cols
                    desired_column_order_unique = list(dict.fromkeys(desired_column_order))
                    final_columns_present = [col for col in desired_column_order_unique if col in converted_df.columns]
                    converted_df = converted_df[final_columns_present]

                    # Append the processed dataframe to the list
                    all_converted_dfs.append(converted_df)
                    files_processed_count += 1

                except Exception as e:
                    st.error(f"Failed to process file '{current_approval_file.name}': {e}")
                    files_failed_count += 1

                # Update progress bar
                progress_bar.progress((i + 1) / len(approval_files))

            status_text.text(f"Processing finished. {files_processed_count} file(s) processed successfully, {files_failed_count} file(s) failed.")

            # --- Concatenate results and provide download ---
            if all_converted_dfs:
                # Use concat to combine all dataframes in the list
                final_combined_df = pd.concat(all_converted_dfs, ignore_index=True)

                st.subheader("✅ Combined Converted Data Preview (First 5 Rows)")
                st.dataframe(final_combined_df.head())
                st.info(f"Final combined output has {len(final_combined_df.columns)} columns and {len(final_combined_df)} rows.")
                st.success("Conversion complete!")

                # --- Download Button for Combined File ---
                output = io.BytesIO()
                final_combined_df.to_csv(output, index=False, encoding='utf-8')
                output.seek(0)

                st.download_button(
                    label="💾 Download Combined CSV",
                    data=output,
                    file_name="combined_converted_approvals.csv", # New filename
                    mime="text/csv",
                )
            else:
                st.warning("No dataframes were successfully processed to combine.")


    except FileNotFoundError:
         st.error("Error: The Template (Claim Details) file could not be read. Please ensure it's a valid CSV.")
    except KeyError as e:
         st.error(f"Error: A column specified in the mapping ('{e}') was not found in the expected source file. Please check the uploaded files.")
    except Exception as e:
         st.error(f"An unexpected error occurred during setup or processing: {e}")
         st.error("Please ensure the uploaded files are valid CSVs and try again.")

elif not approval_files and claim_file is not None:
     st.info("Please upload at least one Approval file.")
elif approval_files and claim_file is None:
     st.info("Please upload the Claim (Template) file.")
else: # Neither uploaded
     st.info("Please upload Approval file(s) and the Claim (Template) file to begin.")