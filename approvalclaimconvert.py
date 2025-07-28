# Import necessary libraries
import streamlit as st
import pandas as pd
import io # Used for creating the download button data
import re # For normalization function
# import numpy as np # Not strictly needed for this version

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide")
st.title("📊 Multi-CSV Column Structure Converter")
st.write("""
Upload **one or more 'Approval Details' CSV files** and **one 'Claim Details' CSV file (Template)**.
The app converts each Approval file to match the Claim file structure using predefined rules.
- **'TPA Ref.' column generated as 'YY/ApprovalID' (from Accident Date YY and Approval ID).**
- 'Total Amount To Pay' column populated from 'Insurance Share Amount'.
- **'Underwriting Year' will be filled (if missing) based on the earliest dates found in the Claim file.**
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
    # 'Underwriting Year' is not mapped here, filled later
}

# --- Helper function to read CSV with encoding fallback ---
def read_csv_with_encoding_fallback(uploaded_file, **kwargs):
    """
    Tries to read a CSV file with common encodings (UTF-8, latin-1, cp1252).
    Resets the file pointer before each attempt.
    Passes any additional kwargs (like nrows) to pd.read_csv.
    """
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig'] # Common encodings
    df = None
    last_exception = None

    read_kwargs = {'low_memory': False, **kwargs}

    if uploaded_file is None:
        raise ValueError("No file provided to read.")

    file_name = getattr(uploaded_file, 'name', 'Unknown File')

    for encoding in encodings_to_try:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding, **read_kwargs)
            return df
        except UnicodeDecodeError as e:
            last_exception = e
        except Exception as e:
            last_exception = e
            break

    if df is None:
        error_msg = f"Error reading file '{file_name}'. Could not decode using common encodings {encodings_to_try} or another error occurred."
        if last_exception:
             error_msg += f" Last error: {last_exception}"
        st.error(error_msg)
        raise ValueError(f"Failed to read CSV '{file_name}'.")

# --- File Uploaders ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Approval File(s)")
    approval_files = st.file_uploader(
        "Upload one or more 'Approval Details' CSV files",
        type=["csv"],
        key="approval_uploader",
        accept_multiple_files=True
    )

with col2:
    st.header("2. Upload Template File")
    claim_file = st.file_uploader(
        "Upload ONE 'Claim Details - Service Level' CSV file",
        type=["csv"],
        key="claim_uploader",
        accept_multiple_files=False
    )

# --- Process Claim File Data for Underwriting Year Logic and Claims Enhancement ---
# Store results in session state to avoid recalculation if only other options change
if claim_file is not None:
    # Check if the file is new or hasn't been processed yet
    if 'claim_file_name_processed' not in st.session_state or st.session_state['claim_file_name_processed'] != claim_file.name:
        try:
            claim_df_for_uw = read_csv_with_encoding_fallback(claim_file)
            st.session_state['claim_df_for_uw'] = claim_df_for_uw # Store for later use
            st.session_state['claim_file_name_processed'] = claim_file.name
            st.success(f"Template file '{claim_file.name}' loaded successfully.")
            
            # Create claims enhancement index for matching
            if 'Approval ID' in claim_df_for_uw.columns and 'Service Code' in claim_df_for_uw.columns:
                # Create composite key for claims data
                claim_df_for_uw['_composite_key'] = claim_df_for_uw['Approval ID'].astype(str) + '_' + claim_df_for_uw['Service Code'].astype(str)
                st.session_state['claims_enhancement_data'] = claim_df_for_uw
                st.success(f"Claims enhancement index created with {len(claim_df_for_uw)} records.")
            else:
                st.warning("Claims enhancement disabled: 'Approval ID' or 'Service Code' columns not found in claims file.")
                st.session_state['claims_enhancement_data'] = None

            # --- Calculate Earliest Dates per Underwriting Year ---
            # *** CORRECTED COLUMN NAME ***
            uw_year_col = 'Underwriting Year'
            # *** ASSUMPTION: Using 'Accident Date' from Claim file to determine start ***
            date_col_for_uw = 'Accident Date' # Change if needed

            year_start_dates = {}
            if uw_year_col in claim_df_for_uw.columns and date_col_for_uw in claim_df_for_uw.columns:
                try:
                    temp_df = claim_df_for_uw[[uw_year_col, date_col_for_uw]].copy()
                    temp_df.dropna(subset=[uw_year_col, date_col_for_uw], inplace=True)
                    temp_df[date_col_for_uw] = pd.to_datetime(temp_df[date_col_for_uw], errors='coerce')
                    temp_df.dropna(subset=[date_col_for_uw], inplace=True)

                    if not temp_df.empty:
                         min_dates_series = temp_df.groupby(uw_year_col)[date_col_for_uw].min()
                         year_start_dates = min_dates_series.to_dict()
                         st.session_state['year_start_dates'] = year_start_dates
                         st.success(f"Calculated start dates for {len(year_start_dates)} Underwriting Year(s) from Claim file.")
                    else:
                         st.warning(f"No valid pairs of '{uw_year_col}' and '{date_col_for_uw}' found in Claim file.")
                         st.session_state['year_start_dates'] = {}

                except Exception as e:
                    st.error(f"Error processing Claim file for Underwriting Year start dates: {e}")
                    st.session_state['year_start_dates'] = {}
            else:
                # Corrected column name in warning
                st.warning(f"Cannot determine Underwriting Year start dates: Required columns ('{uw_year_col}', '{date_col_for_uw}') not found in Claim file.")
                st.session_state['year_start_dates'] = {}

        except Exception as e:
            st.error(f"Failed to read or process Claim Template file '{claim_file.name}': {e}")
            if 'claim_df_for_uw' in st.session_state: del st.session_state['claim_df_for_uw']
            if 'year_start_dates' in st.session_state: del st.session_state['year_start_dates']
            if 'claim_file_name_processed' in st.session_state: del st.session_state['claim_file_name_processed']
            st.stop()

# --- Configuration Options ---
st.divider()
st.header("3. Configuration Options")

claim_df_loaded = 'claim_df_for_uw' in st.session_state and st.session_state['claim_df_for_uw'] is not None

if claim_df_loaded:
    claim_df_cols = st.session_state['claim_df_for_uw'].columns.tolist()

    include_service_chapter = st.checkbox(
        "Include 'Service Chapter' column in output (if present in Claim file)",
        value=True,
        key="include_svc_chapter"
    )
    
    enhance_with_claims = st.checkbox(
        "Enhance output with claims data (merge matching Approval ID + Service Code)",
        value=True,
        key="enhance_claims",
        help="When enabled, rows with matching Approval ID and Service Code will include all available data from the claims file"
    )

    st.subheader("Options for Additional Columns from Approval Files")
    st.write("""Select which columns from the **original Approval files** you want to include in the final output, in addition to the columns matching the Claim file structure.""")
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

    specific_cols_to_keep = []
    if keep_extra_mode == "Keep SPECIFIC original Approval columns (select below)":
        if approval_files:
             try:
                 first_approval_df_for_cols = read_csv_with_encoding_fallback(approval_files[0], nrows=0)
                 first_approval_df_cols = first_approval_df_for_cols.columns.tolist()
                 approval_cols_used_in_mapping = list(AGGRESSIVE_PREDETERMINED_MAPPING.values())
                 approval_cols_used_in_mapping_unique = list(set(approval_cols_used_in_mapping))
                 approval_cols_not_mapped = [
                     col for col in first_approval_df_cols
                     if col not in approval_cols_used_in_mapping_unique
                 ]
                 available_options = [col for col in approval_cols_not_mapped if col in first_approval_df_cols]
                 specific_cols_to_keep = st.multiselect(
                     "Select specific Approval columns (based on first file) to keep:",
                     options=available_options,
                     key="specific_cols_select"
                 )
             except Exception as e:
                 st.warning(f"Could not read columns from the first approval file: {e}. Specific selection may fail.")
        else:
            st.warning("Upload an Approval file to select specific columns.")
else:
    st.info("Upload the Claim Template file to see configuration options.")


# --- Processing Logic ---
st.divider()
st.header("4. Convert and Download")

if approval_files and claim_df_loaded:
    if st.button(f"🔄 Process {len(approval_files)} Approval File(s)"):

        all_converted_dfs = []
        files_processed_count = 0
        files_failed_count = 0

        # --- Determine Target Columns ---
        claim_df = st.session_state['claim_df_for_uw']
        all_claim_columns = claim_df.columns.tolist()
        target_uw_col = 'Underwriting Year' # Corrected Name

        if include_service_chapter:
            target_columns = all_claim_columns
        else:
            target_columns = [col for col in all_claim_columns if col != 'Service Chapter']

        # Ensure Underwriting Year is targeted if present in template
        if target_uw_col not in target_columns and target_uw_col in all_claim_columns:
             target_columns.append(target_uw_col)

        # --- Loop through each uploaded approval file ---
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, current_approval_file in enumerate(approval_files):
            file_name_for_msg = current_approval_file.name
            status_text.text(f"Processing file {i+1}/{len(approval_files)}: {file_name_for_msg}...")
            try:
                approval_df = read_csv_with_encoding_fallback(current_approval_file)
                converted_df = pd.DataFrame()
                for col in target_columns:
                    converted_df[col] = pd.NA

                # Apply mappings
                for target_col in target_columns:
                    if target_col in ['TPA Ref.', target_uw_col]: continue # Skip TPA Ref and UW Year here

                    if target_col in AGGRESSIVE_PREDETERMINED_MAPPING:
                        source_col = AGGRESSIVE_PREDETERMINED_MAPPING[target_col]
                        if source_col in approval_df.columns:
                            converted_df[target_col] = approval_df[source_col].copy()
                            
                # Enhanced Claims Data Integration
                enhance_with_claims_enabled = st.session_state.get('enhance_claims', True)
                claims_data = st.session_state.get('claims_enhancement_data', None)
                
                if enhance_with_claims_enabled and claims_data is not None:
                    try:
                        # Create composite key for approval data
                        if 'Approval ID' in approval_df.columns and 'Detail Service Code' in approval_df.columns:
                            approval_df['_composite_key'] = approval_df['Approval ID'].astype(str) + '_' + approval_df['Detail Service Code'].astype(str)
                            
                            # Create a mapping dictionary from claims data
                            claims_dict = {}
                            for idx, row in claims_data.iterrows():
                                key = row['_composite_key']
                                claims_dict[key] = row.to_dict()
                            
                            # Enhance converted_df with claims data
                            enhanced_rows = []
                            for idx, approval_row in approval_df.iterrows():
                                composite_key = approval_row['_composite_key']
                                
                                # Start with the converted row
                                enhanced_row = converted_df.iloc[idx].to_dict()
                                
                                # If we have matching claims data, merge it
                                if composite_key in claims_dict:
                                    claims_row = claims_dict[composite_key]
                                    
                                    # Merge claims data, prioritizing non-null values from claims
                                    for col, claims_value in claims_row.items():
                                        if col != '_composite_key' and pd.notna(claims_value):
                                            # Special handling for TPA Ref - always use claims value if available
                                            if col == 'TPA Ref.' and pd.notna(claims_value):
                                                enhanced_row[col] = claims_value
                                            # If the column exists in our target and is null/empty, fill it
                                            elif col in enhanced_row and (pd.isna(enhanced_row[col]) or enhanced_row[col] == ''):
                                                enhanced_row[col] = claims_value
                                            # If it's a new column not in target, add it if we're keeping extra columns
                                            elif col not in enhanced_row and col in claims_data.columns:
                                                enhanced_row[col] = claims_value
                                
                                enhanced_rows.append(enhanced_row)
                            
                            # Rebuild converted_df with enhanced data
                            if enhanced_rows:
                                enhanced_df = pd.DataFrame(enhanced_rows)
                                # Ensure we maintain the original column order plus any new ones
                                original_cols = [col for col in converted_df.columns if col in enhanced_df.columns]
                                new_cols = [col for col in enhanced_df.columns if col not in original_cols]
                                final_col_order = original_cols + new_cols
                                converted_df = enhanced_df[final_col_order]
                                
                            # Clean up temporary column
                            if '_composite_key' in approval_df.columns:
                                approval_df.drop('_composite_key', axis=1, inplace=True)
                                
                            st.info(f"File '{file_name_for_msg}': Enhanced with claims data using Approval ID + Service Code matching.")
                        else:
                            st.warning(f"File '{file_name_for_msg}': Claims enhancement skipped - missing required columns.")
                    except Exception as e:
                        st.warning(f"File '{file_name_for_msg}': Claims enhancement failed: {e}")

                # Custom Logic for 'TPA Ref.'
                if 'TPA Ref.' in target_columns:
                    if 'Accident Date' in approval_df.columns and 'Approval ID' in approval_df.columns:
                        try:
                            approval_ids_str = approval_df['Approval ID'].astype(str)
                            accident_dates = pd.to_datetime(approval_df['Accident Date'], errors='coerce')
                            tpa_ref_series = pd.Series([pd.NA] * len(approval_df), index=approval_df.index).astype(object)
                            valid_mask = accident_dates.notna() & approval_ids_str.notna() & (approval_ids_str != '')
                            if valid_mask.any():
                                years = accident_dates[valid_mask].dt.strftime('%y')
                                ids_str_valid = approval_ids_str[valid_mask]
                                tpa_ref_series.loc[valid_mask] = years + '/' + ids_str_valid
                            converted_df['TPA Ref.'] = tpa_ref_series
                        except Exception as e:
                            st.warning(f"File '{file_name_for_msg}': Could not generate 'TPA Ref.': {e}")
                    else:
                        st.warning(f"File '{file_name_for_msg}': 'TPA Ref.' needs 'Accident Date' & 'Approval ID'.")

                # Handle Additional Columns
                added_original_cols = []
                if keep_extra_mode != "Keep only columns matching Claim template structure":
                    approval_cols_original_current = approval_df.columns.tolist()
                    approval_cols_used_in_mapping_values = list(AGGRESSIVE_PREDETERMINED_MAPPING.values())
                    approval_cols_used_in_mapping_unique = list(set(approval_cols_used_in_mapping_values))
                    cols_to_potentially_add = []
                    if keep_extra_mode == "Keep ALL original Approval columns (append extras)":
                         cols_to_potentially_add = [
                             col for col in approval_cols_original_current
                             if col not in approval_cols_used_in_mapping_unique
                         ]
                    elif keep_extra_mode == "Keep SPECIFIC original Approval columns (select below)":
                         selected_specifics = st.session_state.get('specific_cols_select', [])
                         cols_to_potentially_add = [
                             col for col in selected_specifics
                             if col in approval_cols_original_current
                         ]
                    for col in cols_to_potentially_add:
                         if col not in converted_df.columns:
                             converted_df[col] = approval_df[col].copy()
                             added_original_cols.append(col)

                # Final Column Ordering
                final_target_order_for_file = [col for col in target_columns if col in converted_df.columns]
                desired_column_order = final_target_order_for_file + added_original_cols
                seen = set()
                desired_column_order_unique = [x for x in desired_column_order if not (x in seen or seen.add(x))]
                final_columns_present = [col for col in desired_column_order_unique if col in converted_df.columns]
                converted_df = converted_df[final_columns_present]

                all_converted_dfs.append(converted_df)
                files_processed_count += 1

            except ValueError as e:
                 st.error(f"Skipping file '{file_name_for_msg}' due to read error: {e}")
                 files_failed_count += 1
            except KeyError as e:
                 st.error(f"Skipping file '{file_name_for_msg}': Processing error - column '{e}'.")
                 files_failed_count += 1
            except Exception as e:
                st.error(f"Failed to process file '{file_name_for_msg}': {e}")
                files_failed_count += 1

            progress_bar.progress((i + 1) / len(approval_files))

        status_text.text(f"Processing finished. {files_processed_count} file(s) processed, {files_failed_count} failed.")

        # --- Concatenate and Fill Underwriting Year ---
        if all_converted_dfs:
            final_combined_df = pd.concat(all_converted_dfs, ignore_index=True, join='outer')
            st.success(f"Successfully combined data from {files_processed_count} file(s).")

            # --- Fill Underwriting Year ---
            year_start_dates_dict = st.session_state.get('year_start_dates', {})
            uw_year_filled = False
            # *** Assumption: Using 'Accident Date' from combined file to determine UW Year ***
            date_col_for_filling = 'Accident Date' # Change if needed
            # *** CORRECTED COLUMN NAME ***
            target_uw_col = 'Underwriting Year'

            if not year_start_dates_dict:
                st.warning("Skipping 'Underwriting Year' fill: No start dates calculated.")
            # Use corrected name in checks
            elif target_uw_col not in final_combined_df.columns:
                 st.warning(f"Skipping 'Underwriting Year' fill: Column '{target_uw_col}' not found.")
            elif date_col_for_filling not in final_combined_df.columns:
                 st.warning(f"Skipping 'Underwriting Year' fill: Date column '{date_col_for_filling}' not found.")
            else:
                 try:
                     st.info("Attempting to fill missing 'Underwriting Year'...")
                     final_combined_df[date_col_for_filling] = pd.to_datetime(final_combined_df[date_col_for_filling], errors='coerce')
                     sorted_starts = sorted(year_start_dates_dict.items(), key=lambda item: item[1])

                     def get_underwriting_year(approval_date, year_start_dates_sorted):
                         if pd.isna(approval_date): return pd.NA
                         applicable_year = pd.NA
                         for year, start_date in year_start_dates_sorted:
                             if approval_date >= start_date:
                                 applicable_year = year
                             else: break
                         return applicable_year

                     valid_date_mask = final_combined_df[date_col_for_filling].notna()
                     if valid_date_mask.any():
                         calculated_uw_years = final_combined_df.loc[valid_date_mask, date_col_for_filling].apply(
                             lambda date: get_underwriting_year(date, sorted_starts)
                         )
                         # Use corrected name for filling
                         initial_nas = final_combined_df[target_uw_col].isna().sum()
                         final_combined_df[target_uw_col].fillna(calculated_uw_years, inplace=True)
                         final_nas = final_combined_df[target_uw_col].isna().sum()
                         filled_count = initial_nas - final_nas
                         if filled_count > 0:
                              st.success(f"Filled {filled_count} missing 'Underwriting Year' values.")
                              uw_year_filled = True
                         else: st.info("No missing 'Underwriting Year' values needed filling.")
                     else: st.info(f"No valid dates in '{date_col_for_filling}' for 'Underwriting Year'.")
                 except Exception as fill_e:
                      st.error(f"Error filling 'Underwriting Year': {fill_e}")

            # --- Display Final Preview and Download ---
            st.markdown("---")
            st.subheader("✅ Combined Data Preview (First 5 Rows)")
            st.dataframe(final_combined_df.head())
            st.info(f"Final output: {len(final_combined_df.columns)} columns, {len(final_combined_df)} rows.")

            output = io.BytesIO()
            final_combined_df.to_csv(output, index=False, encoding='utf-8-sig')
            output.seek(0)

            st.download_button(
                label="💾 Download Combined CSV",
                data=output,
                file_name="combined_converted_approvals_with_uw_year.csv",
                mime="text/csv",
            )
        elif files_processed_count == 0 and files_failed_count > 0:
             st.error("Processing failed for all uploaded Approval files.")
        else:
            st.warning("No dataframes were successfully processed or combined.")

elif not approval_files and claim_df_loaded:
     st.info("Upload at least one Approval file to enable processing.")
elif not claim_df_loaded:
     st.info("Upload the Claim Template file first.")
else:
     st.info("Upload Approval file(s) and the Claim Template file to begin.")
