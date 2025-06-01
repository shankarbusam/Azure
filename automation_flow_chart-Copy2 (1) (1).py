import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def load_bigquery_table(
    project_id: str,
    dataset_id: str,
    table_name: str,
    credentials_path: str
) -> pd.DataFrame:
    """
    Loads a BigQuery table into a pandas DataFrame.
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    client = bigquery.Client(project=project_id)

    table_ref = f"{project_id}.{dataset_id}.{table_name}"
    query = f"SELECT * FROM `{table_ref}`"
    query_job = client.query(query)

    return query_job.to_dataframe()

def is_material_status_normal(status: str) -> bool:
    """Check if material status is 'normal'."""
    return str(status).strip().lower() == 'normal'

def is_replenishable(mrp_type: str) -> bool:
    """Return True if replenishable (e.g., MRP type 'SB'), False for 'PD'."""
    return 'No' if mrp_type.strip().upper() == 'PD' else 'Yes'
        

def classify_consumption_on_replenishable(
    result_df: pd.DataFrame,
    historical_consumption_df: pd.DataFrame,
    date_col: str = 'Date',
    material_col: str = 'Material Number',
    quantity_col: str = 'Quantity',
    movement_type_col: str = 'Movement Type_Code',
    valid_movement_types: list = [201, 261, 281],
    month_threshold: int = 10
) -> pd.DataFrame:
    """
    Classifies materials based on historical consumption.
    Ensures all materials from result_df appear in the output even with 0 consumption.
    """
    # Step 1: Extract unique materials from result_df
    rep_materials = result_df['Material Number'].unique()
    print("historical consumption cols are",historical_consumption_df.columns)
    # Step 2: Filter historical data by material and movement types
    historical_consumption_df.rename(columns={'Material':'Material Number'},inplace=True)
    df_filtered = historical_consumption_df[
        historical_consumption_df[movement_type_col].isin(valid_movement_types) &
        historical_consumption_df[material_col].isin(rep_materials)
    ].copy()
 
    df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
    today = datetime.today()
    year, month = today.year, today.month
 
    # Step 3: Define year buckets
    if month >= month_threshold:
        year_ranges = {
            'Y-1': (pd.Timestamp(f'{year}-01-01'), today),
            'Y-2': (pd.Timestamp(f'{year-1}-01-01'), pd.Timestamp(f'{year-1}-12-31')),
            'Y-3': (pd.Timestamp(f'{year-2}-01-01'), pd.Timestamp(f'{year-2}-12-31')),
        }
    else:
        year_ranges = {
            'Y-1': (pd.Timestamp(f'{year-1}-01-01'), pd.Timestamp(f'{year-1}-12-31')),
            'Y-2': (pd.Timestamp(f'{year-2}-01-01'), pd.Timestamp(f'{year-2}-12-31')),
            'Y-3': (pd.Timestamp(f'{year-3}-01-01'), pd.Timestamp(f'{year-3}-12-31')),
        }
 
    # Step 4: Initialize result with all materials from result_df
    #result = pd.DataFrame(index=rep_materials)
 
    # Step 5: Populate consumption data for each year
    for label, (start, end) in year_ranges.items():
        mask = (df_filtered[date_col] >= start) & (df_filtered[date_col] <= end)
        yearly = df_filtered[mask].groupby(material_col)[quantity_col].sum().rename(label)
        result_df = result_df.merge(yearly, on=['Material Number'], how='left')
        result_df[label].fillna(0,inplace=True)
    print("result df is ",result_df.columns)
    # Step 6: Fill missing with 0s, compute totals and classify
    #result = result.fillna(0).astype(int)
    result_df['Total Quantity'] = result_df[['Y-1', 'Y-2', 'Y-3']].sum(axis=1)
    print("result head is",result_df.head())
    result_df['Is Consumable'] = result_df.apply(
        lambda row: 'Yes' if all(row[col] > 0 for col in ['Y-1', 'Y-2', 'Y-3']) else 'No',
        axis=1
    )
    print("after result cols for consumption are: ",result_df.columns)
    # Reset index and rename to include original material column
    #return result_df.reset_index().rename(columns={'index': material_col})
    return result_df

def classify_equipment_criticality(merged_df: pd.DataFrame, equipment_criticality_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify materials as Critical or Non-Critical based on equipment_criticality_df.
    Critical if Equipment Criticality is 'A' or 'S', otherwise Non-Critical.
    If material not found in equipment_criticality_df, set Is Critical to 'N/A'.
    """
    # Rename to align columns
    print("columns of equipment table",equipment_criticality_df.columns)
    print("columns of merge table",merged_df.columns)
    #equipment_criticality_df.rename(columns={'Material Number': 'Material'},inplace=True)
    print("columns of equipment table after are:",equipment_criticality_df.columns)
    # Filter only materials present in merged_df
    filtered_crit_df = equipment_criticality_df[
        equipment_criticality_df['Material Number'].isin(merged_df['Material Number'])
    ].copy()
    
    # Standardize criticality values
    filtered_crit_df['Equipment Criticality'] = filtered_crit_df['Equipment Criticality'].str.upper()
    
    # Classify as Critical or Non-Critical
    filtered_crit_df['Is Critical'] = filtered_crit_df['Equipment Criticality'].apply(
        lambda x: 'Yes' if x in ['A', 'S'] else 'No'
    )
    
    filtered_crit_df_unique = filtered_crit_df.groupby('Material Number')['Is Critical'].apply(
    lambda x: 'Yes' if 'Yes' in x.values else 'No'
).reset_index()
    
    # Drop duplicates
    filtered_crit_df = filtered_crit_df_unique[['Material Number', 'Is Critical']].drop_duplicates()
    
    # Merge into merged_df, keeping all materials
    final_df = merged_df.merge(filtered_crit_df, on='Material Number', how='left')
    
    # Fill missing values with 'N/A'
    final_df['Is Critical'] = final_df['Is Critical'].fillna('N/A')
    
    return final_df


def create_modify_pr_based_on_poq(row):
    """
    Creates or modifies PR based on Planned Order Quantity (POQ).
    
    Parameters:
    row: Dictionary-like object containing material information
    
    Returns:
    Dictionary with PR_Status, Final_PR_Quantity, and Detailed_Action
    """
    result = {
        'PR_Status': 'No Action',
        'Final_PR_Quantity': 0,
        'Detailed_Action': ''
    }
    
    # Set the PR quantity to POQ
    result['Final_PR_Quantity'] = row['POQ']
    
    # Check if there's an existing PR (OPRQ > 0)
    if row['OPRQ'] == 0:
        # Create new PR
        result['PR_Status'] = 'Create'
        result['Detailed_Action'] = f"Create new PR with quantity: {row['POQ']:.0f}"
    else:
        # Modify existing PR
        result['PR_Status'] = 'Modify'
        direction = "increase" if row['POQ'] > row['OPRQ'] else "decrease"
        change = abs(row['POQ'] - row['OPRQ'])
        result['Detailed_Action'] = f"Modify PR: {direction} by {change:.0f} (from {row['OPRQ']:.0f} to {row['POQ']:.0f})"
    
    return result


def create_modify_pr_based_on_woq(row):
    """
    Creates or modifies PR based on Work Order Quantity (WOQ).
    
    Parameters:
    row: Dictionary-like object containing material information
    
    Returns:
    Dictionary with PR_Status, Final_PR_Quantity, and Detailed_Action
    """
    result = {
        'PR_Status': 'No Action',
        'Final_PR_Quantity': 0,
        'Detailed_Action': ''
    }
    
    # Set the PR quantity to WOQ
    result['Final_PR_Quantity'] = row['WOQ']
    
    # Check if there's an existing PR (OPRQ > 0)
    if row['OPRQ'] == 0:
        # Create new PR
        result['PR_Status'] = 'Create'
        result['Detailed_Action'] = f"Create new PR with quantity: {row['WOQ']:.0f}"
    else:
        # Modify existing PR
        result['PR_Status'] = 'Modify'
        direction = "increase" if row['WOQ'] > row['OPRQ'] else "decrease"
        change = abs(row['WOQ'] - row['OPRQ'])
        result['Detailed_Action'] = f"Modify PR: {direction} by {change:.0f} (from {row['OPRQ']:.0f} to {row['WOQ']:.0f})"
    
    return result


# Example of how to use these functions within the process_flowchart_logic function
def process_flowchart_logic(final_result):
    """
    For materials with 'Is Replenishable'='Yes', 'Is Critical'='NO', 'Is Consumable'='NO',
    implement the flowchart logic for blocks 10-16:
    - Block 10: Has a Work Order?
    - Block 11: Change MRP type to PD (if no work order)
    - Block 12: WOQ >= OPQ + OPRQ + OHQ?
    - Block 13: Do not create PR (if condition in block 12 is not met)
    - Block 14: WO Qty >= ROP?
    - Block 15: Stop (if condition in block 14 is met)
    - Block 16: Create/Modify PR up to WOQ (if condition in block 12 is met)
    
    For materials with 'Is Consumable'='Yes', go directly to block 20 and continue from there.
    
    Parameters:
    final_result: DataFrame with material info including WOQ, OPQ, OPRQ, OHQ, etc.
    
    Returns:
    Complete DataFrame with processed decisions and actions for filtered materials,
    including an "Update MRP Type" column that shows the required MRP type change
    """
    # Make a copy of the original DataFrame to avoid modifying it directly
    filtered_df = final_result.copy()
    # Initialize all required columns with 'N/A' values
    required_columns = [
        'Is Critical', 'Has_Work_Order', 'WO_Covers_Supply', 'WO_Meets_ROP',
        'Action', 'Final_PR_Quantity', 'Detailed Action', 'Update MRP Type','Is_With_Suitable_Stock_Level',
        'Create_SLC','Need_PR_After_SLC'
    ]
    
    for col in required_columns:
        if col not in filtered_df.columns:
            filtered_df[col] = 'N/A'
    filtered_df['PR_Status'] = 'No Action'
    filtered_df['Final_PR_Quantity'] = 0
    
    # Ensure numeric types for calculations on the filtered DataFrame
    numeric_cols = ['WOQ', 'POQ', 'OPQ', 'OPRQ', 'OHQ', 'Reorder Point']
    for col in numeric_cols:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)
        else:
            filtered_df[col] = 0
    
    # Step 9: Is Critical? (Assuming this step is prior to our flowchart)
    filtered_df['Is Critical'] = filtered_df['Is Critical'].fillna('No').astype(str).str.strip().str.capitalize()
    print("filtered df shape is", filtered_df.shape)
    print("filtered df data is", filtered_df.head())
    
    # Filter for materials with normal status
    target_materials = ((filtered_df['Material Status'].fillna('NA').astype(str).str.strip().str.lower() == 'normal'))
    print("target materials are", target_materials)
    
    # Apply flowchart logic to target materials
    for idx, row in filtered_df.iterrows():
        # Initialize the Update MRP Type column with 'No Change'
        filtered_df.at[idx, 'Update MRP Type'] = 'No Change'
        
        # Skip materials with non-normal status
        if (filtered_df.at[idx, 'Material Status'].lower() != 'normal'):
            print("material status is", filtered_df.at[idx, 'Material Status'])
            filtered_df.at[idx, 'Action'] = f"Out of Scope"
            filtered_df.at[idx, 'Detailed Action'] = f"Out of Scope"
            continue
        
        # Process non-replenishable materials (Is Replenishable = No)
        elif (filtered_df.at[idx, 'Is Replenishable'] == 'No'):
            supply_total = filtered_df.at[idx, 'OPQ'] + filtered_df.at[idx, 'OPRQ'] + filtered_df.at[idx, 'OHQ']
            filtered_df.at[idx, 'WO_Covers_Supply'] = 'Yes' if filtered_df.at[idx, 'WOQ'] >= supply_total else 'No'
            if filtered_df.at[idx, 'WO_Covers_Supply'] == 'Yes':
                # Block 6: Create/Modify PR up to POQ (for non-replenishable)
                pr_result = create_modify_pr_based_on_poq(row)
                filtered_df.at[idx, 'PR_Status'] = pr_result['PR_Status']
                filtered_df.at[idx, 'Final_PR_Quantity'] = pr_result['Final_PR_Quantity']
                filtered_df.at[idx, 'Detailed Action'] = pr_result['Detailed_Action']
            else:
                # Block 7: Do not create PR
                filtered_df.at[idx, 'Detailed Action'] = f"WOQ ({filtered_df.at[idx, 'WOQ']:.0f}) is less than total supply ({supply_total:.0f}). Do not create PR. Stop the process"
                filtered_df.at[idx, 'Final_PR_Quantity'] = 0
        
        # Process consumable materials (Is Consumable = Yes)
        elif filtered_df.at[idx, 'Is Consumable'] == 'Yes':
            # Go directly to block 20: Handle stock level check
            filtered_df = handle_stock_level_check(filtered_df, idx)
            
            # Add prefix to detailed action to indicate it's a consumable material
            if filtered_df.at[idx, 'Detailed Action'].startswith("Stock level"):
                filtered_df.at[idx, 'Detailed Action'] = "Consumable material. " + filtered_df.at[idx, 'Detailed Action']
            elif not filtered_df.at[idx, 'Detailed Action'].startswith("Consumable"):
                filtered_df.at[idx, 'Detailed Action'] = "Consumable material. " + filtered_df.at[idx, 'Detailed Action']
        
        # Process non-consumable materials (Is Consumable = No)
        elif filtered_df.at[idx, 'Is Consumable'] == 'No':
            # Block 10: Check if the material has a work order
            if filtered_df.at[idx, "Is Critical"] == 'No':  # We've reached block 10
                if filtered_df.at[idx, 'WOQ'] == 0:  # No work order
                    # Block 11: Change MRP type to PD
                    filtered_df.at[idx, 'PR_Status'] = 'Do not create PR'
                    filtered_df.at[idx, 'Detailed Action'] = f"No work order. Current MRP Type is {filtered_df.at[idx, 'MRP Type']}. Need to change to PD."
                    filtered_df.at[idx, 'Update MRP Type'] = f"Change {filtered_df.at[idx, 'MRP Type']} to PD"
                else:  # Has work order
                    # Block 12: WOQ >= OPQ + OPRQ + OHQ?
                    supply_total = filtered_df.at[idx, 'OPQ'] + filtered_df.at[idx, 'OPRQ'] + filtered_df.at[idx, 'OHQ']
                    filtered_df.at[idx, 'WO_Covers_Supply'] = 'Yes' if filtered_df.at[idx, 'WOQ'] >= supply_total else 'No'
                    
                    if filtered_df.at[idx, 'WO_Covers_Supply'] == 'Yes':
                        # Block 16: Create/Modify PR up to WOQ
                        pr_result = create_modify_pr_based_on_woq(row)
                        filtered_df.at[idx, 'PR_Status'] = pr_result['PR_Status']
                        filtered_df.at[idx, 'Final_PR_Quantity'] = pr_result['Final_PR_Quantity']
                        filtered_df.at[idx, 'Detailed Action'] = pr_result['Detailed_Action']
                        
                        # Block 14: Check WO Qty >= ROP
                        filtered_df.at[idx, 'WO_Meets_ROP'] = 'Yes' if filtered_df.at[idx, 'WOQ'] >= filtered_df.at[idx, 'Reorder Point'] else 'No'
                        
                        if filtered_df.at[idx, 'WO_Meets_ROP'] == 'Yes':
                            # Block 15: Stop
                            filtered_df.at[idx, 'Detailed Action'] += f". WOQ ({filtered_df.at[idx, 'WOQ']:.0f}) meets or exceeds ROP ({filtered_df.at[idx, 'Reorder Point']:.0f}). No further action needed. Stop."
                        else:
                            # If WOQ < ROP, add note about changing MRP type to PD
                            filtered_df.at[idx, 'Detailed Action'] += f". WOQ ({filtered_df.at[idx, 'WOQ']:.0f}) is less than ROP ({filtered_df.at[idx, 'Reorder Point']:.0f}). Consider changing MRP Type to PD."
                            # Add the MRP type update information
                            filtered_df.at[idx, 'Update MRP Type'] = f"Consider changing {filtered_df.at[idx, 'MRP Type']} to PD"
                    else:
                        # Block 13: Do not create PR
                        filtered_df.at[idx, 'PR_Status'] = 'Do not create PR'
                        filtered_df.at[idx, 'Detailed Action'] = f"WOQ ({filtered_df.at[idx, 'WOQ']:.0f}) is less than total supply ({supply_total:.0f}). Do not create PR."
                        filtered_df.at[idx, 'Final_PR_Quantity'] = 0
                        
                        # Block 14: WO Qty >= ROP?
                        filtered_df.at[idx, 'WO_Meets_ROP'] = 'Yes' if filtered_df.at[idx, 'WOQ'] >= filtered_df.at[idx, 'Reorder Point'] else 'No'
                        
                        if filtered_df.at[idx, 'WO_Meets_ROP'] == 'Yes':
                            # Block 15: Stop
                            filtered_df.at[idx, 'Detailed Action'] = f"WOQ ({filtered_df.at[idx, 'WOQ']:.0f}) meets or exceeds ROP ({filtered_df.at[idx, 'Reorder Point']:.0f}). No further action needed. Stop"
                        else:
                            # If WOQ < ROP, go back to Block 11 - change MRP type to PD
                            filtered_df.at[idx, 'Detailed Action'] = f"WOQ ({filtered_df.at[idx, 'WOQ']:.0f}) is less than ROP ({filtered_df.at[idx, 'Reorder Point']:.0f}). Current MRP Type is {filtered_df.at[idx, 'MRP Type']}. Need to change to PD."
                            # Add the MRP type update information
                            filtered_df.at[idx, 'Update MRP Type'] = f"Change {filtered_df.at[idx, 'MRP Type']} to PD"
                            
                    
            # Process critical materials
            elif filtered_df.at[idx, "Is Critical"] == 'Yes':
                # Block 17: Has a Work Order?
                if filtered_df.at[idx, 'WOQ'] > 0:  # Has work order
                    # Block 18: WOQ >= OPQ + OPRQ + OHQ?
                    supply_total = filtered_df.at[idx, 'OPQ'] + filtered_df.at[idx, 'OPRQ'] + filtered_df.at[idx, 'OHQ']
                    filtered_df.at[idx, 'WO_Covers_Supply'] = 'Yes' if filtered_df.at[idx, 'WOQ'] >= supply_total else 'No'
                    
                    if filtered_df.at[idx, 'WO_Covers_Supply'] == 'Yes':
                        # Block 24: Create/Modify PR up to WOQ
                        pr_result = create_modify_pr_based_on_woq(row)
                        filtered_df.at[idx, 'PR_Status'] = pr_result['PR_Status']
                        filtered_df.at[idx, 'Final_PR_Quantity'] = pr_result['Final_PR_Quantity']
                        filtered_df.at[idx, 'Detailed Action'] = pr_result['Detailed_Action']
                        
                        # Block 19: WO Qty >= ROP?
                        filtered_df.at[idx, 'WO_Meets_ROP'] = 'Yes' if filtered_df.at[idx, 'WOQ'] >= filtered_df.at[idx, 'Reorder Point'] else 'No'
                        
                        if filtered_df.at[idx, 'WO_Meets_ROP'] == 'Yes':
                            # Block 25: Stop
                            filtered_df.at[idx, 'Detailed Action'] += f". WOQ ({filtered_df.at[idx, 'WOQ']:.0f}) meets or exceeds ROP ({filtered_df.at[idx, 'Reorder Point']:.0f}). Stop."
                        else:
                            # If WO doesn't meet ROP, go to stock level check
                            filtered_df = handle_stock_level_check(filtered_df, idx)
                    else:
                        # If WO doesn't cover supply, go directly to block 19
                        # Block 19: WO Qty >= ROP?
                        filtered_df.at[idx, 'WO_Meets_ROP'] = 'Yes' if filtered_df.at[idx, 'WOQ'] >= filtered_df.at[idx, 'Reorder Point'] else 'No'
                        
                        if filtered_df.at[idx, 'WO_Meets_ROP'] == 'Yes':
                            # Block 25: Stop
                            filtered_df.at[idx, 'Detailed Action'] = f"WOQ ({filtered_df.at[idx, 'WOQ']:.0f}) meets or exceeds ROP ({filtered_df.at[idx, 'Reorder Point']:.0f}). No further action needed. Stop."
                        else:
                            # If WO doesn't meet ROP, go to stock level check
                            filtered_df = handle_stock_level_check(filtered_df, idx)
                else:  # WOQ = 0, No work order
                    # Go directly to "Is with suitable stock level" check (Block 20)
                    prefix = "WOQ = 0. "
                    filtered_df = handle_stock_level_check(filtered_df, idx)
                    
                    # Add prefix to detailed action to indicate there's no work order
                    if not filtered_df.at[idx, 'Detailed Action'].startswith(prefix):
                        filtered_df.at[idx, 'Detailed Action'] = prefix + filtered_df.at[idx, 'Detailed Action']

    return filtered_df

# Modified handle_stock_level_check function to use the POQ-based PR function
def handle_stock_level_check(filtered_df, idx):
    """
    Implements the flowchart logic for blocks 20-23 and 26:
    - Block 20: Is with suitable stock level?
    - Block 21: Create SLC
    - Block 22: Need PR after SLC?
    - Block 23: Stop
    - Block 26: Create/Modify PR as per planned order quantity
    
    Parameters:
    filtered_df: DataFrame containing the row to process
    idx: Index of the row to process
    
    Returns:
    Updated DataFrame with the processed row
    """
    # Block 20: Check if current = estimated stock levels
    filtered_df.at[idx, 'Is_With_Suitable_Stock_Level'] = 'Yes' if (
        filtered_df.at[idx, 'Reorder Point'] == filtered_df.at[idx, 'Estimated Reorder Point'] and 
        filtered_df.at[idx, 'Max Stock Level'] == filtered_df.at[idx, 'Estimated Max Stock Level']
    ) else 'No'
    
    if filtered_df.at[idx, 'Is_With_Suitable_Stock_Level'] == 'No':
        # Block 21: Create SLC
        filtered_df.at[idx, 'Create_SLC'] = 'Yes'
        filtered_df.at[idx, 'Detailed Action'] = "Create SLC due to unsuitable stock level"
        
        # Block 22: Need PR after SLC?
        filtered_df.at[idx, 'Need_PR_After_SLC'] = 'Yes'  # This would be determined by business logic
        
        if filtered_df.at[idx, 'Need_PR_After_SLC'] == 'Yes':
            # Block 26: Create/Modify PR as per planned order quantity
            row_dict = filtered_df.loc[idx].to_dict()
            pr_result = create_modify_pr_based_on_poq(row_dict)
            filtered_df.at[idx, 'PR_Status'] = pr_result['PR_Status']
            filtered_df.at[idx, 'Final_PR_Quantity'] = pr_result['Final_PR_Quantity']
            filtered_df.at[idx, 'Detailed Action'] += f". {pr_result['Detailed_Action']}"
        else:
            # Block 23: Stop
            filtered_df.at[idx, 'Detailed Action'] = "Create SLC due to unsuitable stock level. No PR needed after SLC. Stop."
    else:
        # Block 20 "Is with suitable stock level?" = Yes
        # Block 26: Create/Modify PR as per planned order quantity
        row_dict = filtered_df.loc[idx].to_dict()
        pr_result = create_modify_pr_based_on_poq(row_dict)
        filtered_df.at[idx, 'PR_Status'] = pr_result['PR_Status']
        filtered_df.at[idx, 'Final_PR_Quantity'] = pr_result['Final_PR_Quantity']
        filtered_df.at[idx, 'Detailed Action'] = f"Stock level suitable. {pr_result['Detailed_Action']}"
    
    return filtered_df

if __name__ == "__main__":
    # Configuration parameters
    project_id = "sipchem"
    dataset_id = "SAP_Functional_Tables"
    input_dir_path = "/opt/shared/"
    output_dir_path = "/home/balaji_thalari_soothsayeranalyti/Sipchem Consumption code/Generated code/"
    credentials_path = input_dir_path + "sipchem-51d75df0f9f7.json"
    
    print("Loading data from BigQuery...")
    
    try:
        # Load all required tables from BigQuery
        tables_to_load = {
            "equipment_criticality_df": "Equipment_Criticality",
            "historical_consumption_df": "Histarical_Consumption",
            "current_stock_df": "Material_Current_Stock",
            "stock_limit_df": "Material_Current_Stocklimit",
            "equipment_mapping_df": "Material_Equipment_Mapping",
            "price_data_df": "Material_Price_Data",
            "material_info_df": "Material_information",
            "open_pr_df": "Open_PR",
            "open_po_df": "Open_Po",
            "po_df": "PO table",
            "planned_orders_df": "Planned_Orders",
            "work_orders_df": "Work_Orders_Table",
            "estimated_stock_limits_df": "Estimated Current Stock Limits",
            "reserv_order_df": "Reservations Table"
        }
        
        # Load all tables in one loop
        loaded_tables = {}
        for df_name, table_name in tables_to_load.items():
            loaded_tables[df_name] = load_bigquery_table(project_id, dataset_id, table_name, credentials_path)
            print(f"Loaded {table_name}")
        
        print("All tables loaded successfully.")
        
        # Extract tables from dictionary for better readability
        planned_orders_df = loaded_tables["planned_orders_df"]
        material_info_df = loaded_tables["material_info_df"]
        reserv_order_df = loaded_tables["reserv_order_df"]
        open_po_df = loaded_tables["open_po_df"]
        open_pr_df = loaded_tables["open_pr_df"]
        current_stock_df = loaded_tables["current_stock_df"]
        stock_limit_df = loaded_tables["stock_limit_df"]
        estimated_stock_limits_df = loaded_tables["estimated_stock_limits_df"]
        equipment_criticality_df = loaded_tables["equipment_criticality_df"]
        historical_consumption_df = loaded_tables["historical_consumption_df"]
        
        # Rename columns for consistency
        planned_orders_df.rename(columns={'Plant Number': 'Plant', 'Quantity of Material': 'POQ'}, inplace=True)
        reserv_order_df.rename(columns={'Different Quantity': 'WOQ'}, inplace=True)
        
        print("Merging data sources...")
        # Merge all data sources with explicit column selection
        merged = (planned_orders_df
            .merge(
                material_info_df[['Material Number', 'Material Status', 'MRP Type', 'Plant']],
                on=['Material Number', 'Plant'], 
                how='left'
            )
            .merge(
                reserv_order_df[['Material Number', 'Plant', 'WOQ']],
                on=['Material Number', 'Plant'], 
                how='left'
            )
            .merge(
                open_po_df[['Material Number', 'Plant', 'OPO Quantity']],
                on=['Material Number', 'Plant'], 
                how='left'
            )
            .rename(columns={'OPO Quantity': 'OPQ'})
            .merge(
                open_pr_df[['Material Number', 'Plant', 'OPR Quantity']],
                on=['Material Number', 'Plant'], 
                how='left'
            )
            .rename(columns={'OPR Quantity': 'OPRQ'})
            .merge(
                current_stock_df[['Material Number', 'Plant', 'On-hand Quantity']],
                on=['Material Number', 'Plant'], 
                how='left'
            )
            .rename(columns={'On-hand Quantity': 'OHQ'})
        )
        
        # Fill missing quantities with 0
        quantity_columns = ['WOQ', 'POQ', 'OPQ', 'OPRQ', 'OHQ']
        merged[quantity_columns] = merged[quantity_columns].fillna(0)
        
        # Add equipment criticality classification
        print("Classifying equipment criticality...")
        merged = classify_equipment_criticality(merged, equipment_criticality_df)
        
        # Merge stock level information
        print("Merging stock level data...")
        merged = merged.merge(
            stock_limit_df[['Material Number', 'Max Stock Level', 'Reorder Point']],
            on='Material Number',
            how='left'
        )
        
        merged = merged.merge(
            estimated_stock_limits_df[['Material Number', 'Estimated Reorder Point', 'Estimated Max Stock Level']],
            on='Material Number',
            how='left'
        )
        
        # Derive additional classifications
        print("Deriving material classifications...")
        merged['Is Replenishable'] = merged['MRP Type'].apply(is_replenishable)
        merged['Has_Work_Order'] = np.where(merged['WOQ'] > 0, 'Yes', 'No')
        
        # Classify consumption patterns
        print("Classifying consumption patterns...")
        merged = classify_consumption_on_replenishable(merged, historical_consumption_df)
        
        # Using the create_modify_pr functions and process_flowchart_logic defined in this script
        
        # Apply flowchart logic for procurement decisions
        print("Applying procurement decision logic...")
        final_result = process_flowchart_logic(merged)
        
        # Select and order final columns
        final_columns = [
            'Planned order created date', 'Planned order Number', 'Plant',
            'Material Number', 'POQ', 'Material Status', 'MRP Type', 'WOQ', 'OPQ',
            'OPRQ', 'OHQ', 'Is Critical', 'Max Stock Level', 'Reorder Point',
            'Estimated Reorder Point', 'Estimated Max Stock Level',
            'Is Replenishable', 'Y-1', 'Y-2', 'Y-3', 'Total Quantity',
            'Is Consumable', 'Has_Work_Order', 'WO_Covers_Supply', 'WO_Meets_ROP', 
            'Detailed Action', 'Update MRP Type', 'Is_With_Suitable_Stock_Level', 
            'Create_SLC', 'Need_PR_After_SLC', 'PR_Status', 'Final_PR_Quantity'
        ]
        
        # Export the results
        output_file = output_dir_path + "final_output.csv"
        print(f"Exporting results to {output_file}")
        final_result[final_columns].to_csv(output_file, index=False)
        print("Processing completed successfully")
        
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()