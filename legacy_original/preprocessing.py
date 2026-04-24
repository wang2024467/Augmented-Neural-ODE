import scanpy as sc
import numpy as np
import scipy.sparse
from fuzzywuzzy import process
import pandas as pd
import scipy.sparse as sp

def gene_preprocessing(adata, num=5000, 
                       log_transform=False, normalize=False,
                       filter_column=None, filter_name=None, 
                       select_by="variance", select_direction = "top"
                       ):
    """
    Preprocess data on gene expression (AnnData object).
    
    Parameters:
    - adata: AnnData object
    - num: Restricted to N number of genes
    - log_transform: If employ normalization (True or False)
    - normalize: If normalize by total counts (True or False)
    - filter_column: Choose column in adata.obs (default: None)
    - filter_name: Choose name in filter_column(default: None)
    - select_by: Method to select top genes ("variance", "mean", "information", "random")
    - select_direction: Direction to select top genes ("top", "bottom")
    
    
    Returns:
    - Preprocessed AnnData object
    """
    
    if filter_column and filter_name:
        if filter_column not in adata.obs.columns:
            print(f"Warning: Column '{filter_column}' not found in adata.obs")
        grouped = adata.obs[adata.obs[filter_column] == filter.name].index.tolist()
        adata = adata[:, grouped]

    if normalize == True:
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray()  
     
        if sp.issparse(adata.X):
            total_counts = np.array(adata.X.sum(axis=1)).flatten()
            adata.X = adata.X.multiply(1e4 / total_counts[:, None])  # Scale data while keeping it sparse
        else:
            total_counts = np.array(adata.X.sum(axis=1)).flatten()
            adata.X = (adata.X / total_counts[:, None]) * 1e4  # Only used if already dense


    if log_transform == True:
        sc.pp.log1p(adata)

    if select_by == "variance":
        if select_direction == "top":
            sc.pp.highly_variable_genes(adata, n_top_genes=num)
            selected_genes = adata.var[adata.var['highly_variable']].index.tolist()
        elif select_direction == "bottom":
            gene_variances = np.array(adata.X.var(axis=0)).flatten()
            gene_index = np.argsort(gene_variances)[:num]
            selected_genes = adata.var_names[gene_index]
            
    elif select_by == "mean":
        if select_direction == "top":
            gene_means = np.array(adata.X.mean(axis=0)).flatten()
            gene_index = np.argsort(-gene_means)[:num]  
            selected_genes = adata.var_names[gene_index]
        elif select_direction == "bottom":
            gene_means = np.array(adata.X.mean(axis=0)).flatten()
            gene_index = np.argsort(gene_means)[:num]  
            selected_genes = adata.var_names[gene_index]
        
    elif select_by == "information":
        if select_direction == "top":
            gene_means = np.array(adata.X.mean(axis=0)).flatten()
            gene_variances = np.array(adata.X.var(axis=0)).flatten()
            information = gene_means * gene_variances 
            gene_index = np.argsort(-information)[:num]
            selected_genes = adata.var_names[gene_index]
        elif select_direction == "bottom":
            gene_means = np.array(adata.X.mean(axis=0)).flatten()
            gene_variances = np.array(adata.X.var(axis=0)).flatten()
            information = gene_means * gene_variances 
            gene_index = np.argsort(information)[:num]
            selected_genes = adata.var_names[gene_index]
    elif select_by == "random":
        selected_genes = np.random.choice(adata.var_names, size=num, replace=False)
    

    else:
        raise ValueError("Invalid select_by method. Choose from 'variance', 'mean', or 'informativeness'.")

    adata = adata[:, selected_genes]

    print(f"Final dataset shape after preprocessing: {adata.shape}")
    
    return adata



def merge_perturbed_genes(original_adata, preprocessed_adata, perturbation_names):
    """
    During the data preprocessing, the most significant genes are extracted, while the embeddings for perturbed genes are required
    in GEARS model. This function is aimed to adding the information of missing perturbed genes back to the preprocessing AnnData.

    Parameters:
    - original_adata: AnnData (before preprocessing)
    - preprocessed_adata: AnnData (after preprocessing)
    - perturbation_names: List of perturbed gene names (e.g., ['m_Rel_3', 'm_Nfkb1_2'])

    Returns:
    - Merged AnnData containing missing perturbed genes.
    """
    
    # Sketch the symbols in original AnnData gene names
    original_adata.var["symbols"] = original_adata.var["0"].str.split("_").str[-1]
    
    # Extract gene symbols from preprocessed AnnData (genes already kept)
    preprocessed_adata.var["symbols"] = preprocessed_adata.var["0"].str.split("_").str[-1]

    # Find the cleaned perturbed names "m_Rel_1": "Rel"
    cleaned_perturbation_genes = [gene.split("_")[1] for gene in perturbation_names if gene != "control"]

    # Match the cleaned perturbed names with symbols in original AnnData
    matched_genes = {}
    for gene in cleaned_perturbation_genes:
        match, score = process.extractOne(gene, original_adata.var["symbols"].values)
        
        # Find the best matches
        if score > 85 and gene.lower() in match.lower():
            matched_genes[gene] = match

    print("Matched Genes:", matched_genes)
    
    existing_genes = preprocessed_adata.var["symbols"].tolist()
    missing_genes = [gene for gene in matched_genes.values() if gene not in existing_genes]

    # Sketch the indices for the removed genes in the original AnnData
    remove_indices = original_adata.var.index[original_adata.var["symbols"].isin(missing_genes)].tolist()

    if not remove_indices:
        print("No missing perturbed genes to recover. Returning preprocessed AnnData.")
        return preprocessed_adata

    # Extract the removed X and var in the AnnData for the missing genes
    adding_X = original_adata[:, remove_indices].X
    adding_var = original_adata.var.loc[remove_indices]

    # Add the information of the missing genes to preprocessed_adata
    new_X = np.hstack((preprocessed_adata.X, adding_X))  # Append columns to AnnData.X
    new_var = pd.concat([preprocessed_adata.var, adding_var])  # Append rows to AnnData.var

    # Update the AnnData with missing genes
    merged_adata = sc.AnnData(X=new_X, obs=preprocessed_adata.obs.copy(), var=new_var.copy())

    print(f"Recovered {len(remove_indices)} perturbed genes and merged them back.")
    
    return merged_adata



def sketch_perturbation_genes(adata, perturbation_names):
    
    # Sketch the symbols in original AnnData gene names
    adata.var["symbols"] = adata.var["0"].str.split("_").str[-1]

    # Find the cleaned perturbed names "m_Rel_1": "Rel"
    cleaned_perturbation_genes = [gene.split("_")[1] for gene in perturbation_names if gene != "control"]

    # Match the cleaned perturbed names with symbols in original AnnData
    matched_genes = {}
    for gene in cleaned_perturbation_genes:
        match, score = process.extractOne(gene, adata.var["symbols"].values)
        
        # Find the best matches
        if score > 85 and gene.lower() in match.lower():
            matched_genes[gene] = match

    print("Matched Genes:", matched_genes)

    # Sketch the indices for the removed genes in the original AnnData
    remove_indices = adata.var.index[adata.var["symbols"].isin(matched_genes.values())].tolist()

    # Extract the removed X and var in the AnnData for the missing genes
    adding_X = adata[:, remove_indices].X
    adding_var = adata.var.loc[remove_indices]

    # Update the AnnData with missing genes
    perturbation_adata = sc.AnnData(X=adding_X, obs=adata.obs.copy(), var=adding_var.copy())

    print(f"Recovered {len(remove_indices)} perturbation genes.")
    
    return perturbation_adata 




def merge_perturbed_genes_same_size(original_adata, preprocessed_adata, perturbation_names):
    """
    Merges missing perturbed genes into preprocessed AnnData while keeping the total number of genes unchanged.
    
    Parameters:
    - original_adata: AnnData (before preprocessing)
    - preprocessed_adata: AnnData (after preprocessing)
    - perturbation_names: List of perturbed gene names (e.g., ['m_Rel_3', 'm_Nfkb1_2'])
    
    Returns:
    - AnnData with merged perturbed genes while maintaining the original size.
    """
    
    # Extract gene symbols
    original_adata.var["symbols"] = original_adata.var["0"].str.split("_").str[-1]
    preprocessed_adata.var["symbols"] = preprocessed_adata.var["0"].str.split("_").str[-1]

    # Extract cleaned perturbed genes (e.g., 'm_Rel_1' → 'Rel')
    cleaned_perturbation_genes = [gene.split("_")[1] for gene in perturbation_names if gene != "control"]

    # Match cleaned perturbed names to original AnnData
    matched_genes = {}
    for gene in cleaned_perturbation_genes:
        match, score = process.extractOne(gene, original_adata.var["symbols"].values)
        if score > 85 and gene.lower() in match.lower():
            matched_genes[gene] = match

    print("Matched Genes:", matched_genes)

    # Identify missing genes not in preprocessed AnnData
    existing_genes = preprocessed_adata.var["symbols"].tolist()
    missing_genes = [gene for gene in matched_genes.values() if gene not in existing_genes]

    # Get indices of missing genes
    remove_indices = original_adata.var.index[original_adata.var["symbols"].isin(missing_genes)].tolist()
    
    if not remove_indices:
        print("No missing perturbed genes to recover. Returning preprocessed AnnData.")
        return preprocessed_adata

    # Extract missing gene data
    adding_X = original_adata[:, remove_indices].X
    adding_var = original_adata.var.loc[remove_indices]

    # Determine how many extra genes we need to remove
    num_genes_to_remove = adding_X.shape[1]  # Number of genes we are adding
    num_genes_preprocessed = preprocessed_adata.shape[1]  # Current gene count
    
    if num_genes_to_remove > num_genes_preprocessed:
        print(f"Too many perturbed genes ({num_genes_to_remove}), truncating.")
        adding_X = adding_X[:, :num_genes_preprocessed]
        adding_var = adding_var.iloc[:num_genes_preprocessed]
    
    if num_genes_to_remove <=1:
        new_X = np.hstack((preprocessed_adata.X, adding_X))  # Append columns to AnnData.X
        new_var = pd.concat([preprocessed_adata.var, adding_var])  # Append rows to AnnData.var

        # Update the AnnData with missing genes
        merged_adata = sc.AnnData(X=new_X, obs=preprocessed_adata.obs.copy(), var=new_var.copy())

        print(f"Recovered {len(remove_indices)} perturbed genes and merged them back.")
        
        return merged_adata
    
    else:
        # Select genes to remove (last few genes in preprocessed AnnData)
        remove_genes_indices = preprocessed_adata.var.index[-num_genes_to_remove:]

        # Remove extra genes to maintain size
        preprocessed_X = np.delete(preprocessed_adata.X, remove_genes_indices, axis=1)
        preprocessed_var = preprocessed_adata.var.drop(remove_genes_indices)

        # Merge missing genes into preprocessed AnnData
        new_X = np.hstack((preprocessed_X, adding_X))
        new_var = pd.concat([preprocessed_var, adding_var])

        # Ensure final shape is the same
        final_gene_count = preprocessed_adata.shape[1]
        new_X = new_X[:, :final_gene_count]
        new_var = new_var.iloc[:final_gene_count]

        # Create final AnnData object
        merged_adata = sc.AnnData(X=new_X, obs=preprocessed_adata.obs.copy(), var=new_var.copy())

        print(f"Recovered {len(remove_indices)} perturbed genes and replaced {num_genes_to_remove} existing genes to maintain size.")
    
        return merged_adata
    