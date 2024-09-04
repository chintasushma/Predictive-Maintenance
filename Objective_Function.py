import numpy as np
from Global_vars import Global_vars
def Objective_Function(Soln):
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln

        Deep_Fused_Features = np.concatenate(
            (Global_vars.RBM_Feat, Global_vars.Autoencoder_Feat, Global_vars.oneDCNN_Feat), axis=-1)
        Weighted_Deep_Fused_Features = Deep_Fused_Features * sol
        # Calculate the correlation coefficients
        correlation_matrix = np.corrcoef(Weighted_Deep_Fused_Features)

        correlation_matrix = np.nan_to_num(correlation_matrix)
        # Extract the upper triangular part of the correlation matrix excluding the diagonal
        upper_triangle = np.triu(correlation_matrix, k=1)

        # Calculate the mean of correlation coefficients
        mean_correlation = np.mean(upper_triangle)

        Fitn[i] = 1 / (mean_correlation)
    return Fitn
