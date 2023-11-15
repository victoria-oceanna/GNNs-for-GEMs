import mackinac as mk
import numpy as np
import cobra
import time
import os
from multiprocessing import Process
from scipy import sparse

def run_with_limited_time(func, args, kwargs, time):
    """Runs function with time limit

    :param func: Function to run
    :param args: Functions args, given as tuple
    :param kwargs: Function keywords, given as dict
    :param time: The time limit in seconds
    :return: True if function ended successfully, False if terminated.
    """
    p = Process(target=func, args=args, kwargs=kwargs)
    p.start()
    p.join(time)
    if p.is_alive():
        p.terminate()
        return False

    return True

def construct_models(metadata, model_index):
    """Submits reconstruction request to ModelSEED

    :param metadata: Dataframe containing acolumn named "Genome ID"
                    which contains a list of PATRIC ID #s
    :param model_index: Index within Dataframe
    :return: None
    """
    mk.reconstruct_modelseed_model(metadata.at[model_id,'Genome ID'])
    return
    
def get_model_info(model_id_list, model_index, current_working_directory): 
    """Gets stoichiometric matrix and metabolite info for models

    :param model_id_list: List of model ids existing in ModelSEED
                          workspace
    :param model_index: Index corresponding to model of given ID #
    :param current_working_directory: Cwd to contain subdirectories
    :return: None; will save npz file for stoichiometric matrix and
             csv file for metabolite names/formulas
    """
    model_ids = model_id_list
    current_id = model_id_list[model_index]
    parent_directory = current_working_directory
    compound_path = os.path.join(str(parent_directory), "Compounds") 
    formula_path = os.path.join(str(parent_directory), "Formulas") 
    stoichiometric_path = os.path.join(str(parent_directory), "Stoichiometric_matrices") 
    
    if os.path.exists(compound_path) != True:
        os.mkdir(compound_path) 
    else: 
        pass
    
    if os.path.exists(formula_path) != True:
        os.mkdir(formula_path)
    else: 
        pass
    
    if os.path.exists(stoichiometric_path) != True:
        os.mkdir(stoichiometric_path) 
    else:
        pass
    
    model = mk.create_cobra_model_from_modelseed_model(str(current_id))
    compound_list = []
    formula_list = []
    for i in range(len(model.metabolites)):
        formula_list.append(model.metabolites[i].formula)
        compound_list.append(model.metabolites[i])
    compound_list = np.asarray(compound_list)
    formula_list = np.asarray(formula_list)
    
    stoichiometric_matrix = cobra.util.array.create_stoichiometric_matrix(model)
    stoichiometric_matrix = sparse.csr_matrix(stoichiometric_matrix)
    np.savetxt(compound_path + '/' + str(current_id) + "_compounds.csv", 
               compound_list, delimiter =',', fmt='%s' )
    np.savetxt(formula_path + '/' + str(current_id) + "_formulas.csv",
               formula_list, delimiter =',', fmt='%s' ) 
    sparse.save_npz(stoichiometric_path + '/' + str(current_id) + "_s.npz", 
                    stoichiometric_matrix)
