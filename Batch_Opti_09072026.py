import numpy as np
import torch
import pandas as pd
import time
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.sampling import get_polytope_samples
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from batch_reactor import BatchReactor

#=================================================================================================================================================================
#Reactor Simulation
#=================================================================================================================================================================

reactor = BatchReactor()
params = reactor.params


#=======================================================================================================================================
#Optimization
#=======================================================================================================================================

#boundaries of the parameter domain in v/v (PPO, NAD, GluDH, FDH)=======================================================================
LOWER = np.array([0.0001, 0.0001, 0.0001, 0.0001])
UPPER = np.array([0.8, 0.05, 0.20, 0.20])

w = np.array([1 + params["c_PPOS"] / params["c_AFS"], 1.0, 1.0, 1.0]) #weighting of initial volumes to account for the volume of AF stock solution
v_ges_min = float(np.dot(w, LOWER))                                   #smallest possible total volume
v_ges_max = 1.0                                                       #biggest possible total volume

#Initial design==========================================================================================================================
N_INIT = 100                   #number of initial evaluations
d = 4                          #dimensions of parameter domain

weight = torch.tensor(w, dtype=torch.double)                                                                #weighting from w
bounds = torch.stack([torch.tensor(LOWER, dtype=torch.double), torch.tensor(UPPER, dtype=torch.double),])   #boundaries for sampling
inequality_constraints = [(torch.arange(d), -weight, -1.0)]                                                 #ensuring v*w<=1

#Sampling: Markov chain monte carlo sampler
X_real = get_polytope_samples(
    n=N_INIT,
    bounds=bounds,
    inequality_constraints=inequality_constraints,
    n_burnin=10000,                                 #first 10000 points are discarded, so they are not affected by the starting point
    n_thinning=32,                                  #sampling only every 32nd point
).numpy()

X = np.log10(X_real)                                #log10 of sampling parameters
Y = np.array([reactor.simulate(row) for row in X_real])     #objective values for X_real

#Termination criteria=================================================================================================================================
# #Criterium 1:
#if the hypervolume for hv_window iterations doesnt improve by at least hv_tol*100 % 
hv_history = []
hv_tol = 0.000005
hv_window = 50

#Criterium 2:
#if there are min_pareto_points points on the determined pareto frontier
min_pareto_points = 15

#Criterium 3:
#maximum amount of iterations
max_bo = 500
it = 0

#Bayesian Optimization loop=====================================================================================================================================================
while it < max_bo:
    #training data
    train_x = torch.tensor(X, dtype=torch.double)
    train_y = torch.tensor(Y, dtype=torch.double)

    train_y_objectives = train_y[:, :3]    #STY, TON_E, TON_COF
    train_y_constraint = train_y[:, 3:4]   #X_PPO     
    
    #boundaries transformed into log10
    bounds = torch.tensor(np.array([np.log10(LOWER), np.log10(UPPER)]),dtype=torch.double)

    #combined Gaussian Process------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #single GPs for objectives and constraint combined im mll
    models = []
    for i in range(4):
        gp = SingleTaskGP(
            train_x,
            train_y[:, i:i+1],
            input_transform=Normalize(d=4, bounds=bounds),
            outcome_transform=Standardize(m=1),   #standardize outputs (mean 0, std 1) for numerical stability in the linear objective space
        )    #GP for each objective/constraint
        models.append(gp)
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)                                             #combined marginal likelihood of the 4 GPs
    t0 = time.perf_counter()
    fit_gpytorch_mll(mll)
    print(f"Time GP fitting:{time.perf_counter() - t0}s")

    #ensuring conversion constraint while determining points of pareto frontier
    feasible_mask = (train_y_constraint[:, 0] >= params["X_PPO_target"])
    if feasible_mask.sum() > 0:
        pareto_mask_full = torch.zeros(len(train_y), dtype=torch.bool)
        feasible_idx = feasible_mask.nonzero(as_tuple=True)[0]
        local_pareto = is_non_dominated(train_y_objectives[feasible_mask])
        pareto_mask_full[feasible_idx[local_pareto]] = True
    else:
        pareto_mask_full = is_non_dominated(train_y_objectives)

    pareto_Y_obj = train_y_objectives[pareto_mask_full]
    n_pareto = int(pareto_mask_full.sum())

    #hypervolume calculation
    ref_point = [0, 0, 0] #(train_y_objectives.min(dim=0).values - 1.0).tolist() #reference point
    hv = Hypervolume(torch.tensor(ref_point, dtype=torch.double))
    current_hv = hv.compute(pareto_Y_obj)                                       #current hypervolume of pareto points
    hv_history.append(float(current_hv))                                        #history for stopping criterium 1

    #Acquisition function-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    constraints = [lambda Z: params["X_PPO_target"] - Z[..., 3]]                                                    #conversion constraint: X_PPO_target - X_PPO <=0

    #qNEHVI
    qnehvi = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_x[feasible_mask],
        prune_baseline=True,
        constraints=constraints,
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1, 2]),
    )

    t0 = time.perf_counter()
    candidate, _ = optimize_acqf(
        acq_function=qnehvi,
        bounds=bounds,      #parameter boundaries
        q=1,                #number of points suggested by the acquisition function
        num_restarts=10,    #number of optimization starting points
        raw_samples=128,    #number of evaluations of the acquisition function to choose num_restarts from
    )
    print(f"Time AF optimization: {time.perf_counter() - t0}s")

    candidates = candidate.detach().numpy()

    for x_new in candidates:
        y_new = reactor.simulate(10**x_new)

        X = np.vstack([X, x_new])
        Y = np.vstack([Y, y_new])

    #Termination criteria---------------------------------------------------------------------------------------------------------------------------------------
    if len(hv_history) > hv_window:
        
        hv_old_1 = hv_history[-2]               #hypervolume before current iteration
        hv_old_20 = hv_history[-hv_window-1]    #hyperolume hv_window iterations ago
        hv_new = hv_history[-1]                 #current hypervolume

        rel_improvement_1 = (hv_new - hv_old_1) / abs(hv_old_1)
        rel_improvement_20 = (hv_new - hv_old_20) / abs(hv_old_20) #relative improvement of hypervolume

        print(f"HV improvement over {hv_window} iterations: {100*rel_improvement_20:.10f}%")
        print(f"HV improvement since last iteration: {100*rel_improvement_1:.10f}%")

        if rel_improvement_20 < hv_tol and n_pareto >= min_pareto_points:
            print(
                "Stopping criterion reached."
            )
            break

    print(f"Iteration {it+1} complete")
    print(f"Pareto-Punkte: {n_pareto}")
    print("=====================================================================================")
    it += 1

#====================================================================================================================================================================================================
#Results
#====================================================================================================================================================================================================

#Feasible Pareto front (only points meeting the conversion constraint X_PPO >= target)------------------------------
Y_obj = torch.tensor(Y[:, :3], dtype=torch.double)     #objectives (STY, TON_E, TON_COF)
Y_con = torch.tensor(Y[:, 3], dtype=torch.double)      #constraint (X_PPO)

feasible_mask = (Y_con >= params["X_PPO_target"])
if feasible_mask.sum() > 0:
    mask = np.zeros(len(Y), dtype=bool)
    feasible_idx = feasible_mask.nonzero(as_tuple=True)[0]
    local_pareto = is_non_dominated(Y_obj[feasible_mask])
    mask[feasible_idx[local_pareto].numpy()] = True
else:
    mask = is_non_dominated(Y_obj).numpy()

pareto_X = 10 ** X[mask]                                            #starting volumes
pareto_Y = Y[mask, :3]                                             #objective function values
pareto_PPO_final = np.array(reactor.ppo_final_history)[mask].reshape(-1, 1) #final PPO concentration

#Excel------------------------------------------------------------------------------------------------------------------------------------------------
col_x = ["PPO [mL/mL]", "NAD [mL/mL]", "E_GDH [mL/mL]", "E_FDH [mL/mL]", "PPO_final [mM]"]
col_y = ["STY", "TON_E", "TON_COF"]

df = pd.DataFrame(np.hstack([pareto_X, pareto_PPO_final, pareto_Y]), columns=col_x + col_y,)

#Hypervolume
df_hv = pd.DataFrame({
    "Iteration": range(1, len(hv_history) + 1),
    "Hypervolume": hv_history,
})

with pd.ExcelWriter("batch_pareto_0407.xlsx") as writer:
    df.to_excel(writer, sheet_name="Pareto", index=False)
    df_hv.to_excel(writer, sheet_name="Hypervolume", index=False)

print("Pareto-Frontier saved in batch_pareto_0907.xlsx")
