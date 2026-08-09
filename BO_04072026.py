import numpy as np
import torch
import pandas as pd
import time
import os
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.sampling import get_polytope_samples
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch import gen_candidates_torch
from batch_reactor_LeuDH import BatchReactor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=================================================================================================================================================================
#Reactor Simulation
#=================================================================================================================================================================
reactor = BatchReactor()
params = reactor.params

#=======================================================================================================================================
#Optimization
#=======================================================================================================================================
#Results saving=================================================================================================================================================================
def save_checkpoint(X, Y, hv_history, t_iter, reactor, params, filename):
    Y_obj = torch.tensor(Y[:, :3], dtype=torch.double)
    Y_con = torch.tensor(Y[:, 3], dtype=torch.double)

    feasible_mask = (Y_con >= params["X_PPO_target"])
    if feasible_mask.sum() > 0:
        mask = np.zeros(len(Y), dtype=bool)
        feasible_idx = feasible_mask.nonzero(as_tuple=True)[0]
        local_pareto = is_non_dominated(Y_obj[feasible_mask])
        mask[feasible_idx[local_pareto].numpy()] = True
    else:
        mask = is_non_dominated(Y_obj).numpy()

    pareto_X = 10 ** X[mask]
    pareto_Y = Y[mask, :3]
    pareto_PPO_final = np.array(reactor.ppo_final_history)[mask].reshape(-1, 1)

    col_x = ["PPO [mL/mL]", "NAD [mL/mL]", "E_GDH [mL/mL]", "E_FDH [mL/mL]", "PPO_final [mM]"]
    col_y = ["STY", "TON_E", "TON_COF"]
    df = pd.DataFrame(np.hstack([pareto_X, pareto_PPO_final, pareto_Y]), columns=col_x + col_y)

    df_hv = pd.DataFrame({
        "Iteration": range(1, len(hv_history) + 1),
        "Hypervolume": hv_history,
        "Duration [s]": t_iter,
    })

    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, sheet_name="Pareto", index=False)
        df_hv.to_excel(writer, sheet_name="Hypervolume", index=False)

#State of the loop ssaving==========================================================================================================================
STATE_FILE = "bo_state_checkpoint.npz"

def save_state(X, Y, hv_history, t_iter, reactor, it, filename=STATE_FILE, max_retries=5, retry_delay=1.0):
    base, ext = os.path.splitext(filename)
    tmp_filename = f"{base}.tmp{ext}"
    np.savez(
        tmp_filename,
        X=X,
        Y=Y,
        hv_history=np.array(hv_history),
        t_iter=np.array(t_iter),
        ppo_final_history=np.array(reactor.ppo_final_history),
        it=it,
    )
    for attempt in range(max_retries):
        try:
            os.replace(tmp_filename, filename)
            return
        except PermissionError:
            if attempt == max_retries - 1:
                print(f"Warning: {filename} could not be updated after {max_retries} attempts, skipping this checkpoint.")
                return
            time.sleep(retry_delay)


#boundaries of the parameter domain in v/v (PPO, NAD, GluDH, FDH)=======================================================================
LOWER = np.array([0.0001, 0.0005, 0.0001, 0.0001])
UPPER = np.array([0.95, 0.05, 0.20, 0.20])

w = np.array([1 + params["c_TMPS"] / params["c_ForS"], 1.0, 1.0, 1.0]) #weighting of initial volumes to account for the volume of AF stock solution
v_ges_min = float(np.dot(w, LOWER))                                   #smallest possible total volume
v_ges_max = 1.0                                                       #biggest possible total volume

#Initial design or start with old data==========================================================================================================================
N_INIT = 100                   #number of initial evaluations
d = 4                          #dimensions of parameter domain

weight = torch.tensor(w, dtype=torch.double)                                                                #weighting from w
bounds = torch.stack([torch.tensor(LOWER, dtype=torch.double), torch.tensor(UPPER, dtype=torch.double),])   #boundaries for sampling
inequality_constraints = [(torch.arange(d), -weight, -1.0)]                                                 #ensuring v*w<=1

STATE_FILE = "bo_state_checkpoint.npz"

if os.path.exists(STATE_FILE):
    with np.load(STATE_FILE) as data:
        X = data["X"]
        Y = data["Y"]
        hv_history = data["hv_history"].tolist()
        t_iter = data["t_iter"].tolist()
        reactor.ppo_final_history = data["ppo_final_history"].tolist()
        it = int(data["it"])
    print(f"Resuming from checkpoint at iteration {it}")
else:
    X_real = get_polytope_samples(
        n=N_INIT,
        bounds=bounds,
        inequality_constraints=inequality_constraints,
        n_burnin=10000,
        n_thinning=32,
    ).numpy()
    X = np.log10(X_real)
    Y = np.array([reactor.simulate(row) for row in X_real])
    hv_history = []
    t_iter = []
    it = 0

print(len(X), len(Y), len(hv_history), len(reactor.ppo_final_history))
X_PPO_TARGET_WARPED = -np.log(1 - params["X_PPO_target"])   #fixed warped conversion constraint for the GP model

#Termination criteria=================================================================================================================================
# #Criterium 1:
#if the hypervolume for hv_window iterations doesnt improve by at least hv_tol*100 % 
ref_point_hv = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)
hv_tol = 0.0001
hv_window = 200

#Criterium 2:
#if there are min_pareto_points points on the determined pareto frontier
min_pareto_points = 50

#Criterium 3:
#maximum amount of iterations
max_bo = 5000

#Bayesian Optimization loop=====================================================================================================================================================
prev_state = None
while it < max_bo:
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    #training data
    train_x = torch.tensor(X, dtype=torch.double)
    train_y = torch.tensor(Y, dtype=torch.double)

    #warping the conversion constraint to ensure that the GP model can handle the constraint properly
    g_warped = -torch.log(1 - train_y[:, 3])
    train_y = torch.cat([train_y[:, :3], g_warped.unsqueeze(-1)], dim=-1)

    train_y_objectives = train_y[:, :3]    #STY, TON_E, TON_COF
    train_y_constraint = train_y[:, 3:4]   #g = -log(1 - X_PPO)
        
    
    #boundaries transformed into log10
    bounds = torch.tensor(np.array([np.log10(LOWER), np.log10(UPPER)]),dtype=torch.double)

    #combined Gaussian Process------------------------------------------------------------------------------------------------------------------------------------------------------------------
    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=Normalize(d=4, bounds=bounds),
        outcome_transform=Standardize(m=4),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    if prev_state is not None:
        hypers = {k: v for k, v in prev_state.items()
                if "outcome_transform" not in k and "input_transform" not in k}
        model.load_state_dict(hypers, strict=False)

    tgp = time.perf_counter()
    fit_gpytorch_mll(mll)
    print(f"Time GP fitting:{time.perf_counter() - tgp}s")

    prev_state = model.state_dict()

    #ensuring conversion constraint while determining points of pareto frontier
    feasible_mask = (train_y_constraint[:, 0] >= X_PPO_TARGET_WARPED)

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
    ref_point_acq = infer_reference_point(pareto_Y_obj).to(device)
    hv = Hypervolume(ref_point_hv)
    current_hv = hv.compute(pareto_Y_obj)                                       #current hypervolume of pareto points
    hv_history.append(float(current_hv))                                        #history for stopping criterium 1

    #GP model transformed to GPU for AF-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    model_gpu = model.to(device)
    bounds_gpu = bounds.to(device)

    #Acquisition function-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    constraints = [lambda Z: X_PPO_TARGET_WARPED - Z[..., 3]]                                                    #conversion constraint: X_PPO_target - X_PPO <=0

    #qEHVI
    tehvi = time.perf_counter()
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point_acq,
        Y=pareto_Y_obj.to(device),
    )
    qehvi = qLogExpectedHypervolumeImprovement(
        model=model_gpu,
        ref_point=ref_point_acq,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])),
        partitioning=partitioning,
        constraints=constraints,
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1, 2]),
    )
    print(f"Time qEHVI construction: {time.perf_counter() - tehvi}s")

    torch.cuda.synchronize()
    taf = time.perf_counter()
    candidate, _ = optimize_acqf(
        acq_function=qehvi,
        bounds=bounds_gpu,      #parameter boundaries
        q=1,                #number of points suggested by the acquisition function
        num_restarts=10,    #number of optimization starting points
        raw_samples=512,    #number of evaluations of the acquisition function to choose num_restarts from
        gen_candidates=gen_candidates_torch,
        options={"maxiter": 200},
    )
    torch.cuda.synchronize()
    print(f"Time AF optimization: {time.perf_counter() - taf}s")

    candidates = candidate.detach().cpu().numpy()

    for x_new in candidates:
        y_new = reactor.simulate(10**x_new)

        X = np.vstack([X, x_new])
        Y = np.vstack([Y, y_new])

    torch.cuda.synchronize()
    t_iter.append(time.perf_counter() - t0)

    #Termination criteria---------------------------------------------------------------------------------------------------------------------------------------
    if len(hv_history) > hv_window:
        
        hv_old_1 = hv_history[-2]               #hypervolume before current iteration
        hv_old_20 = hv_history[-hv_window-1]    #hyperolume hv_window iterations ago
        hv_new = hv_history[-1]                 #current hypervolume

        rel_improvement_1 = (hv_new - hv_old_1) / abs(hv_old_1)     #relative improvement of hypervolume since last iteration
        rel_improvement_20 = ((hv_new - hv_old_20) / abs(hv_old_20))/hv_window  #median relative improvement of hypervolume over hv_window iterations

        print(f"Median HV improvement over {hv_window} iterations: {100*rel_improvement_20:.10f}%")
        print(f"HV improvement since last iteration: {100*rel_improvement_1:.10f}%")

        if rel_improvement_20 < hv_tol and n_pareto >= min_pareto_points:
            print(
                "Stopping criterion reached."
            )
            save_checkpoint(X, Y, hv_history, t_iter, reactor, params, "batch_pareto_0407_cpugpu_checkpoint.xlsx")
            save_state(X, Y, hv_history, t_iter, reactor, it + 1)
            break

    checkpoint_interval = 5   #save every __ iterations

    if (it + 1) % checkpoint_interval == 0:
        save_checkpoint(X, Y, hv_history, t_iter, reactor, params, "batch_pareto_0407_cpugpu_checkpoint.xlsx")
        save_state(X, Y, hv_history, t_iter, reactor, it + 1)



    print(f"Time Iteration: {time.perf_counter() - t0}s")
    print(f"Iteration {it+1} complete")
    print(f"Pareto-Punkte: {n_pareto}")
    print("=====================================================================================")
    it += 1
